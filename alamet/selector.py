"""
C5 — Sinyal modu seçici: 4 yöntemin onay çıktısından sembol başına tek sinyal üretir.

Kaynak Pine: pine/C5_selector.pine (DEPV2 f_gv, f_dirM, f_setSel, f_selUpdate,
f_selOut: 810-935). Modlar: 0-3 tek yöntem (TREND, MOMENTUM, VOLATİLİTE, HACİM), 4 "Aynı anda
conf", 5 "Sıralı conf", 6 "Otomatik (emtia başına en iyi)": gölge kasada (shadow.py) yeterli işlemi
olan ve sönümlü net K/Z'si eşiği aşan en iyi yöntem.

Seçili sinyal yapışkandır: tek yöntem modunda her kapanmış barda yöntemin güncel değerini alır;
conf modlarında koşul sağlanmadıkça önceki seçim aynen kalır. Durum yalnızca kapanmış barda
ilerler (`track=False` yalnızca değerlendirir).

Yöntem başına girdi sütunları (A8 GateResult sırası): yön, başlangıç fiyatı, güç kodu,
bekleyen yön, son onay bar indeksi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MODE_CONF = 4
MODE_SEQ = 5
MODE_AUTO = 6
NO_BAR = -1e9


def parse_seq_order(text: str) -> tuple[int, ...]:
    order: list[int] = []
    for part in text.split(","):
        try:
            value = float(part.replace(" ", ""))
        except ValueError as exc:
            raise ValueError("Sıralı conf sırası 1-4 arası sayılar olmalı") from exc
        if math.isnan(value) or value < 1 or value > 4:
            raise ValueError("Sıralı conf sırası 1-4 arası sayılar olmalı")
        method = int(round(value)) - 1
        if method not in order:
            order.append(method)
    return tuple(order)


@dataclass(frozen=True)
class SelectorParams:  # pylint: disable=too-many-instance-attributes
    mode: int = MODE_CONF
    enabled: tuple[bool, bool, bool, bool] = (True, True, True, True)
    conf_n: int = 2
    seq_order: tuple[int, ...] = (0, 1, 3, 2)
    seq_n: int = 3
    seq_win: int = 24
    auto_min_n: int = 3
    auto_min_net: float = 0.0
    auto_block: bool = True

    def __post_init__(self) -> None:
        if not any(self.enabled):
            raise ValueError("En az bir sinyal yöntemi aktif olmalı")
        if self.mode not in range(7):
            raise ValueError("Mod 0-6 olmalı")

    @property
    def fallback_method(self) -> int:
        for m in range(4):
            if self.enabled[m]:
                return m
        return 3


@dataclass
class SelectorOutput:
    sig_type: np.ndarray
    start: np.ndarray
    strength: np.ndarray
    pending: np.ndarray
    method: np.ndarray
    auto: np.ndarray


class SignalSelector:  # pylint: disable=too-few-public-methods
    def __init__(self, n_symbols: int, params: SelectorParams = SelectorParams()) -> None:
        self.n = n_symbols
        self.p = params
        self.cb_dir = np.zeros(n_symbols, dtype=int)
        self.cb_m = np.zeros(n_symbols, dtype=int)
        self.cb_start = np.full(n_symbols, np.nan)
        self.cb_str = np.zeros(n_symbols, dtype=int)
        self.auto_m = np.full(n_symbols, -1, dtype=int)

    @staticmethod
    def _dir(dirs: np.ndarray, k: int, m: int) -> int:
        value = dirs[k, m]
        return 0 if math.isnan(value) else int(round(value))

    @staticmethod
    def _sig_bar(sigs: np.ndarray, k: int, m: int) -> float:
        value = sigs[k, m]
        return NO_BAR if math.isnan(value) else float(value)

    def _set(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, k: int, m: int, dirs: np.ndarray, starts: np.ndarray, strs: np.ndarray
    ) -> None:
        direction = self._dir(dirs, k, m)
        self.cb_dir[k] = direction
        self.cb_m[k] = m
        if direction != 0:
            self.cb_start[k] = starts[k, m]
            value = strs[k, m]
            self.cb_str[k] = 0 if math.isnan(value) else int(round(value))
        else:
            self.cb_start[k] = np.nan
            self.cb_str[k] = 0

    def _concurrent(self, k: int, dirs: np.ndarray, sigs: np.ndarray) -> int | None:
        n_long = n_short = 0
        best_long = best_short = -1e10
        m_long = m_short = -1
        for m in range(4):
            if not self.p.enabled[m]:
                continue
            direction = self._dir(dirs, k, m)
            sig_bar = self._sig_bar(sigs, k, m)
            if direction == 1:
                n_long += 1
                if sig_bar >= best_long:
                    best_long, m_long = sig_bar, m
            elif direction == -1:
                n_short += 1
                if sig_bar >= best_short:
                    best_short, m_short = sig_bar, m
        need = min(self.p.conf_n, sum(self.p.enabled))
        if n_long >= need and n_long > n_short and m_long >= 0:
            return m_long
        if n_short >= need and n_short > n_long and m_short >= 0:
            return m_short
        return None

    def _sequential(self, k: int, dirs: np.ndarray, sigs: np.ndarray) -> int | None:
        count = 0
        first_dir = 0
        ok = True
        first_bar = math.nan
        prev_bar = -1e10
        last = -1
        for m in self.p.seq_order:
            if count < self.p.seq_n and self.p.enabled[m]:
                direction = self._dir(dirs, k, m)
                sig_bar = self._sig_bar(sigs, k, m)
                if direction == 0:
                    ok = False
                elif count == 0:
                    first_dir = direction
                    first_bar = sig_bar
                elif direction != first_dir or sig_bar < prev_bar:
                    ok = False
                prev_bar = sig_bar
                last = m
                count += 1
        if ok and count >= 1 and last >= 0 and prev_bar - first_bar <= self.p.seq_win:
            return last
        return None

    def _auto(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, k: int, dirs, starts, strs, shadow_cnt, shadow_net
    ) -> None:
        best = -1
        best_net = self.p.auto_min_net
        any_data = False
        for m in range(4):
            if self.p.enabled[m] and shadow_cnt[k, m] >= self.p.auto_min_n:
                any_data = True
                if shadow_net[k, m] > best_net:
                    best_net = shadow_net[k, m]
                    best = m
        self.auto_m[k] = best
        if best >= 0:
            self._set(k, best, dirs, starts, strs)
        elif any_data and self.p.auto_block:
            self.cb_dir[k] = 0
            self.cb_start[k] = np.nan
            self.cb_str[k] = 0
        else:
            self._set(k, self.p.fallback_method, dirs, starts, strs)

    def _update(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, dirs, starts, strs, sigs, shadow_cnt, shadow_net
    ) -> None:
        for k in range(self.n):
            if self.p.mode <= 3:
                self._set(k, self.p.mode, dirs, starts, strs)
                continue
            if self.p.mode == MODE_AUTO:
                self._auto(k, dirs, starts, strs, shadow_cnt, shadow_net)
                continue
            pick = (
                self._concurrent(k, dirs, sigs)
                if self.p.mode == MODE_CONF
                else self._sequential(k, dirs, sigs)
            )
            if pick is not None:
                self._set(k, pick, dirs, starts, strs)

    def _pending(self, k: int, pends: np.ndarray) -> int:
        def value(m: int) -> int:
            v = pends[k, m]
            return 0 if math.isnan(v) else int(round(v))

        if self.p.mode in (MODE_CONF, MODE_SEQ):
            for m in range(4):
                if self.p.enabled[m] and value(m) != 0:
                    return value(m)
            return 0
        return value(int(self.cb_m[k]))

    def step(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        dirs: np.ndarray,
        starts: np.ndarray,
        strs: np.ndarray,
        pends: np.ndarray,
        sigs: np.ndarray,
        track: bool = True,
        shadow_cnt: np.ndarray | None = None,
        shadow_net: np.ndarray | None = None,
    ) -> SelectorOutput:
        if track:
            self._update(dirs, starts, strs, sigs, shadow_cnt, shadow_net)
        return SelectorOutput(
            sig_type=self.cb_dir.astype(float),
            start=self.cb_start.copy(),
            strength=self.cb_str.astype(float),
            pending=np.array([self._pending(k, pends) for k in range(self.n)], dtype=float),
            method=self.cb_m.copy(),
            auto=self.auto_m.copy(),
        )


def run_selector(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dirs: np.ndarray,
    starts: np.ndarray,
    strs: np.ndarray,
    pends: np.ndarray,
    sigs: np.ndarray,
    params: SelectorParams = SelectorParams(),
    track_last: bool = True,
) -> dict[str, np.ndarray]:
    """Girdiler (bar, sembol, 4 yöntem) biçimindedir."""
    bars, n, _ = np.asarray(dirs).shape
    selector = SignalSelector(n, params)
    keys = ("sig_type", "start", "strength", "pending", "method")
    out = {key: [] for key in keys}
    for i in range(bars):
        step = selector.step(
            dirs[i], starts[i], strs[i], pends[i], sigs[i], track_last or i < bars - 1
        )
        for key in keys:
            out[key].append(getattr(step, key))
    return {key: np.array(values) for key, values in out.items()}
