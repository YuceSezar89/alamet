"""
C6 — Gölge kasa: her emtia × (4 yöntem + seçili mod) kendi sinyal yönünü sanal olarak izler.

Kaynak Pine: pine/C6_shadow.pine (DEPV2 f_shadow 1437-1473, f_tPnl/f_tR/f_tUsd
1394-1404). Giriş: yön değişiminde kapanıştan; çıkış: yön değişimi/kapanma. Büyüklük başlangıç
kasasından sabit risk (bileşik değil). Net K/Z komisyon + slippage dahil. Sönümlü net K/Z
(`net_d`) her `decay_bars` sinyal barında `decay` ile çarpılır; Otomatik mod bunu kullanır.

Sıra (Pine ile aynı, kapanmış barda): seçici güncellenir (önceki barın gölge durumuyla), sonra
gölge kasa güncellenir (5. hesap seçilen modun yönünü izler). `ss_idx` 0'dan başlar
(Pine'da `timeframe.change` ilk barda false olduğundan sayaç ikinci barda 1 olur).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alamet.selector import SelectorParams, SignalSelector

ACCOUNTS = 5
UNSET = 9
NEVER_LOST = -100000


@dataclass(frozen=True)
class ShadowParams:  # pylint: disable=too-many-instance-attributes
    fee_in: float = 0.02
    fee_out: float = 0.02
    slippage: float = 0.0
    r_unit: float = 1.5
    risk_mode: bool = True
    start_cap: float = 1000.0
    risk_pct: float = 0.5
    max_lev: float = 5.0
    fix_margin: float = 100.0
    leverage: float = 5.0
    decay: float = 0.97
    decay_bars: int = 96

    @property
    def fee_in_e(self) -> float:
        return self.fee_in + self.slippage

    @property
    def fee_out_e(self) -> float:
        return self.fee_out + self.slippage


def trade_usd(params: ShadowParams, notional: float, side: int, entry: float, px: float) -> float:
    return (
        notional * side * (px / entry - 1.0)
        - notional * params.fee_in_e / 100.0
        - notional * (px / entry) * params.fee_out_e / 100.0
    )


def trade_r(params: ShadowParams, side: int, entry: float, risk: float, px: float) -> float:
    if risk <= 0:
        return 0.0
    cost = params.fee_in_e / 100.0 * entry + params.fee_out_e / 100.0 * px
    return (side * (px - entry) - cost) / risk


class ShadowBook:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    def __init__(self, n_symbols: int, params: ShadowParams = ShadowParams()) -> None:
        self.n = n_symbols
        self.p = params
        shape = (n_symbols, ACCOUNTS)
        self.side = np.zeros(shape, dtype=int)
        self.prev = np.full(shape, UNSET, dtype=int)
        self.entry = np.full(shape, np.nan)
        self.notional = np.zeros(shape)
        self.risk = np.full(shape, np.nan)
        self.count = np.zeros(shape, dtype=int)
        self.wins = np.zeros(shape, dtype=int)
        self.net = np.zeros(shape)
        self.net_d = np.zeros(shape)
        self.sum_r = np.zeros(shape)
        self.last_loss = np.full(shape, NEVER_LOST, dtype=int)

    def _size(self, risk_dist: float) -> float:
        p = self.p
        cap = p.start_cap * p.max_lev
        if p.risk_mode:
            return min(p.start_cap * p.risk_pct / 100.0 / risk_dist, cap)
        return min(p.fix_margin * p.leverage, cap)

    def _close_trade(self, k: int, a: int, px: float, ss_idx: int) -> None:
        side, entry = int(self.side[k, a]), self.entry[k, a]
        usd = trade_usd(self.p, self.notional[k, a], side, entry, px)
        self.net[k, a] += usd
        self.net_d[k, a] += usd
        self.sum_r[k, a] += trade_r(self.p, side, entry, self.risk[k, a], px)
        self.count[k, a] += 1
        if usd > 0:
            self.wins[k, a] += 1
        else:
            self.last_loss[k, a] = ss_idx
        self.side[k, a] = 0

    def step(self, ss_idx: int, close: np.ndarray, atr: np.ndarray, dirs: np.ndarray) -> None:
        """dirs: (sembol, 5) — 4 yöntemin güncel yönü ve seçilen modun yönü."""
        if ss_idx % self.p.decay_bars == 0:
            self.net_d *= self.p.decay
        for k in range(self.n):
            px, atr_k = float(close[k]), float(atr[k])
            if math.isnan(px) or math.isnan(atr_k) or atr_k <= 0 or px <= 0:
                continue
            for a in range(ACCOUNTS):
                new = int(dirs[k, a])
                old = int(self.prev[k, a])
                if old == UNSET:
                    self.prev[k, a] = new
                    continue
                if new == old:
                    continue
                if self.side[k, a] != 0:
                    self._close_trade(k, a, px, ss_idx)
                if new != 0:
                    risk = self.p.r_unit * atr_k
                    self.side[k, a] = new
                    self.entry[k, a] = px
                    self.risk[k, a] = risk
                    self.notional[k, a] = self._size(risk / px)
                self.prev[k, a] = new


def _method_dirs(dirs: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(dirs), 0, np.round(np.nan_to_num(dirs))).astype(int)


def run_with_shadow(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
    gates: dict[str, np.ndarray],
    close: np.ndarray,
    atr: np.ndarray,
    selectors: tuple[SelectorParams, ...],
    shadow: ShadowParams = ShadowParams(),
    track_last: bool = True,
) -> dict[str, np.ndarray]:
    """gates: dirs/starts/strs/pends/sigs (bar, sembol, 4); close/atr: (bar, sembol).

    İlk seçici gölge kasanın 5. hesabını (seçili mod) besler; diğerleri yalnızca okur.
    """
    bars, n, _ = gates["dirs"].shape
    book = ShadowBook(n, shadow)
    picks = [SignalSelector(n, params) for params in selectors]
    out: dict[str, list] = {}

    def put(key: str, value) -> None:
        out.setdefault(key, []).append(np.array(value, copy=True))

    for i in range(bars):
        track = track_last or i < bars - 1
        g = {key: value[i] for key, value in gates.items()}
        results = [
            sel.step(
                g["dirs"],
                g["starts"],
                g["strs"],
                g["pends"],
                g["sigs"],
                track,
                book.count[:, :4],
                book.net_d[:, :4],
            )
            for sel in picks
        ]
        if track:
            book.step(
                i,
                close[i],
                atr[i],
                np.concatenate(
                    [_method_dirs(g["dirs"]), picks[0].cb_dir[:, None].astype(int)], axis=1
                ),
            )
        for j, res in enumerate(results):
            for key in ("sig_type", "start", "strength", "pending", "method", "auto"):
                put(f"s{j}_{key}", getattr(res, key))
        for key in ("side", "count", "wins", "net", "net_d", "sum_r", "last_loss"):
            put(f"book_{key}", getattr(book, key))
    result = {key: np.array(values) for key, values in out.items()}
    result["last_entry"], result["last_notional"] = book.entry.copy(), book.notional.copy()
    return result
