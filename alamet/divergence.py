"""
C4 — Ayrışma takibi, çift ayrışma, yön kilidi ve işlem uygunluğu.

Kaynak Pine: pine/C4_divergence.pine (DEPV2 f_own, f_upd, f_sc, f_lock,
f_lastDbl, f_blockA'nın ikinci yarısı ve f_elig: 1318-1391, 1823-1854, 1863-1880).

"Ayrışma": bir sembol ilk N'e (topArr) girdiği an başlar (giriş fiyatı kaydedilir), ilk N'den
çıkınca ya da sinyal yönü ters dönünce biter. Ayrışma boyunca hareket (kapanış / giriş - 1) × 100
her barda halka tampona (son max_div ayrışma) yazılır; ayrışma bitince son değer kalır.
Çift ayrışma: yön lehine hareket, önceki ayrışmaların ortalaması + z_min × std değerini aşar.
Yön kilidi: son max_div ayrışmanın yukarı/aşağı hareket toplamlarından kazanan taraf
(1 LONG, -1 SHORT, 2 belirsiz, 0 geçmiş yok). Uygun sembol: ilk N'de, ayrışması aktif,
(çift ayrışma), kilit yönüyle aynı.

Orijinaldeki mxA/mnA (ayrışma içi en yüksek/en düşük hareket) yalnızca kendini güncellediği ve
hiçbir yerde okunmadığı için taşınmadı.

Kapanmamış bar: Pine takibi yalnızca `barstate.isconfirmed` iken ilerletir. `track=False` durumu
ilerletmeden mevcut girdilerle yalnızca değerlendirir (TradingView CSV'sinin son satırı hep budur).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DivergenceParams:  # pylint: disable=too-many-instance-attributes
    max_div: int = 10
    min_prev: int = 2
    z_min: float = 1.0
    dbl_pos_only: bool = False
    min_score: float = 0.0
    min_share: float = 50.0
    need_dbl: bool = True
    trade_min_n: int = 1
    max_slots: int = 4
    allow_long: bool = True
    allow_short: bool = True


@dataclass
class StepOutput:  # pylint: disable=too-many-instance-attributes
    count: np.ndarray
    active: np.ndarray
    cur_move: np.ndarray
    last_dbl: np.ndarray
    lock: np.ndarray
    eligible: np.ndarray
    t_rank: np.ndarray
    events: list[tuple] = field(default_factory=list)


def direction(value: float) -> int:
    if math.isnan(value):
        return 0
    return 1 if value == 1.0 else (-1 if value == -1.0 else 0)


class DivergenceTracker:
    def __init__(self, n_symbols: int, params: DivergenceParams = DivergenceParams()) -> None:
        self.n = n_symbols
        self.p = params
        self.count = np.zeros(n_symbols, dtype=int)
        self.active = np.zeros(n_symbols, dtype=bool)
        self.entry = np.full(n_symbols, np.nan)
        self.dir = np.zeros(n_symbols, dtype=int)
        self.res_mv = np.full((n_symbols, params.max_div), np.nan)
        self.res_db = np.zeros((n_symbols, params.max_div), dtype=bool)
        self._events: list[tuple] = []

    def _is_double(self, k: int, dn: int, cur: float, sign: int) -> bool:
        md = self.p.max_div
        total = square = 0.0
        seen = 0
        for j in range(max(1, dn - md + 1), dn):
            value = self.res_mv[k, (j - 1) % md] * sign
            if not math.isnan(value) and (not self.p.dbl_pos_only or value > 0):
                total += value
                square += value * value
                seen += 1
        mean = total / seen if seen > 0 else math.nan
        std = math.sqrt(max(square / seen - mean * mean, 0.0)) if seen > 1 else 0.0
        scaled = cur * sign
        return (
            seen >= self.p.min_prev
            and not math.isnan(scaled)
            and scaled > mean + self.p.z_min * std
            and scaled > 0
        )

    def _update(self, k: int, close: float) -> None:
        move = (close / self.entry[k] - 1.0) * 100.0
        dn = int(self.count[k])
        slot = (dn - 1) % self.p.max_div
        self.res_mv[k, slot] = move
        sign = -1 if self.dir[k] == -1 else 1
        if self._is_double(k, dn, move, sign) and not self.res_db[k, slot]:
            self.res_db[k, slot] = True
            self._events.append(("cift_ayrisma", k, dn, move * sign))

    def _start(self, k: int, close: float, signal_type: float) -> None:
        self.count[k] += 1
        n = int(self.count[k])
        self.active[k] = True
        self.entry[k] = close
        self.dir[k] = 1 if signal_type == 1.0 else -1
        self.res_db[k, (n - 1) % self.p.max_div] = False
        self._update(k, close)
        if n >= 2:
            self._events.append(("ayrisma", k, n, int(self.dir[k])))

    def _track(self, k: int, close: float, signal_type: float, is_top: bool) -> None:
        was_active = bool(self.active[k])
        now = direction(signal_type)
        valid = not math.isnan(close)
        if was_active and is_top and now != 0 and self.dir[k] != now:
            if valid:
                self._update(k, close)
            self.active[k] = False
            was_active = False
        if is_top:
            if not was_active and valid:
                self._start(k, close, signal_type)
            elif was_active and valid:
                self._update(k, close)
        elif was_active:
            if valid:
                self._update(k, close)
            self.active[k] = False

    def last_double(self, k: int) -> bool:
        n = int(self.count[k])
        return bool(self.res_db[k, (n - 1) % self.p.max_div]) if n > 0 else False

    def lock(self, k: int) -> int:
        md = self.p.max_div
        dn = int(self.count[k])
        up = down = 0.0
        seen = 0
        for j in range(max(1, dn - md + 1), dn + 1):
            value = self.res_mv[k, (j - 1) % md]
            if not math.isnan(value) and value != 0.0:
                seen += 1
                if value > 0:
                    up += value
                else:
                    down += -value
        if seen == 0:
            return 0
        total = up + down
        share_up = up / total * 100.0 if total > 0 else 0.0
        share_down = down / total * 100.0 if total > 0 else 0.0
        if up > down and up > self.p.min_score and share_up >= self.p.min_share:
            return 1
        if down > up and down > self.p.min_score and share_down >= self.p.min_share:
            return -1
        return 2

    def _eligible(self, k: int, signal_type: float, is_top: bool) -> bool:
        side = direction(signal_type)
        allowed = self.p.allow_long if side == 1 else self.p.allow_short
        return (
            side != 0
            and allowed
            and is_top
            and bool(self.active[k])
            and self.count[k] >= self.p.trade_min_n
            and (not self.p.need_dbl or self.last_double(k))
            and self.lock(k) == side
        )

    def step(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        close: np.ndarray,
        types: np.ndarray,
        top: np.ndarray,
        idx_long: np.ndarray,
        idx_short: np.ndarray,
        track: bool = True,
    ) -> StepOutput:
        self._events = []
        if track:
            for k in range(self.n):
                self._track(k, float(close[k]), float(types[k]), bool(top[k]))
        eligible = np.array(
            [self._eligible(k, float(types[k]), bool(top[k])) for k in range(self.n)], dtype=bool
        )
        t_rank = np.zeros(self.n, dtype=int)
        for r in range(min(self.p.max_slots, self.n)):
            k_long, k_short = int(idx_long[r]), int(idx_short[r])
            if eligible[k_long] and types[k_long] == 1.0:
                t_rank[k_long] = r + 1
            if eligible[k_short] and types[k_short] == -1.0:
                t_rank[k_short] = r + 1
        cur_move = np.array(
            [
                (
                    self.res_mv[k, (self.count[k] - 1) % self.p.max_div]
                    if self.count[k] > 0
                    else np.nan
                )
                for k in range(self.n)
            ]
        )
        return StepOutput(
            count=self.count.copy(),
            active=self.active.copy(),
            cur_move=cur_move,
            last_dbl=np.array([self.last_double(k) for k in range(self.n)], dtype=bool),
            lock=np.array([self.lock(k) for k in range(self.n)], dtype=int),
            eligible=eligible,
            t_rank=t_rank,
            events=list(self._events),
        )


def run_divergence(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    close: np.ndarray,
    types: np.ndarray,
    top: np.ndarray,
    idx_long: np.ndarray,
    idx_short: np.ndarray,
    params: DivergenceParams = DivergenceParams(),
    track_last: bool = True,
) -> dict[str, np.ndarray]:
    bars, n = np.asarray(close).shape
    tracker = DivergenceTracker(n, params)
    keys = ("count", "active", "cur_move", "last_dbl", "lock", "eligible", "t_rank")
    out = {key: [] for key in (*keys, "events")}
    for i in range(bars):
        step = tracker.step(
            close[i], types[i], top[i], idx_long[i], idx_short[i], track_last or i < bars - 1
        )
        for key in keys:
            out[key].append(getattr(step, key))
        out["events"].append(step.events)
    return {key: (v if key == "events" else np.array(v)) for key, v in out.items()}
