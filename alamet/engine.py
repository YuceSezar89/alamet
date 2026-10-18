"""
D2–D5 — İşlem motoru: çıkışlar, korumalar, giriş (slot, swap, boyut, ısı) ve kasa muhasebesi.

Kaynak: DEPV2 f_blockA (işlem kısmı), f_close, f_closeAll, f_blockB, f_peak, f_unreal, f_heat,
f_qual (entry_quality.py), f_capNot, f_entryMult/f_dynState (sadece kapalı-öğrenme kolu).
Bağımlılık şeması: docs/BAGIMLILIK.md.

Bilerek dışarıda: dinamik kasa kovaları 7-11 (karara girmiyor). Kelly motoru (kelly.py) ve havuz
(pool.py) varsayılan KAPALI; açıkken kapanışta kovalara/emtia net K/Z'sine yazılır.
Kasa değeri her yerde `ks[0] + f_unreal()`. Bar sırası: çıkışlar → korumalar → girişler → zirve.
"""

# pylint: disable=too-many-locals,chained-comparison
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from alamet.entry_quality import EntryQualityParams, entry_flags
from alamet.events import (
    PHASE_ENTRY,
    PHASE_EXIT,
    PHASE_PROTECT,
    CloseEvent,
    DayLossEvent,
    EquityLockEvent,
    EventLog,
    HardDrawdownEvent,
    OpenEvent,
)
from alamet.kelly import KellyEngine, KellyParams, bucket_indices, ease_base, ease_tier, open_side
from alamet.pool import PoolParams, PoolState
from alamet.primitives import ge, le
from alamet.stats import DYN_BUCKETS, StatsParams, TradeStats

CAP_RISK = "risk"
HUGE = 1e10
NEVER_LOST = -100000

REASON_TP, REASON_GIVE, REASON_SIGNAL, REASON_RANK = 0, 1, 2, 3
REASON_STOP, REASON_TIME, REASON_HARD_DD, REASON_DAY, REASON_LOCK = 4, 5, 6, 7, 8
STATUS_TP_LOCK, STATUS_RANK, STATUS_SIGNAL, STATUS_STOP = 1, 2, 3, 4


@dataclass(frozen=True)
class TradeParams:  # pylint: disable=too-many-instance-attributes
    start_cap: float = 1000.0
    risk_mode: bool = True
    risk_pct: float = 0.5
    max_lev: float = 5.0
    fix_margin: float = 100.0
    leverage: float = 5.0
    fee_in: float = 0.02
    fee_out: float = 0.02
    slippage: float = 0.0
    r_unit: float = 1.5
    use_tp: bool = False
    tp_r: float = 2.0
    use_stop: bool = False
    stop_r: float = 4.0
    use_time_stop: bool = False
    max_hold: int = 96
    use_give: bool = True
    give_arm: float = 1.5
    give_frac: float = 0.5
    max_slots: int = 4
    swap_edge: float = 0.3
    stay_reset_bars: int = 3
    re_entry: bool = True
    rank2_mult: float = 0.5
    cap_on: bool = True
    cap_pct: float = 1.0
    cap_risk_mode: bool = True
    max_exp: float = 3.0
    dd_hard_on: bool = True
    dd_hard_pct: float = 15.0
    dd_hard_bars: int = 96
    day_loss_on: bool = True
    day_loss_pct: float = 3.0
    day_loss_close: bool = False
    lock_on: bool = True
    lock_arm: float = 3.0
    lock_frac: float = 0.4
    lock_bars: int = 48
    neg_mult: float = 0.6
    allow_long: bool = True
    allow_short: bool = True
    dr_on: bool = False
    dr_min_m: float = 0.4
    dr_max_m: float = 2.0
    dr_pow: float = 1.0
    dr_block_on: bool = False
    dr_block_pct: float = 90.0
    use_dyn: bool = False
    dyn_min_n: float = 5.0
    dyn_decay: float = 0.97
    dyn_decay_bars: int = 96
    dyn_gain: float = 1.5
    dyn_min_mult: float = 0.25
    dyn_max_mult: float = 2.0
    use_kill: bool = True
    kill_n: float = 6.0
    kill_r: float = 0.5
    dd_throttle: float = 5.0
    dd_mult: float = 0.5
    dd_halt: float = 10.0
    halt_bars: int = 48
    streak_n: int = 3
    streak_mult: float = 0.5
    quality: EntryQualityParams = EntryQualityParams()
    kelly: KellyParams = KellyParams()
    pool: PoolParams = PoolParams()
    hold_short: int = 3
    hold_long: int = 12
    log_size: int = 8

    @property
    def fee_in_e(self) -> float:
        return self.fee_in + self.slippage

    @property
    def fee_out_e(self) -> float:
        return self.fee_out + self.slippage


@dataclass
class BarInputs:  # pylint: disable=too-many-instance-attributes
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    types: np.ndarray
    starts: np.ndarray
    power: np.ndarray
    scores: np.ndarray
    eligible: np.ndarray
    t_rank: np.ndarray
    idx_long: np.ndarray
    idx_short: np.ndarray
    daily_open: np.ndarray
    weekly_open: np.ndarray
    monthly_open: np.ndarray
    day_changed: bool = False
    dr_pct: np.ndarray | None = None
    div_count: np.ndarray | None = None
    strengths: np.ndarray | None = None
    methods: np.ndarray | None = None
    dr_ratio: np.ndarray | None = None


@dataclass
class Trade:  # pylint: disable=too-many-instance-attributes
    symbol: int
    side: int
    open_bar: int
    entry: float
    risk: float
    notional: float
    rank: int
    close_bar: int = -1
    exit: float = math.nan
    usd: float = math.nan
    r: float = math.nan
    reason: int = -1
    div_n: int = 0
    kelly_ix: tuple = ()
    method: int = 0
    strength: int = 0
    op_side: int = 0
    tier: int = 57
    slot: int = -1
    trade_id: int = 0
    mfe: float = 0.0
    hold: int = 0
    pnl: float = math.nan
    signal_pct: float = 0.0
    open_time: int = 0
    close_time: int = 0


@dataclass
class EngineState:  # pylint: disable=too-many-instance-attributes
    n: int
    start_cap: float
    active: np.ndarray = field(init=False)
    side: np.ndarray = field(init=False)
    entry: np.ndarray = field(init=False)
    risk: np.ndarray = field(init=False)
    take: np.ndarray = field(init=False)
    notional: np.ndarray = field(init=False)
    mfe: np.ndarray = field(init=False)
    open_bar: np.ndarray = field(init=False)
    stay_used: np.ndarray = field(init=False)
    off_count: np.ndarray = field(init=False)
    last_loss: np.ndarray = field(init=False)
    trade_ref: list = field(init=False)
    ks: np.ndarray = field(init=False)
    dg: np.ndarray = field(init=False)
    kl: np.ndarray = field(init=False)
    bne: np.ndarray = field(init=False)
    bsr: np.ndarray = field(init=False)
    dyn: np.ndarray = field(init=False)
    day_pending: bool = False

    def __post_init__(self) -> None:
        n = self.n
        self.active = np.zeros(n, dtype=bool)
        self.side = np.zeros(n, dtype=int)
        self.entry = np.full(n, np.nan)
        self.risk = np.full(n, np.nan)
        self.take = np.full(n, np.nan)
        self.notional = np.zeros(n)
        self.mfe = np.zeros(n)
        self.open_bar = np.zeros(n, dtype=int)
        self.stay_used = np.zeros(n, dtype=bool)
        self.off_count = np.zeros(n, dtype=int)
        self.last_loss = np.full(n, NEVER_LOST, dtype=int)
        self.trade_ref = [None] * n
        self.ks = np.array([self.start_cap, self.start_cap] + [0.0] * 8)
        self.dg = np.array([self.start_cap, 0.0, self.start_cap, 0.0, 0.0, 0.0])
        self.kl = np.array([0.0, self.start_cap, 0.0])
        self.bne = np.zeros(DYN_BUCKETS)
        self.bsr = np.zeros(DYN_BUCKETS)
        self.dyn = np.array([self.start_cap, 0.0, 0.0])


class TradeEngine:  # pylint: disable=too-many-public-methods,too-many-instance-attributes
    def __init__(self, n_symbols: int, params: TradeParams = TradeParams()) -> None:
        self.p = params
        self.n = n_symbols
        self.s = EngineState(n_symbols, params.start_cap)
        self.trades: list[Trade] = []
        self.events: EventLog | None = None
        self.phase = PHASE_EXIT
        self.kelly = KellyEngine(params.kelly, n_symbols)
        self.ease_base = ease_base(n_symbols)
        self.pool = PoolState(n_symbols, params.pool, params.start_cap)
        self.stats = TradeStats(
            n_symbols, StatsParams(params.hold_short, params.hold_long, params.log_size)
        )

    def trade_usd(self, notional: float, side: int, entry: float, px: float) -> float:
        p = self.p
        return (
            notional * side * (px / entry - 1.0)
            - notional * p.fee_in_e / 100.0
            - notional * (px / entry) * p.fee_out_e / 100.0
        )

    def trade_r(self, side: int, entry: float, risk: float, px: float) -> float:
        if not risk > 0:
            return 0.0
        cost = self.p.fee_in_e / 100.0 * entry + self.p.fee_out_e / 100.0 * px
        return (side * (px - entry) - cost) / risk

    def trade_pnl(self, side: int, entry: float, px: float) -> float:
        """İşlem getirisi % (f_tPnl): ücret ve kayma dahil."""
        p = self.p
        return side * (px / entry - 1.0) * 100.0 - p.fee_in_e - p.fee_out_e * (px / entry)

    def open_direction(self, close: np.ndarray, side: int) -> np.ndarray:
        """`f_openDir`: açık işlem sayısı, gerçekleşmemiş $, R toplamı, getiri % toplamı."""
        s = self.s
        count, usd, r_sum, pnl_sum = 0, 0.0, 0.0, 0.0
        for k in range(self.n):
            if s.active[k] and s.side[k] == side and not math.isnan(close[k]):
                count += 1
                usd += self.trade_usd(s.notional[k], side, s.entry[k], close[k])
                r_sum += self.trade_r(side, s.entry[k], s.risk[k], close[k])
                pnl_sum += self.trade_pnl(side, s.entry[k], close[k])
        return np.array([count, usd, r_sum, pnl_sum], dtype=float)

    def unrealised(self, close: np.ndarray) -> float:
        s, total = self.s, 0.0
        for k in range(self.n):
            if s.active[k] and not math.isnan(close[k]):
                total += self.trade_usd(s.notional[k], s.side[k], s.entry[k], close[k])
        return total

    def heat(self, close: np.ndarray) -> float:
        s, total = self.s, 0.0
        for k in range(self.n):
            if s.active[k]:
                px, ent = close[k], s.entry[k]
                bad = math.isnan(px) or math.isnan(ent) or ent <= 0
                total += s.notional[k] * (1.0 if bad else px / ent)
        return total

    def equity(self, close: np.ndarray) -> float:
        return float(self.s.ks[0] + self.unrealised(close))

    def emit(self, kind, step_i: int, **fields) -> None:
        if self.events is not None:
            self.events.add(kind, step_i, self.phase, **fields)

    def close_trade(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, k: int, px: float, status: int, reason: int, step_i: int
    ) -> None:
        s = self.s
        side, entry, notional = int(s.side[k]), s.entry[k], s.notional[k]
        usd = self.trade_usd(notional, side, entry, px)
        r = self.trade_r(side, entry, s.risk[k], px)
        fee = (
            notional * self.p.fee_in_e / 100.0 + notional * (px / entry) * self.p.fee_out_e / 100.0
        )
        s.ks[0] += usd
        s.ks[5] += fee
        if usd > 0:
            s.ks[3] += usd
            s.ks[8] = 0.0
        else:
            s.last_loss[k] = step_i
            s.ks[4] -= usd
            s.ks[8] += 1.0
            s.ks[9] = max(s.ks[9], s.ks[8])
        s.ks[6] = max(s.ks[6], usd)
        s.ks[7] = min(s.ks[7], usd)
        trade = s.trade_ref[k]
        self.pool.add(k, usd)
        if trade.kelly_ix:
            self.kelly.accumulate(list(trade.kelly_ix), r)
        hold = step_i - int(s.open_bar[k])
        for bucket in self.dyn_write_buckets(side, trade, hold):
            s.bne[bucket] += 1.0
            s.bsr[bucket] += r
        trade.close_bar, trade.exit, trade.usd, trade.r, trade.reason = step_i, px, usd, r, reason
        trade.mfe, trade.hold = float(s.mfe[k]), hold
        trade.pnl = self.trade_pnl(side, entry, px)
        self.stats.record(trade, usd, trade.pnl, reason, usd <= 0 and side * (px / entry - 1.0) > 0)
        self.stats.log.close(trade.slot, trade.trade_id, px, status)
        if self.events is not None:
            self.events.add(
                CloseEvent, step_i, self.phase, symbol=k, side=side, trade_id=trade.trade_id,
                status=status, reason=reason, entry=entry, exit=px, r=r, pnl_pct=trade.pnl,
                usd=usd, hold=hold,
            )  # fmt: skip
        s.active[k] = False
        if status == STATUS_RANK:
            s.stay_used[k] = False

    def close_all(self, close: np.ndarray, status: int, reason: int, step_i: int) -> int:
        count = 0
        for k in range(self.n):
            if self.s.active[k] and not math.isnan(close[k]):
                self.close_trade(k, close[k], status, reason, step_i)
                count += 1
        return count

    def manage_open(self, step_i: int, b: BarInputs) -> None:  # pylint: disable=too-many-locals
        p, s = self.p, self.s
        for k in range(self.n):
            if not s.active[k] or math.isnan(b.high[k]) or math.isnan(b.low[k]):
                continue
            d, risk, entry = int(s.side[k]), s.risk[k], s.entry[k]
            if not math.isnan(risk) and risk > 0:
                extreme = b.high[k] if d == 1 else b.low[k]
                s.mfe[k] = max(s.mfe[k], self.trade_r(d, entry, risk, extreme))
            close = b.close[k]
            tp = s.take[k]
            hit_tp = (
                p.use_tp and not math.isnan(tp) and (b.high[k] >= tp if d == 1 else b.low[k] <= tp)
            )
            stop_px = (
                entry - d * p.stop_r * risk
                if p.use_stop and not math.isnan(risk) and risk > 0
                else math.nan
            )
            hit_stop = not math.isnan(stop_px) and (
                b.low[k] <= stop_px if d == 1 else b.high[k] >= stop_px
            )
            hit_time = p.use_time_stop and step_i - s.open_bar[k] >= p.max_hold
            signal_exit = not math.isnan(b.types[k]) and not math.isnan(close) and b.types[k] == -d
            cur_r = math.nan
            if not math.isnan(close) and not math.isnan(risk) and risk > 0:
                cur_r = self.trade_r(d, entry, risk, close)
            hit_give = (
                p.use_give
                and not math.isnan(cur_r)
                and s.mfe[k] >= p.give_arm
                and cur_r <= s.mfe[k] * (1.0 - p.give_frac)
            )
            if hit_stop:
                gap = b.open[k]
                use_open = not math.isnan(gap) and (gap < stop_px if d == 1 else gap > stop_px)
                self.close_trade(k, gap if use_open else stop_px, STATUS_STOP, REASON_STOP, step_i)
            elif hit_tp:
                self.close_trade(k, tp, STATUS_TP_LOCK, REASON_TP, step_i)
            elif hit_give:
                self.close_trade(k, close, STATUS_TP_LOCK, REASON_GIVE, step_i)
            elif signal_exit:
                self.close_trade(k, close, STATUS_SIGNAL, REASON_SIGNAL, step_i)
            elif hit_time and not math.isnan(close):
                self.close_trade(k, close, STATUS_STOP, REASON_TIME, step_i)

    @staticmethod
    def dyn_buckets(side: int, rank: int, div_n: int) -> tuple[int, int, int]:
        group = 4 if div_n <= 3 else (5 if div_n <= 5 else 6)
        return (0 if side == 1 else 1, 2 if rank == 1 else 3, group)

    def dyn_write_buckets(self, side: int, trade: Trade, hold: int) -> tuple[int, ...]:
        """Kapanışta yazılan 5 dinamik kasa kovası: yön, sıra, ayrışma, tutma süresi, open yönü.
        Kasa kararı yalnızca ilk üçünü okur (`dyn_score`); 7-11 kayıt içindir."""
        return (
            *self.dyn_buckets(side, trade.rank, trade.div_n),
            self.stats.hold_bucket(hold, 7),
            11 if trade.op_side == -1 else 10,
        )

    def dyn_update(self, step_i: int, close: np.ndarray) -> None:
        p, s = self.p, self.s
        eq = self.equity(close)
        if s.dyn[1] > 0.0 and step_i >= s.dyn[1]:
            s.dyn[0], s.dyn[1] = eq, 0.0
        s.dyn[0] = max(s.dyn[0], eq)
        drawdown = (s.dyn[0] - eq) / s.dyn[0] * 100.0 if s.dyn[0] > 0 else 0.0
        if p.use_dyn and drawdown >= p.dd_halt and s.dyn[1] == 0.0:
            s.dyn[1] = float(step_i + p.halt_bars)
            s.dyn[2] += 1.0
        if p.use_dyn and step_i % p.dyn_decay_bars == 0:
            s.bne *= p.dyn_decay
            s.bsr *= p.dyn_decay

    def dyn_raw(self, side: int, rank: int, div_n: int) -> tuple[float, bool]:
        """Dinamik kasa skoru (R) ve kova blok bayrağı."""
        p, s = self.p, self.s
        score, kill = 0.0, False
        for bucket in self.dyn_buckets(side, rank, div_n):
            seen, total = s.bne[bucket], s.bsr[bucket]
            score += total / (seen + p.dyn_min_n) / 3.0
            if p.use_kill and bucket >= 2 and ge(seen, p.kill_n) and le(total / seen, -p.kill_r):
                kill = True
        return score, kill

    def dyn_score(self, side: int, rank: int, div_n: int) -> tuple[float, bool]:
        p = self.p
        score, kill = self.dyn_raw(side, rank, div_n)
        mult = max(p.dyn_min_mult, min(p.dyn_max_mult, 1.0 + p.dyn_gain * score))
        return mult, kill

    def protections(self, step_i: int, b: BarInputs) -> tuple[bool, float]:
        p, s = self.p, self.s
        eq = self.equity(b.close)
        if s.day_pending:
            s.dg[0], s.dg[5], s.day_pending = eq, 0.0, False
        if s.dg[1] > 0.0 and step_i >= s.dg[1]:
            s.dg[1], s.dg[2] = 0.0, eq
        if p.dd_hard_on:
            in_halt = s.dg[1] > 0.0 and step_i < s.dg[1]
            if not in_halt:
                s.dg[2] = max(s.dg[2], eq)
                dd = (s.dg[2] - eq) / s.dg[2] * 100.0 if s.dg[2] > 0 else 0.0
                if dd >= p.dd_hard_pct:
                    closed = self.close_all(b.close, STATUS_STOP, REASON_HARD_DD, step_i)
                    self.emit(
                        HardDrawdownEvent, step_i, drawdown_pct=dd, closed=closed,
                        wait_bars=p.dd_hard_bars,
                    )  # fmt: skip
                    s.dg[1] = float(step_i + p.dd_hard_bars)
                    s.dg[3] += 1.0
                    s.dg[2] = self.equity(b.close)
        if p.day_loss_on and s.dg[5] == 0.0:
            eq_day = self.equity(b.close)
            loss = (s.dg[0] - eq_day) / s.dg[0] * 100.0 if s.dg[0] > 0 else 0.0
            if loss >= p.day_loss_pct:
                s.dg[5] = 1.0
                s.dg[4] += 1.0
                closed = 0
                if p.day_loss_close:
                    closed = self.close_all(b.close, STATUS_STOP, REASON_DAY, step_i)
                self.emit(
                    DayLossEvent, step_i, loss_pct=loss, closed=closed,
                    closes_trades=p.day_loss_close,
                )  # fmt: skip
        if p.lock_on:
            eq_lock = self.equity(b.close)
            if s.kl[0] > 0.0 and step_i >= s.kl[0]:
                s.kl[0], s.kl[1] = 0.0, eq_lock
            in_lock = s.kl[0] > 0.0 and step_i < s.kl[0]
            if not in_lock:
                s.kl[1] = max(s.kl[1], eq_lock)
                peak_profit = s.kl[1] - p.start_cap
                armed = peak_profit >= p.start_cap * p.lock_arm / 100.0
                if armed and eq_lock <= s.kl[1] - peak_profit * p.lock_frac:
                    closed = self.close_all(b.close, STATUS_STOP, REASON_LOCK, step_i)
                    self.emit(
                        EquityLockEvent, step_i, peak_profit=peak_profit, closed=closed,
                        wait_bars=p.lock_bars,
                    )  # fmt: skip
                    s.kl[0] = float(step_i + p.lock_bars)
                    s.kl[2] += 1.0
                    s.kl[1] = self.equity(b.close)
        halted = (
            (s.dg[1] > 0.0 and step_i < s.dg[1])
            or s.dg[5] > 0.0
            or (p.lock_on and s.kl[0] > 0.0 and step_i < s.kl[0])
        )
        eq_now = self.equity(b.close)
        peak = max(s.dyn[0], eq_now)
        drawdown = max(0.0, (peak - eq_now) / peak * 100.0) if peak > 0 else 0.0
        halted = halted or (p.use_dyn and s.dyn[1] > 0.0 and step_i < s.dyn[1])
        return halted, drawdown

    def entry_multiplier(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, eq: float, side: int, rank: int, div_n: int, drawdown: float
    ) -> tuple[float, bool]:
        """Dinamik kasa + kasa kâr kilidi çarpanı ve kova bloğu (DEPV2 f_entryMult)."""
        p, s = self.p, self.s
        mult, kill = 1.0, False
        if p.use_dyn:
            scale, kill = self.dyn_score(side, rank, div_n)
            throttle = p.dd_mult if drawdown >= p.dd_throttle else 1.0
            streak = p.streak_mult if s.ks[8] >= p.streak_n else 1.0
            replaced = p.kelly.on and p.kelly.replace
            mult = (1.0 if replaced else scale * throttle) * streak
        if p.lock_on and eq < p.start_cap:
            mult *= p.neg_mult
        return mult, kill

    @staticmethod
    def entry_meta(
        b: BarInputs, k: int, side: int, price: float, percentile: float, base: int
    ) -> tuple[int, int, int, int]:
        """Girişte bilinen sınıflandırma: (yöntem, güç kodu, günlük open yönü, Devisso dilimi)."""
        strength = 0 if b.strengths is None or math.isnan(b.strengths[k]) else int(b.strengths[k])
        method = 0 if b.methods is None else int(b.methods[k])
        return (
            method,
            strength,
            open_side(side, price, b.daily_open[k]),
            ease_tier(percentile, base),
        )

    def kelly_indices(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, k: int, side: int, rank: int, div_n: int, meta: tuple[int, int, int, int]
    ) -> tuple:
        if not self.p.kelly.on:
            return ()
        method, strength, op_side, tier = meta
        return tuple(bucket_indices(k, side, rank, div_n, strength, method, op_side, tier))

    def dr_ease(self, percentile: float) -> tuple[float, bool]:
        """Devisso kolaylık çarpanı ve DR blok (DEPV2 f_drEase)."""
        p = self.p
        if not p.dr_on:
            return 1.0, False
        known = not math.isnan(percentile)
        ease = 1.0 - percentile / 100.0 if known else 0.5
        mult = p.dr_min_m + (p.dr_max_m - p.dr_min_m) * max(ease, 0.0) ** p.dr_pow
        return mult, p.dr_block_on and known and percentile > p.dr_block_pct

    def cap_notional(self, eq: float, dist: float) -> float:
        p = self.p
        if not p.cap_on:
            return HUGE
        return (
            eq * p.cap_pct / 100.0 / dist if p.cap_risk_mode else eq * p.cap_pct / 100.0 * p.max_lev
        )

    def base_notional(self, eq: float, dist: float, mult: float) -> float:
        p = self.p
        if p.risk_mode:
            return min(eq * p.risk_pct / 100.0 * mult / dist, eq * p.max_lev)
        return min(p.fix_margin * p.leverage * mult, eq * p.max_lev)

    def weakest_open(self, side: int, b: BarInputs) -> tuple[int, float]:
        s = self.s
        weakest, lowest = -1, 1e10
        for j in range(self.n):
            if s.active[j] and s.side[j] == side:
                in_target = b.t_rank[j] > 0 and b.types[j] == side
                if not in_target:
                    score = b.scores[j]
                    key = -1e9 if math.isnan(score) else score
                    if key < lowest:
                        lowest, weakest = key, j
        return weakest, lowest

    def open_count(self, side: int) -> int:
        return int(((self.s.side == side) & self.s.active).sum())

    def open_trade(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, k, side, rank, price, risk, notional, step_i, take=math.nan, div_n=0, kelly_ix=(),
        meta=None, signal_pct=0.0,
    ) -> None:  # fmt: skip
        s = self.s
        meta = meta or (0, 0, 0, self.ease_base + 1)
        s.active[k], s.side[k], s.entry[k], s.risk[k] = True, side, price, risk
        s.notional[k], s.take[k], s.mfe[k], s.open_bar[k] = notional, take, 0.0, step_i
        s.stay_used[k] = True
        trade = Trade(k, side, step_i, price, risk, notional, rank, div_n=div_n, kelly_ix=kelly_ix)
        trade.method, trade.strength, trade.op_side, trade.tier = meta
        trade.signal_pct = signal_pct
        log = self.stats.log
        self.stats.id_seq += 1
        trade.trade_id = self.stats.id_seq
        trade.slot = log.open(trade.trade_id, k, side, price, risk, take, signal_pct, notional)
        s.trade_ref[k] = trade
        self.trades.append(trade)

    def try_entry(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
        self, step_i: int, b: BarInputs, side: int, rank: int, k: int, halted: bool, drawdown: float
    ) -> None:
        p, s = self.p, self.s
        price = b.close[k]
        if math.isnan(price) or halted:
            return
        risk = p.r_unit * b.atr[k]
        if math.isnan(risk) or not risk > 0:
            return
        take = price + side * p.tp_r * risk if p.use_tp else math.nan
        eq = self.equity(b.close)
        percentile = math.nan if b.dr_pct is None else float(b.dr_pct[k])
        dr_mult, dr_block = self.dr_ease(percentile)
        div_n = 0 if b.div_count is None else int(b.div_count[k])
        mult, kill = self.entry_multiplier(eq, side, rank, div_n, drawdown)
        dyn_mult = mult
        meta = self.entry_meta(b, k, side, price, percentile, self.ease_base)
        kelly_ix = self.kelly_indices(k, side, rank, div_n, meta)
        kelly_mult, kelly_block = self.kelly.evaluate(
            list(kelly_ix), drawdown, p.dd_hard_pct, p.risk_pct
        )
        mult *= kelly_mult * dr_mult
        dist = risk / price
        flags = entry_flags(
            side, b.starts[k], price, b.atr[k], b.power[k], b.daily_open[k],
            b.weekly_open[k], b.monthly_open[k], step_i - s.last_loss[k], p.quality,
        )  # fmt: skip
        rank_mult = 1.0 if rank == 1 else p.rank2_mult
        notional = self.base_notional(eq, dist, mult)
        notional = min(notional, self.cap_notional(eq, dist)) * rank_mult
        tp_ok = not p.use_tp or take > 0
        if (
            not (tp_ok and eq > 0 and notional > 0 and int(flags) == 0 and rank_mult > 0)
            or dr_block
            or kill
            or kelly_block
        ):
            return
        n_open = self.open_count(side)
        if n_open >= p.max_slots:
            weakest, lowest = self.weakest_open(side, b)
            fresh = b.scores[k]
            if weakest >= 0 and not math.isnan(fresh) and fresh - lowest >= p.swap_edge:
                px_w = b.close[weakest]
                room = eq * p.max_exp - (self.heat(b.close) - s.notional[weakest])
                if not math.isnan(px_w) and room > 0:
                    self.close_trade(weakest, px_w, STATUS_RANK, REASON_RANK, step_i)
                    n_open -= 1
        final = notional
        if n_open < p.max_slots:
            room = eq * p.max_exp - self.heat(b.close)
            if room <= 0:
                return
            final = min(notional, room)
            if final > 0:
                start = b.starts[k]
                signal_pct = (
                    0.0 if math.isnan(start) or start == 0 else (price - start) / start * 100.0
                )
                self.open_trade(
                    k,
                    side,
                    rank,
                    price,
                    risk,
                    final,
                    step_i,
                    take,
                    div_n,
                    kelly_ix,
                    meta,
                    signal_pct,
                )
                if self.events is not None:
                    ratio = math.nan if b.dr_ratio is None else float(b.dr_ratio[k])
                    self.emit_open(
                        step_i, k, side, rank, div_n, price, take, final, eq, dist, mult * rank_mult,
                        (dyn_mult, kelly_mult, dr_mult, rank_mult), kelly_ix, drawdown, ratio,
                        percentile,
                    )  # fmt: skip

    def emit_open(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, step_i, k, side, rank, div_n, price, take, final, eq, dist, total, parts, kelly_ix,
        drawdown, ratio, percentile,
    ) -> None:  # fmt: skip
        """AÇ olayı: büyüklük zincirinin parçalarını salt-okunur yeniden hesaplar."""
        p, s = self.p, self.s
        trade = s.trade_ref[k]
        score = math.nan
        dd_factor = streak_factor = 1.0
        if p.use_dyn:
            score, _ = self.dyn_raw(side, rank, div_n)
            dd_factor = p.dd_mult if drawdown >= p.dd_throttle else 1.0
            streak_factor = p.streak_mult if s.ks[8] >= p.streak_n else 1.0
        mean, effective, exploring = 0.0, 0.0, False
        if p.kelly.on:
            mean, effective = self.kelly.estimate(list(kelly_ix))
            exploring = not ge(effective, p.kelly.min_n)
        self.emit(
            OpenEvent, step_i, symbol=k, side=side, trade_id=trade.trade_id, method=trade.method,
            strength=trade.strength, rank=rank, div_n=div_n, entry=price, take=take,
            notional=final, risk_pct=final * dist / eq * 100.0, multiplier=total,
            dyn_mult=parts[0], kelly_mult=parts[1], dr_mult=parts[2], rank_mult=parts[3],
            dyn_on=p.use_dyn, dyn_score=score, dd_factor=dd_factor, streak_factor=streak_factor,
            below_start_cap=p.lock_on and eq < p.start_cap, kelly_on=p.kelly.on, kelly_mean=mean,
            kelly_effective_n=effective, kelly_exploring=exploring, dr_on=p.dr_on,
            dr_ratio=ratio, dr_percentile=percentile, dist=dist,
        )  # fmt: skip

    def enter(self, step_i: int, b: BarInputs, halted: bool, drawdown: float) -> None:
        p, s = self.p, self.s
        for k in range(self.n):
            s.off_count[k] = 0 if b.eligible[k] else s.off_count[k] + 1
            if s.off_count[k] >= p.stay_reset_bars:
                s.stay_used[k] = False
        for side, idx, allowed in (
            (1, b.idx_long, p.allow_long),
            (-1, b.idx_short, p.allow_short),
        ):
            if not allowed:
                continue
            for r in range(p.max_slots):
                k = int(idx[r])
                if b.t_rank[k] != r + 1 or b.types[k] != side or s.active[k]:
                    continue
                if not (p.re_entry or not s.stay_used[k]):
                    continue
                self.try_entry(step_i, b, side, r + 1, k, halted, drawdown)

    def track_peak(self, close: np.ndarray) -> None:
        ks = self.s.ks
        eq = self.equity(close)
        ks[1] = max(ks[1], eq)
        dd = (ks[1] - eq) / ks[1] * 100.0 if ks[1] > 0 else 0.0
        ks[2] = max(ks[2], dd)

    def step(self, step_i: int, b: BarInputs, confirmed: bool = True) -> None:
        if b.day_changed:
            self.s.day_pending = True
        if not confirmed:
            return
        self.phase = PHASE_EXIT
        self.manage_open(step_i, b)
        self.pool.decay_step(step_i)
        self.dyn_update(step_i, b.close)
        self.kelly.decay_step(step_i)
        self.phase = PHASE_PROTECT
        halted, drawdown = self.protections(step_i, b)
        self.phase = PHASE_ENTRY
        self.enter(step_i, b, halted, drawdown)
        self.track_peak(b.close)
