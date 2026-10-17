"""
Uçtan uca sistem: 20 sembollü OHLCV → sinyal paneli → işlem motoru (bar bar).

Çıktı, bütün-sistem doğrulaması için orijinal DEPV2 debug kopyasının plotlarıyla aynı
büyüklükleri taşır: sembol başına açık yön / giriş fiyatı / notional ve kasa (ks[0..2]).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from alamet.daily import DayInputs
from alamet.divergence import DivergenceTracker
from alamet.engine import BarInputs, EngineState, Trade, TradeEngine, TradeParams
from alamet.events import PHASE_DIVERGENCE, DivergenceEvent, DoubleDivergenceEvent, EventLog
from alamet.panel import PanelParams, SignalPanel, build_panel
from alamet.ranking import RankResult, rank_all
from alamet.stats import live_rows, stats_vector
from alamet.universe import Universe
from alamet.validate import validate_params, validate_timeframe

SECONDS_PER_DAY = 86400


@dataclass
class SystemResult:  # pylint: disable=too-many-instance-attributes
    panel: SignalPanel
    sides: np.ndarray
    entries: np.ndarray
    notionals: np.ndarray
    realised: np.ndarray
    peak: np.ndarray
    max_dd: np.ndarray
    closed_count: np.ndarray
    trades: list[Trade]
    state: EngineState
    stats: np.ndarray | None = None
    events: EventLog | None = None
    engine: TradeEngine | None = None


def day_changes(time_s: np.ndarray, open_hour: int = 0) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=np.int64)
    changed = np.zeros(len(time_s), dtype=bool)
    if open_hour == 0:
        days = time_s // SECONDS_PER_DAY
        changed[1:] = days[1:] != days[:-1]
    else:
        hours = (time_s % SECONDS_PER_DAY) // 3600
        changed[1:] = (hours[1:] == open_hour) & (hours[:-1] != open_hour)
    return changed


def _log_divergence(log: EventLog, bar: int, raw_events) -> None:
    """Takipçinin ham olaylarını (`ayrisma`, `cift_ayrisma`) tipli olaylara çevirir."""
    for kind, symbol, number, value in raw_events:
        if kind == "ayrisma":
            log.add(
                DivergenceEvent,
                bar,
                PHASE_DIVERGENCE,
                symbol=symbol,
                number=number,
                direction=value,
            )
        else:
            log.add(
                DoubleDivergenceEvent,
                bar,
                PHASE_DIVERGENCE,
                symbol=symbol,
                number=number,
                move=value,
            )


def _pooled_row(engine: TradeEngine, tracker: DivergenceTracker, panel: SignalPanel, params, i):
    """Havuz açıkken bir barın sıralaması ve ayrışma adımı (havuz maskesi motor durumundan)."""
    pool = engine.pool
    allowed = pool.protect_top1(pool.mask(), panel.scores[i], panel.types[i], panel.own_ok[i])
    rank = rank_all(
        panel.scores[i : i + 1], panel.types[i : i + 1], panel.own_ok[i : i + 1],
        params.n_top, params.gap_min, allowed[None, :], "pine",
    )  # fmt: skip
    step = tracker.step(
        panel.universe.close[i], panel.types[i], rank.top[0], rank.idx_long[0], rank.idx_short[0],
        bool(panel.confirmed[i]),
    )  # fmt: skip
    return rank, step


def _with_pooled(panel: SignalPanel, ranks: list, steps: list) -> SignalPanel:
    stacked = RankResult(
        *(np.concatenate([getattr(r, f.name) for r in ranks]) for f in fields(RankResult))
    )
    keys = ("count", "active", "cur_move", "last_dbl", "lock", "eligible", "t_rank")
    divergence = {key: np.array([getattr(step, key) for step in steps]) for key in keys}
    divergence["events"] = [step.events for step in steps]
    return replace(panel, rank=stacked, divergence=divergence)


def run_system(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    universe: Universe,
    panel_params: PanelParams = PanelParams(),
    trade_params: TradeParams = TradeParams(),
    confirm_last: bool = False,
    panel: SignalPanel | None = None,
    start: int = 0,
    signal_start: int | None = None,
    day: DayInputs | None = None,
    trade_enabled: bool = True,
    collect_stats: bool = False,
    collect_events: bool = False,
) -> SystemResult:
    validate_params(panel_params, trade_params)
    validate_timeframe(panel_params, universe)
    panel = panel or build_panel(universe, panel_params, confirm_last, start, signal_start, day)
    universe = panel.universe
    engine = TradeEngine(universe.n, trade_params)
    log = EventLog() if collect_events else None
    if log is not None:
        log.times = universe.time
        engine.events = log
    bars, n = universe.bars, universe.n
    sides, entries, notionals = (np.zeros((bars, n)) for _ in range(3))
    ks = np.zeros((bars, 3))
    closed = np.zeros(bars, dtype=int)
    changed = day_changes(universe.time, panel_params.daily_open_hour)
    pooled = trade_params.pool.on
    tracker = DivergenceTracker(n, panel_params.divergence) if pooled else None
    ranks, steps = [], []
    known, live, closed_total = 0, [], 0
    stats_rows: list[np.ndarray] = []
    for i in range(bars):
        if pooled:
            rank_row, div_row = _pooled_row(engine, tracker, panel, panel_params, i)
            ranks.append(rank_row)
            steps.append(div_row)
            idx_long, idx_short = rank_row.idx_long[0], rank_row.idx_short[0]
            eligible, t_rank, count = div_row.eligible, div_row.t_rank, div_row.count
            top, active, cur_move = rank_row.top[0], div_row.active, div_row.cur_move
        else:
            div = panel.divergence
            idx_long, idx_short = panel.rank.idx_long[i], panel.rank.idx_short[i]
            eligible, t_rank, count = div["eligible"][i], div["t_rank"][i], div["count"][i]
            top, active, cur_move = panel.rank.top[i], div["active"][i], div["cur_move"][i]
        if log is not None:
            stored = None if pooled else div.get("events")
            _log_divergence(log, i, div_row.events if pooled else (stored[i] if stored else ()))
        inputs = BarInputs(
            universe.open[i], universe.high[i], universe.low[i], universe.close[i],
            panel.atr[i], panel.types[i], panel.starts[i], panel.power[i], panel.scores[i],
            eligible, t_rank, idx_long, idx_short,
            panel.daily_open[i], panel.weekly_open[i], panel.monthly_open[i],
            bool(changed[i]), panel.dr_percentile[i], count,
            panel.strengths[i], panel.methods[i],
            None if panel.dr_ratio is None else panel.dr_ratio[i],
        )  # fmt: skip
        if trade_enabled:
            engine.step(i, inputs, bool(panel.confirmed[i]))
        s = engine.s
        sides[i] = np.where(s.active, s.side, 0)
        entries[i] = np.where(s.active, s.entry, np.nan)
        notionals[i] = np.where(s.active, s.notional, 0.0)
        ks[i] = s.ks[:3]
        live.extend(engine.trades[known:])
        known = len(engine.trades)
        still = [t for t in live if t.close_bar < 0]
        closed_total += len(live) - len(still)
        live = still
        closed[i] = closed_total
        if collect_stats:
            close = universe.close[i]
            rows = live_rows(top, active, count, cur_move, panel.scores[i], panel.types[i], close)
            stats_rows.append(
                stats_vector(
                    engine, rows, engine.open_direction(close, 1), engine.open_direction(close, -1)
                )
            )
    if pooled:
        panel = _with_pooled(panel, ranks, steps)
    return SystemResult(
        panel,
        sides,
        entries,
        notionals,
        ks[:, 0],
        ks[:, 1],
        ks[:, 2],
        closed,
        engine.trades,
        engine.s,
        np.array(stats_rows) if collect_stats else None,
        log,
        engine,
    )
