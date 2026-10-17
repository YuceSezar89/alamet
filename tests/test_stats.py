import numpy as np

from alamet.engine import TradeParams
from alamet.stages import panel_params
from alamet.stats import (
    DYN_BUCKETS,
    StatsParams,
    TradeLog,
    TradeStats,
    cause_buckets,
    live_rows,
    stats_layout,
    vector_names,
)
from alamet.system import run_system
from alamet.universe import Universe


def _universe(n=6, bars=1200, seed=9):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, (bars, n)), axis=0))
    spread = close * 0.004
    open_ = np.vstack([close[:1], close[:-1]])
    time = 1_767_571_200 + np.arange(bars) * 900
    return Universe(
        tuple(f"S{i}" for i in range(n)), time, open_, np.maximum(open_, close) + spread,
        np.minimum(open_, close) - spread, close, rng.uniform(100, 200, (bars, n)),
    )  # fmt: skip


def _run(**trade):
    return run_system(
        _universe(), panel_params(n_top=3, max_slots=2), TradeParams(max_slots=2, **trade), True,
        collect_stats=True,
    )  # fmt: skip


def test_hold_bucket_boundaries():
    stats = TradeStats(3, StatsParams(hold_short=3, hold_long=12))
    assert [stats.hold_bucket(h, 24) for h in (0, 3, 4, 12, 13)] == [24, 24, 25, 25, 26]
    assert [stats.hold_bucket(h, 7) for h in (3, 4, 13)] == [7, 8, 9]


def test_log_ring_overwrites_oldest_and_closes_only_matching_ids():
    log = TradeLog(2)
    first = log.open(1, 4, 1, 100.0, 1.0, np.nan, 0.5, 50.0)
    log.open(2, 5, -1, 90.0, 1.0, np.nan, 0.0, 40.0)
    third = log.open(3, 6, 1, 80.0, 1.0, np.nan, 0.0, 30.0)
    assert first == third == 0 and log.head == 3 and list(log.id) == [3, 2]
    log.close(first, 1, 99.0, 4)
    assert np.isnan(log.exit[0])
    log.close(third, 3, 99.0, 4)
    assert log.exit[0] == 99.0 and log.status[0] == 4


def test_every_closed_trade_falls_into_exactly_one_reason_direction_and_symbol_bucket():
    result = _run()
    closed = [t for t in result.trades if t.close_bar >= 0]
    assert len(closed) > 20
    vec = result.stats[-1]
    names = vector_names(n_symbols=6)
    offset = {}
    position = 0
    for name, size in stats_layout(n_symbols=6):
        offset[name] = position
        position += size
    wins = vec[offset["lsW"] : offset["lsW"] + cause_buckets(6)]
    losses = vec[offset["lsL"] : offset["lsL"] + cause_buckets(6)]
    total = wins + losses
    assert total[:9].sum() == len(closed)
    assert total[9:11].sum() == len(closed)
    assert total[11:15].sum() == len(closed)
    assert total[15:18].sum() == len(closed)
    assert total[18:20].sum() == len(closed)
    assert total[20:24].sum() == len(closed)
    assert total[24:27].sum() == len(closed)
    assert total[27:31].sum() == len(closed)
    assert total[31:33].sum() == len(closed)
    assert total[33:35].sum() == len(closed)
    assert total[36:42].sum() == len(closed)
    assert len(names) == len(vec)
    assert vec[offset["tCnt"] : offset["tCnt"] + 2].sum() == len(closed)
    assert np.isclose(
        vec[offset["symNet"] : offset["symNet"] + 6].sum(), sum(t.usd for t in closed)
    )
    assert np.isclose(vec[0], result.realised[-1])


def test_dynamic_bucket_extension_only_adds_bookkeeping_buckets():
    result = _run()
    names = vector_names(n_symbols=6)
    start = names.index("bNe[0]")
    bne = result.stats[-1][start : start + DYN_BUCKETS]
    assert bne[0] + bne[1] == bne[2] + bne[3] == bne[4] + bne[5] + bne[6]
    assert bne[7] + bne[8] + bne[9] == bne[10] + bne[11] == bne[0] + bne[1]


def test_live_rows_flags_target_rows_and_orders_by_score():
    top = np.array([True, True, False, True])
    active = np.array([True, False, False, True])
    count = np.array([2, 0, 1, 1])
    move = np.array([1.5, np.nan, np.nan, -0.5])
    scores = np.array([3.0, 5.0, 4.0, 1.0])
    types = np.array([1.0, 1.0, -1.0, 1.0])
    n_live, lv, long_order, short_order = live_rows(
        top, active, count, move, scores, types, np.array([1.0, 1.0, 1.0, 1.0])
    )
    assert list(n_live) == [2, 1, 1, 1]
    assert lv[0] == 1.5 and lv[1] == 0.0 and np.isnan(lv[2]) and lv[3] == -0.5
    assert long_order[0] == 1 and short_order[0] == 2


def test_stats_are_independent_of_the_decisions():
    plain = run_system(
        _universe(), panel_params(n_top=3, max_slots=2), TradeParams(max_slots=2), True
    )
    counted = _run()
    assert [t.usd for t in plain.trades] == [t.usd for t in counted.trades]
