from dataclasses import replace

import numpy as np

from alamet.engine import TradeParams
from alamet.pool import PoolParams, PoolState
from alamet.primitives import EPS, ge, gt, le, lt
from alamet.stages import panel_params
from alamet.system import run_system
from alamet.universe import Universe

CAP = 1000.0


def state(count, net, **kw):
    params = PoolParams(on=True, **kw)
    pool = PoolState(len(count), params, CAP)
    pool.count[:] = count
    pool.net[:] = net
    return pool


def test_off_pool_allows_everything():
    pool = PoolState(4, PoolParams(), CAP)
    assert pool.mask().all()


def test_mask_ranks_judged_symbols_and_applies_size_and_probation():
    pool = state([5, 5, 5, 5, 1], [10.0, -3.0, 4.0, 1.0, 50.0], size=2, min_n=4.0, prob=False)
    assert list(pool.mask()) == [True, False, True, False, False]
    probation = state([5, 5, 5, 5, 1], [10.0, -3.0, 4.0, 1.0, 50.0], size=2, min_n=4.0)
    assert list(probation.mask()) == [True, False, True, True, True]


def test_forgiveness_readmits_slightly_negative_symbols():
    pool = state([6, 6], [-1.0, -50.0], size=1, min_n=4.0, forgive=0.2, prob=True)
    assert list(pool.mask()) == [True, False]
    pool.net[1] = -1.5
    assert pool.mask()[1]
    pool.net[1] = -3.0
    assert not pool.mask()[1]


def test_top1_protection_keeps_best_raw_symbol_unless_net_below_floor():
    scores = np.array([1.0, 3.0, 2.0, 5.0])
    types = np.array([1.0, 1.0, -1.0, -1.0])
    own = np.array([True, True, True, False])
    pool = state([6, 6, 6, 6], [-100.0, -100.0, 5.0, 5.0], size=1, min_n=4.0, prob=False)
    allowed = pool.mask()
    assert not allowed[1]
    protected = pool.protect_top1(allowed, scores, types, own)
    assert not protected[1]
    pool.net[1] = -5.0
    assert pool.protect_top1(pool.mask(), scores, types, own)[1]
    pool.count[1] = 1.0
    assert pool.protect_top1(pool.mask(), scores, types, own)[1]


def test_decay_only_scales_net_on_interval_and_when_on():
    pool = state([3, 3], [10.0, -10.0], decay_bars=4, decay=0.5)
    pool.decay_step(3)
    assert list(pool.net) == [10.0, -10.0]
    pool.decay_step(8)
    assert list(pool.net) == [5.0, -5.0] and list(pool.count) == [3, 3]
    off = PoolState(2, PoolParams(), CAP)
    off.net[:] = 4.0
    off.decay_step(0)
    assert list(off.net) == [4.0, 4.0]


def _universe(n=8, bars=900, seed=7):
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0, 0.004, (bars, n)), axis=0)
    close = 100 * np.exp(walk)
    spread = close * 0.003
    open_ = np.vstack([close[:1], close[:-1]])
    time = 1_767_571_200 + np.arange(bars) * 900
    symbols = tuple(f"S{i}" for i in range(n))
    return Universe(
        symbols, time, open_, np.maximum(open_, close) + spread, np.minimum(open_, close) - spread,
        close, rng.uniform(100, 200, (bars, n)),
    )  # fmt: skip


def test_permissive_pool_matches_static_pipeline_exactly():
    universe = _universe()
    panel_p = panel_params(n_top=3, max_slots=2)
    base = TradeParams(max_slots=2, swap_edge=0.3)
    plain = run_system(universe, panel_p, base, True)
    pooled_params = replace(base, pool=PoolParams(on=True, min_n=1e9, prob=True, top1=False))
    pooled = run_system(universe, panel_p, pooled_params, True)
    assert len(plain.trades) > 0
    assert np.array_equal(plain.sides, pooled.sides)
    assert np.array_equal(plain.realised, pooled.realised)
    assert np.array_equal(plain.panel.rank.idx_long, pooled.panel.rank.idx_long)
    assert np.array_equal(plain.panel.divergence["t_rank"], pooled.panel.divergence["t_rank"])


def test_strict_pool_changes_ranking_and_records_trade_net():
    universe = _universe()
    panel_p = panel_params(n_top=3, max_slots=2)
    base = TradeParams(max_slots=2, swap_edge=0.3)
    strict = replace(
        base, pool=PoolParams(on=True, size=2, min_n=1.0, prob=False, top1=False, decay_bars=10)
    )
    plain = run_system(universe, panel_p, base, True)
    pooled = run_system(universe, panel_p, strict, True)
    assert not np.array_equal(plain.panel.rank.idx_long, pooled.panel.rank.idx_long)
    assert pooled.state is not None


def test_trade_enabled_false_runs_signals_only():
    universe = _universe()
    panel_p = panel_params(n_top=3, max_slots=2)
    base = TradeParams(max_slots=2, swap_edge=0.3)
    on = run_system(universe, panel_p, base, True)
    off = run_system(universe, panel_p, base, True, trade_enabled=False)
    assert len(on.trades) > 0 and len(off.trades) == 0
    assert not off.sides.any() and (off.realised == base.start_cap).all()
    assert np.array_equal(on.panel.types, off.panel.types)


def test_tiny_decayed_nets_do_not_count_as_positive_like_pine():
    assert not gt(1.39e-11, 0.0) and gt(2e-10, 0.0) and not gt(0.0, 0.0)
    assert lt(-2e-10, 0.0) and not lt(-1e-11, 0.0)
    assert ge(5e-11, 0.0) and ge(-5e-11, 0.0) and not ge(-2 * EPS, 0.0)
    assert le(5e-11, 0.0) and le(-5e-11, 0.0) and not le(2 * EPS, 0.0)

    pool = state([1, 1, 1], [1.4e-11, 2e-10, -1e-11], size=3, min_n=1.0, forgive=0.0)
    assert list(pool.mask()) == [False, True, False]
    protected = pool.protect_top1(
        pool.mask(), np.array([3.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0]), np.array([True] * 3)
    )
    assert protected[0]
