import math

import pytest

from alamet.kelly import (
    BUCKETS,
    KellyEngine,
    KellyParams,
    bucket_indices,
    ease_tier,
    open_side,
)


def test_bucket_indices_follow_pine_layout():
    ix = bucket_indices(3, 1, 1, 2, 0, 2, 1, 56)
    assert ix == [9, 13, 15, 18, 20, 31, 33, 56, 39]
    ix = bucket_indices(19, -1, 2, 7, 5, 3, -1, 59)
    assert ix == [10, 14, 17, 19, 23, 32, 34, 59, 55]
    assert bucket_indices(0, 1, 1, 3, 1, 0, 0, 58)[4] == 21
    assert bucket_indices(0, 1, 1, 5, 1, 0, 0, 58)[4] == 22


def test_ease_tier_and_open_side():
    assert [ease_tier(x) for x in (10, 25, 40, 75, 90)] == [56, 56, 57, 58, 59]
    assert ease_tier(math.nan) == 58
    assert open_side(1, 101.0, 100.0) == 1 and open_side(1, 99.0, 100.0) == -1
    assert open_side(-1, 99.0, 100.0) == 1 and open_side(-1, 101.0, 100.0) == -1
    assert open_side(1, 100.0, math.nan) == 0


def test_disabled_engine_is_neutral():
    engine = KellyEngine(KellyParams(on=False))
    assert engine.evaluate(list(range(9)), 5.0, 15.0, 0.5) == (1.0, False)


def test_explore_until_enough_data_then_block_on_losses():
    params = KellyParams(on=True, prior=2.0, min_n=2.0, block=0.05)
    engine = KellyEngine(params)
    ix = bucket_indices(0, 1, 1, 2, 0, 0, 1, 56)
    assert engine.evaluate(ix, 0.0, 15.0, 0.5) == (params.explore, False)
    for _ in range(6):
        engine.accumulate(ix, -1.0)
    mult, block = engine.evaluate(ix, 0.0, 15.0, 0.5)
    assert block and mult == params.min_mult


def test_kelly_multiplier_matches_formula_for_winning_buckets():
    params = KellyParams(on=True, prior=2.0, min_n=2.0, z=0.0, dd_scale=False)
    engine = KellyEngine(params)
    ix = bucket_indices(0, 1, 1, 2, 0, 0, 1, 56)
    for r in (1.0, 2.0, -0.5, 1.5, 1.0, 0.5, 2.0, 1.0):
        engine.accumulate(ix, r)
    mult, block = engine.evaluate(ix, 0.0, 15.0, 0.5)
    n = engine.n[ix[0]]
    weights = [engine.n[b] / (engine.n[b] + 2.0) for b in ix]
    mean = sum(w * engine.total[b] / (engine.n[b] + 2.0) for w, b in zip(weights, ix)) / sum(
        weights
    )
    sd = engine.deviation()
    expected = min(3.0, max(0.25, 0.25 * (mean / (sd * sd)) / 0.005))
    assert n == 8 and not block
    assert mult == pytest.approx(expected)


def test_clip_decay_and_drawdown_scale():
    params = KellyParams(on=True, clip=3.0, decay_bars=10, decay=0.5, prior=1.0, min_n=1.0, z=0.0)
    engine = KellyEngine(params)
    ix = bucket_indices(0, 1, 1, 2, 0, 0, 1, 56)
    engine.accumulate(ix, 10.0)
    assert engine.total[ix[0]] == 3.0 and engine.total_sq[ix[0]] == 9.0
    engine.decay_step(7)
    assert engine.n[ix[0]] == 1.0
    engine.decay_step(10)
    assert engine.n[ix[0]] == 0.5 and engine.total[ix[0]] == 1.5
    flat = KellyEngine(KellyParams(on=True, explore=1.0, min_n=100.0))
    base, _ = flat.evaluate(ix, 0.0, 10.0, 0.5)
    scaled, _ = flat.evaluate(ix, 5.0, 10.0, 0.5)
    floor, _ = flat.evaluate(ix, 9.9, 10.0, 0.5)
    assert (base, scaled, floor) == (1.0, 0.5, 0.25)
    assert len(flat.n) == BUCKETS
