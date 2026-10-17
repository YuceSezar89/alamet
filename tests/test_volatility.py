"""A4 — VOLATİLİTE yöntemi: durum makinesi, güç kodu ve Pine na/nz davranışı."""

import numpy as np

from alamet.volatility import volatility_signal, volatility_state, volatility_strength


def test_state_up_down_and_calm_reset():
    up = np.array([True, False, False, False, False, False])
    down = np.array([False, False, False, True, False, False])
    ratio = np.array([1.0, 0.9, 0.7, 1.0, 0.9, 0.5])
    assert volatility_state(up, down, ratio, 0.8).tolist() == [1, 1, 0, -1, -1, 0]


def test_up_wins_when_both_flags_true():
    up = np.array([True])
    down = np.array([True])
    assert volatility_state(up, down, np.array([1.0]), 0.8).tolist() == [1]


def test_state_does_not_reset_when_ratio_above_calm():
    up = np.array([True, False, False])
    down = np.array([False, False, False])
    ratio = np.array([1.0, 0.8, 1.2])
    assert volatility_state(up, down, ratio, 0.8).tolist() == [1, 1, 1]


def test_strength_thresholds():
    x = np.array([1.0, 1.6, 1.9, 2.0, 3.0, np.nan])
    assert volatility_strength(x, 1.3).tolist() == [0, 1, 1, 2, 2, 0]


def _bars(n, seed):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    open_ = close - rng.normal(0, 0.5, n)
    spread = rng.uniform(0.2, 2.0, n)
    return open_, np.maximum(open_, close) + spread, np.minimum(open_, close) - spread, close


def test_warmup_defaults_follow_pine_nz_semantics():
    open_, high, low, close = _bars(60, 5)
    res = volatility_signal(open_, high, low, close)
    assert (res.x[:12] == 1.0).all()
    assert res.x[12] != 1.0
    assert (res.ratio[:11] == 1.0).all()
    assert res.ratio[11] != 1.0


def test_expansion_bar_sets_direction():
    n = 40
    base = np.full(n, 100.0)
    open_ = base.copy()
    close = base + 0.1
    high = base + 0.5
    low = base - 0.5
    open_[30], close[30], high[30], low[30] = 100.0, 106.0, 107.0, 99.5
    res = volatility_signal(open_, high, low, close)
    assert res.state[29] == 0
    assert res.state[30] == 1
    open_[31], close[31], high[31], low[31] = 106.0, 98.0, 106.5, 97.5
    res = volatility_signal(open_, high, low, close)
    assert res.state[31] == -1
    assert res.strength[30] == 2
