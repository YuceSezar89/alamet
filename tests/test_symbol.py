"""f_sym orkestratörü: 4 yöntem + MTF onayı + onay makinesi birlikte."""

import numpy as np

from alamet.symbol import METHOD_NAMES, symbol_signals

T0 = 1772323200


def _series(n, seed, drift=0.0):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(drift, 1.0, n))
    open_ = close - rng.normal(0, 0.5, n)
    spread = rng.uniform(0.2, 1.5, n)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(50.0, 500.0, n)
    time_s = T0 + np.arange(n) * 900
    return open_, high, low, close, volume, time_s


def test_four_methods_with_consistent_shapes_and_ranges():
    n = 500
    res = symbol_signals(*_series(n, 1))
    assert len(res.methods) == len(METHOD_NAMES) == 4
    for gate in res.methods:
        assert len(gate.last_type) == n
        assert set(np.unique(gate.last_type)) <= {-1.0, 0.0, 1.0}
        assert set(np.unique(gate.pend_dir)) <= {-1.0, 0.0, 1.0}
        assert (gate.last_str >= 0).all()
        fired = gate.last_type != 0
        assert (gate.last_sig[fired] >= 0).all()
        assert (gate.last_sig[fired] < n).all()


def test_deterministic():
    data = _series(300, 2)
    a = symbol_signals(*data)
    b = symbol_signals(*data)
    for ga, gb in zip(a.methods, b.methods):
        np.testing.assert_array_equal(ga.last_type, gb.last_type)
        np.testing.assert_array_equal(ga.last_sig, gb.last_sig)


def test_steady_uptrend_gives_long_trend_signal():
    n = 250
    base = np.linspace(100.0, 220.0, n)
    time_s = T0 + np.arange(n) * 900
    res = symbol_signals(base - 0.2, base + 0.5, base - 0.5, base, np.full(n, 100.0), time_s)
    assert res.mtf.n_bull[-1] == 4
    assert res.methods[0].last_type[-1] == 1.0
    assert res.methods[0].last_start[-1] > 0


def test_steady_downtrend_gives_short_trend_signal():
    n = 250
    base = np.linspace(220.0, 100.0, n)
    time_s = T0 + np.arange(n) * 900
    res = symbol_signals(base + 0.2, base + 0.5, base - 0.5, base, np.full(n, 100.0), time_s)
    assert res.mtf.n_bear[-1] == 4
    assert res.methods[0].last_type[-1] == -1.0
