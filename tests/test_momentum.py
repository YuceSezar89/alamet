"""A3 — MOMENTUM yöntemi: stdev, warm-up ve bağımsız döngü referansı."""

import math
import statistics

import numpy as np

from alamet.momentum import momentum_signal
from alamet.primitives import pine_stdev


def _naive_z(close, length, z_len):
    n = len(close)
    out = [0.0] * n
    for i in range(n):
        if i < z_len:
            continue
        window = [math.log(close[j] / close[j - 1]) for j in range(i - z_len + 1, i + 1)]
        sd = statistics.pstdev(window)
        if sd > 0:
            out[i] = math.log(close[i] / close[i - length]) / (sd * math.sqrt(length))
    return out


def test_stdev_is_population():
    out = pine_stdev(np.array([1.0, 2.0, 3.0, 4.0]), 2)
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1:], [0.5, 0.5, 0.5])


def test_stdev_nan_in_window_gives_nan():
    out = pine_stdev(np.array([np.nan, 1.0, 2.0, 3.0]), 2)
    assert np.isnan(out[:2]).all()
    np.testing.assert_allclose(out[2:], [0.5, 0.5])


def test_z_matches_naive_loop():
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 200)))
    res = momentum_signal(close, length=12, z_len=48)
    np.testing.assert_allclose(res.z, _naive_z(close.tolist(), 12, 48), atol=1e-9)


def test_warmup_is_zero_until_stdev_ready():
    rng = np.random.default_rng(3)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 80)))
    res = momentum_signal(close)
    assert (res.z[:48] == 0.0).all()
    assert not np.isnan(res.z).any()
    assert (res.z[48:] != 0.0).any()


def test_zero_stdev_gives_zero_z():
    res = momentum_signal(np.full(80, 100.0))
    assert (res.z == 0.0).all()
    assert (res.state == 0).all()


def test_strong_uptrend_gives_long_state():
    rng = np.random.default_rng(11)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.01, 0.002, 120)))
    res = momentum_signal(close)
    assert res.state[-1] == 1
    assert res.strength[-1] >= 1


def test_short_series_is_all_zero():
    res = momentum_signal(np.array([100.0, 101.0, 102.0]))
    assert (res.z == 0.0).all()
