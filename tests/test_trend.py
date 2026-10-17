"""A2 — TREND yöntemi: durum makinesi, güç kodu ve Pine na davranışı."""

import numpy as np

from alamet.threshold import threshold_state, threshold_strength
from alamet.trend import trend_signal


def test_state_machine_sequence():
    z = np.array([0.6, 0.3, -0.1, -0.6, -0.2, 0.2, np.nan])
    assert threshold_state(z, 0.5).tolist() == [1, 1, 0, -1, -1, 0, 0]


def test_state_holds_on_nan():
    z = np.array([0.6, np.nan, np.nan])
    assert threshold_state(z, 0.5).tolist() == [1, 1, 1]


def test_strength_thresholds():
    z = np.array([0.2, 0.5, 1.0, -1.0, 1.5, -2.0, np.nan])
    assert threshold_strength(z, 0.5).tolist() == [0, 0, 1, 1, 2, 2, 0]


def test_warmup_follows_pine_na_semantics():
    n = 40
    base = np.linspace(100.0, 140.0, n)
    res = trend_signal(base + 1.0, base - 1.0, base)
    assert (res.z[:9] == 0.0).all()
    assert np.isnan(res.z[9:23]).all()
    assert not np.isnan(res.z[23:]).any()


def test_rising_series_gives_long_state():
    n = 60
    base = np.linspace(100.0, 200.0, n)
    res = trend_signal(base + 1.0, base - 1.0, base)
    assert res.state[-1] == 1
    assert res.z[-1] > 0.5


def test_falling_series_gives_short_state():
    n = 60
    base = np.linspace(200.0, 100.0, n)
    res = trend_signal(base + 1.0, base - 1.0, base)
    assert res.state[-1] == -1
    assert res.z[-1] < -0.5
