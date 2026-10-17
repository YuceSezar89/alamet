"""A6 — MTF Heikin Ashi gücü: elle hesaplanmış örnek ve HTF bar birleştirme."""

import numpy as np

from alamet.heikin import bucket_ids, ha_score

OPEN = np.array([10.0, 11.0, 12.0, 13.0])
HIGH = np.array([12.0, 13.0, 14.0, 15.0])
LOW = np.array([9.0, 10.0, 11.0, 12.0])
CLOSE = np.array([11.0, 12.0, 13.0, 14.0])
TIME = np.array([0, 900, 1800, 2700])


def test_bucket_ids_group_bars_into_htf():
    assert bucket_ids(TIME, 1800).tolist() == [0, 0, 1, 1]


def test_hand_calculated_two_bucket_example():
    out = ha_score(OPEN, HIGH, LOW, CLOSE, bucket_ids(TIME, 1800))
    np.testing.assert_allclose(out, [0.0, 0.0, 50.0, 50.0])


def test_bucket_of_one_bar_follows_plain_ha_of_each_bar():
    out = ha_score(OPEN, HIGH, LOW, CLOSE, np.arange(4))
    assert out[0] == 0.0
    assert (out[1:] > 0).all()


def test_flat_series_gives_zero_not_nan():
    flat = np.full(10, 100.0)
    out = ha_score(flat, flat, flat, flat, np.zeros(10, dtype=int))
    assert (out == 0.0).all()


def test_rising_series_is_positive_and_falling_negative():
    base = np.linspace(100.0, 130.0, 60)
    bucket = bucket_ids(np.arange(60) * 900, 3600)
    up = ha_score(base - 0.2, base + 0.5, base - 0.5, base, bucket)
    down = ha_score(base[::-1] + 0.2, base[::-1] + 0.5, base[::-1] - 0.5, base[::-1], bucket)
    assert up[-1] > 0
    assert down[-1] < 0


def test_different_timeframes_give_different_scores():
    rng = np.random.default_rng(9)
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, 200))
    open_ = close - rng.normal(0, 0.4, 200)
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    t = np.arange(200) * 900
    a = ha_score(open_, high, low, close, bucket_ids(t, 1800))
    b = ha_score(open_, high, low, close, bucket_ids(t, 5400))
    assert not np.allclose(a, b)
    assert np.all(np.abs(a) <= 100.0) and np.all(np.abs(b) <= 100.0)


def test_data_starting_mid_bucket_still_initialises():
    out = ha_score(OPEN[1:], HIGH[1:], LOW[1:], CLOSE[1:], bucket_ids(TIME[1:], 1800))
    assert np.isfinite(out).all()
