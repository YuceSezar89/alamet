"""B2 — DEPV2 günlük Devisso ratio: taşıma, mutlak değer ve yüzdelik."""

import numpy as np

from alamet.devisso import devisso_daily, ratio_percentile, ratio_series


def _closes(n, seed):
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.normal(0, 1.0, n))


def test_ratio_is_non_negative_and_nan_until_rsi_ready():
    res = devisso_daily(_closes(200, 1))
    assert np.isnan(res.ratio[:4]).all()
    valid = res.ratio[~np.isnan(res.ratio)]
    assert len(valid) > 150
    assert (valid >= 0).all()


def test_flat_day_carries_previous_ratio():
    closes = np.array([100, 101, 102, 101, 103, 104, 104, 105, 104, 106], dtype=float)
    res = devisso_daily(closes)
    assert closes[6] == closes[5]
    assert res.ratio[6] == res.ratio[5]
    assert not np.isnan(res.ratio[6])


def test_prev_ratio_is_shifted_by_one_day():
    res = devisso_daily(_closes(100, 2))
    assert np.isnan(res.prev_ratio[0])
    np.testing.assert_array_equal(res.prev_ratio[1:], res.ratio[:-1])


def test_total_amount_asymmetry_is_kept():
    res = devisso_daily(np.array([100.0, 110.0, 99.0]))
    up = (110.0 - 100.0) / 100.0 * 100.0
    down = -(110.0 - 99.0) / 99.0 * 100.0
    np.testing.assert_allclose(res.total, [0.0, up, up + down])


def test_percentile_hand_calculated():
    ratio = np.array([5.0, 3.0, 4.0, 1.0, 2.0])
    out = ratio_percentile(ratio, look=3, min_n=2)
    assert out[4] == 0.0
    assert out[3] == 50.0
    assert np.isnan(out[2])
    assert np.isnan(out[0])


def test_percentile_ties_count_half():
    ratio = np.array([2.0, 2.0, 2.0, 2.0])
    out = ratio_percentile(ratio, look=3, min_n=2)
    assert out[3] == 50.0
    assert np.isnan(out[2])


def test_percentile_needs_min_samples():
    res = devisso_daily(_closes(40, 3))
    first_valid = int(np.flatnonzero(~np.isnan(res.percentile))[0])
    assert first_valid >= 4 + 1 + 10


def test_percentile_range():
    res = devisso_daily(_closes(300, 4))
    valid = res.percentile[~np.isnan(res.percentile)]
    assert len(valid) > 200
    assert valid.min() >= 0.0 and valid.max() <= 100.0


def test_ratio_series_skips_nan_and_zero_total():
    total = np.array([0.0, 1.0, 1.0, 2.0, 1.5, 2.5, 2.5])
    _, ratio = ratio_series(total, 3)
    assert ratio[6] == ratio[5]
