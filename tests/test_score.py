"""C1 — temel skor, hedef mesafesi ve Devisso kolaylık puanı: elle hesaplanmış örnekler."""

import numpy as np

from alamet.score import (
    DR_OWN,
    DR_XS,
    RANK_RAW,
    TGT_BIG,
    TGT_NEAR,
    atr_percent,
    base_score,
    combine_percentile,
    combine_score,
    cross_percentile,
    ease_bonus,
    near_level,
    signal_pct,
    target_bonus,
    target_distance,
    target_distances,
)

NAN = np.nan


def test_atr_percent():
    out = atr_percent(np.array([2.0, NAN, 1.0]), np.array([100.0, 100.0, 0.0]))
    assert out[0] == 2.0
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_signal_pct_rules():
    out = signal_pct(np.array([110.0, 110.0, 110.0]), np.array([100.0, NAN, 0.0]))
    np.testing.assert_allclose(out, [10.0, 0.0, 0.0])


def test_base_score_atr_and_raw_modes():
    types = np.array([1.0, -1.0, 0.0, 1.0])
    pcts = np.array([2.0, -3.0, 5.0, 4.0])
    atr_p = np.array([1.0, 1.5, 1.0, 0.0])
    atr_mode = base_score(types, pcts, atr_p)
    np.testing.assert_allclose(atr_mode[:2], [2.0, 2.0])
    assert np.isnan(atr_mode[2]) and np.isnan(atr_mode[3])
    raw_mode = base_score(types, pcts, atr_p, RANK_RAW)
    np.testing.assert_allclose(raw_mode[[0, 1, 3]], [2.0, 3.0, 4.0])
    assert np.isnan(raw_mode[2])


def _levels():
    row = np.full(18, NAN)
    row[0], row[2], row[6], row[12] = 102.0, 90.0, 110.0, 150.0
    return row[None, :]


def test_near_level_up_and_down():
    px = np.array([100.0])
    assert near_level(_levels(), px, np.array([True]), 0, 17)[0] == 102.0
    assert near_level(_levels(), px, np.array([True]), 6, 17)[0] == 110.0
    assert near_level(_levels(), px, np.array([False]), 0, 17)[0] == 90.0
    assert np.isnan(near_level(_levels(), px, np.array([False]), 6, 17)[0])


def test_near_level_is_strict():
    row = np.full((1, 18), NAN)
    row[0, 0] = 100.0
    assert np.isnan(near_level(row, np.array([100.0]), np.array([True]), 0, 17)[0])
    assert np.isnan(near_level(row, np.array([100.0]), np.array([False]), 0, 17)[0])


def test_target_distances_and_modes():
    px, atr = np.array([100.0]), np.array([2.0])
    d_near, d_big = target_distances(_levels(), px, atr, np.array([True]))
    assert d_near[0] == 1.0 and d_big[0] == 5.0
    assert target_distance(_levels(), px, atr, np.array([True]))[0] == 3.0
    assert target_distance(_levels(), px, atr, np.array([True]), TGT_NEAR)[0] == 1.0
    assert target_distance(_levels(), px, atr, np.array([True]), TGT_BIG)[0] == 5.0
    assert target_distance(_levels(), px, atr, np.array([False]))[0] == 5.0


def test_target_distance_needs_valid_atr():
    out = target_distance(_levels(), np.array([100.0]), np.array([0.0]), np.array([True]))
    assert np.isnan(out[0])


def test_target_bonus_cap_and_raw_scale():
    dist = np.array([3.0, 12.0, NAN])
    np.testing.assert_allclose(target_bonus(dist, np.array([1.0] * 3))[:2], [0.75, 2.0])
    assert np.isnan(target_bonus(dist, np.array([1.0] * 3))[2])
    raw = target_bonus(dist, np.array([2.0, 2.0, 2.0]), rank_mode=RANK_RAW)
    np.testing.assert_allclose(raw[:2], [1.5, 4.0])


def test_cross_percentile_hand_calculated():
    out = cross_percentile(np.array([[1.0, 2.0, 3.0, NAN, 5.0]]))[0]
    np.testing.assert_allclose(out[[0, 1, 2, 4]], [0.0, 100.0 / 3, 200.0 / 3, 100.0])
    assert np.isnan(out[3])


def test_cross_percentile_ties_and_minimum_symbols():
    np.testing.assert_allclose(
        cross_percentile(np.array([[1.0, 1.0, 2.0]]))[0], [25.0, 25.0, 100.0]
    )
    assert np.isnan(cross_percentile(np.array([[1.0, 2.0, NAN]]))).all()


def test_combine_percentile_sources():
    own = np.array([20.0, NAN, 40.0, NAN])
    xs = np.array([60.0, 80.0, NAN, NAN])
    np.testing.assert_allclose(combine_percentile(own, xs)[:3], [40.0, 80.0, 40.0])
    assert np.isnan(combine_percentile(own, xs)[3])
    np.testing.assert_allclose(combine_percentile(own, xs, DR_OWN)[:3], [20.0, 80.0, 40.0])
    np.testing.assert_allclose(combine_percentile(own, xs, DR_XS)[:3], [60.0, 80.0, 40.0])


def test_ease_bonus():
    out = ease_bonus(np.array([0.0, 50.0, 100.0, NAN]), 1.0)
    np.testing.assert_allclose(out[:3], [1.0, 0.5, 0.0])
    assert np.isnan(out[3])
    raw = ease_bonus(np.array([0.0]), 1.0, np.array([2.0]), RANK_RAW)
    assert raw[0] == 2.0


def test_combine_score_rules():
    base = np.array([1.0, NAN, 2.0, 3.0])
    tgt = np.array([0.5, 0.5, NAN, 1.0])
    dr = np.array([0.25, 0.25, 0.25, NAN])
    np.testing.assert_allclose(combine_score(base, tgt, dr)[[0, 2, 3]], [1.75, 2.25, 4.0])
    assert np.isnan(combine_score(base, tgt, dr)[1])
    blocked = combine_score(base, tgt, dr, np.array([False, True, True, False]))
    np.testing.assert_allclose(blocked[[0, 3]], [1.25, 3.0])
