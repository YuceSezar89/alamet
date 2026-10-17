"""A7 — MTF güç birleşimi ve TF onayı."""

import numpy as np

from alamet.mtf import mtf_confirm


def _scores(*values):
    return [np.array([v], dtype=float) for v in values]


def test_weighted_final_pct_and_counts():
    res = mtf_confirm(_scores(10.0, 20.0, -30.0, 40.0))
    np.testing.assert_allclose(res.final_pct, [200.0 / 15.0])
    assert res.n_bull.tolist() == [3]
    assert res.n_bear.tolist() == [1]
    assert res.ok_long.tolist() == [True]
    assert res.ok_short.tolist() == [False]


def test_short_confirmation_needs_three_bearish():
    res = mtf_confirm(_scores(-5.0, -1.0, -2.0, 3.0))
    assert res.ok_short.tolist() == [True]
    assert res.ok_long.tolist() == [False]


def test_zero_scores_confirm_nothing():
    res = mtf_confirm(_scores(0.0, 0.0, 0.0, 0.0))
    assert res.n_bull.tolist() == [0]
    assert res.n_bear.tolist() == [0]
    assert res.ok_long.tolist() == [False]
    assert res.ok_short.tolist() == [False]


def test_confirmation_disabled_allows_both_directions():
    res = mtf_confirm(_scores(0.0, 0.0, 0.0, 0.0), tf_on=False)
    assert res.ok_long.tolist() == [True]
    assert res.ok_short.tolist() == [True]


def test_custom_need():
    res = mtf_confirm(_scores(1.0, 1.0, -1.0, -1.0), need=2)
    assert res.ok_long.tolist() == [True]
    assert res.ok_short.tolist() == [True]


def test_vectorised_over_bars():
    res = mtf_confirm([np.array([1.0, -1.0])] * 4)
    assert res.ok_long.tolist() == [True, False]
    assert res.ok_short.tolist() == [False, True]
