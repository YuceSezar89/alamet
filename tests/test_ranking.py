"""C2 — sıralama ve ilk N: elle hesaplanmış örnekler."""

from pathlib import Path

import numpy as np
import pandas as pd

from alamet.ranking import SENTINEL, rank_all, sort_descending

SCORES = np.array([[3.0, 1.0, 2.0, 5.0, 4.0]])
TYPES = np.array([[1.0, 1.0, -1.0, 1.0, -1.0]])
OK = np.ones((1, 5), dtype=bool)


def test_hand_calculated_ranking_and_top():
    res = rank_all(SCORES, TYPES, OK, n_top=2)
    np.testing.assert_allclose(res.r_long[0], [3.0, 1.0, SENTINEL, 5.0, SENTINEL])
    np.testing.assert_allclose(res.r_short[0], [SENTINEL, SENTINEL, 2.0, SENTINEL, 4.0])
    assert res.idx_long[0, :3].tolist() == [3, 0, 1]
    assert res.idx_short[0, :2].tolist() == [4, 2]
    assert res.cut_long[0] == 1.0
    assert res.cut_short[0] == SENTINEL
    assert res.top[0].tolist() == [True, False, True, True, True]


def test_gap_min_filters_close_candidates():
    res = rank_all(SCORES, TYPES, OK, n_top=2, gap_min=2.5)
    assert res.top[0].tolist() == [False, False, True, True, True]


def test_own_gate_false_removes_symbol_from_ranking():
    own_ok = OK.copy()
    own_ok[0, 3] = False
    res = rank_all(SCORES, TYPES, own_ok, n_top=2)
    assert res.r_long[0, 3] == SENTINEL
    assert res.idx_long[0, 0] == 0
    assert not res.top[0, 3]


def test_nan_score_and_zero_type_are_excluded():
    scores = np.array([[np.nan, 1.0, 2.0, 5.0, 4.0]])
    types = np.array([[1.0, 0.0, -1.0, 1.0, -1.0]])
    res = rank_all(scores, types, OK, n_top=2)
    assert res.r_long[0, 0] == SENTINEL
    assert res.r_long[0, 1] == SENTINEL
    assert res.r_long[0, 3] == 5.0


def test_pool_mask_removes_symbol():
    pool = np.array([[True, True, True, False, True]])
    res = rank_all(SCORES, TYPES, OK, pool_ok=pool)
    assert res.r_long[0, 3] == SENTINEL


def test_n_top_is_clipped_to_symbol_count_minus_one():
    res = rank_all(SCORES, TYPES, OK, n_top=10)
    assert res.cut_long.shape == (1,)
    assert res.top[0].any()


def test_tie_rules():
    values = np.array([[2.0, 2.0, 1.0]])
    assert sort_descending(values, "first")[0].tolist() == [0, 1, 2]
    assert sort_descending(values, "last")[0].tolist() == [1, 0, 2]


def test_pine_tie_rule_from_c2_export():
    pair = np.array([[2.0, 2.0, 1.0]])
    assert sort_descending(pair, "pine")[0].tolist() == [1, 0, 2]
    groups = np.array([[5.0, 1.0, 5.0, 1.0, 5.0]])
    assert sort_descending(groups, "pine")[0].tolist() == [4, 2, 0, 3, 1]
    equal = np.full((1, 5), SENTINEL)
    assert sort_descending(equal, "pine")[0].tolist() == [0, 1, 2, 3, 4]


def test_pine_tie_rule_keeps_order_only_for_untouched_array():
    touched_equal = np.full((1, 5), 2.0)
    assert sort_descending(touched_equal, "pine")[0].tolist() == [4, 3, 2, 1, 0]
    assert sort_descending(touched_equal, "pine", fill=2.0)[0].tolist() == [0, 1, 2, 3, 4]


def test_pine_tie_rule_matches_p1_probe_export():
    data = pd.read_csv(Path(__file__).parent / "data" / "pine_sort_probe.csv")
    for prefix, size in (("5", 5), ("20", 20)):
        values = data[[f"A{prefix}_{i}" for i in range(size)]].to_numpy()
        expected = data[[f"S{prefix}_{i}" for i in range(size)]].to_numpy()
        np.testing.assert_array_equal(sort_descending(values, "pine", fill=0.0), expected)


def test_sort_is_descending_for_random_rows():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(50, 7))
    order = sort_descending(values)
    sorted_values = np.take_along_axis(values, order, axis=1)
    assert (np.diff(sorted_values, axis=1) <= 0).all()


def test_vectorised_over_bars():
    scores = np.vstack([SCORES, SCORES[:, ::-1]])
    types = np.vstack([TYPES, TYPES[:, ::-1]])
    res = rank_all(scores, types, np.ones((2, 5), dtype=bool), n_top=2)
    assert res.idx_long[0, 0] == 3
    assert res.idx_long[1, 0] == 1
