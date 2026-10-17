"""C4 — ayrışma takibi, çift ayrışma, yön kilidi ve uygunluk: elle izlenmiş senaryolar."""

import numpy as np
import pytest

from alamet.divergence import DivergenceParams, DivergenceTracker, run_divergence

NAN = np.nan


def _one(rows, **params):
    """rows: (kapanış, ilk-N mi, sinyal yönü). Tek sembol, idx listeleri [0]."""
    close = np.array([[r[0]] for r in rows], dtype=float)
    top = np.array([[r[1]] for r in rows], dtype=bool)
    types = np.array([[r[2]] for r in rows], dtype=float)
    idx = np.zeros((len(rows), 1), dtype=int)
    return run_divergence(close, types, top, idx, idx, DivergenceParams(**params))


def _legs(*gains, start=100.0):
    rows = []
    for gain in gains:
        rows += [(start, True, 1.0), (start + gain, True, 1.0), (start + gain, False, 1.0)]
    return rows


LIFECYCLE = [
    (100.0, False, 1.0),
    (100.0, True, 1.0),
    (102.0, True, 1.0),
    (101.0, False, 1.0),
    (100.0, True, 1.0),
    (103.0, True, 1.0),
    (103.0, False, 1.0),
    (100.0, True, 1.0),
    (104.0, True, 1.0),
]


def test_lifecycle_counts_and_activity():
    out = _one(LIFECYCLE)
    assert out["count"][:, 0].tolist() == [0, 1, 1, 1, 2, 2, 2, 3, 3]
    assert out["active"][:, 0].tolist() == [0, 1, 1, 0, 1, 1, 0, 1, 1]


def test_lifecycle_moves_and_final_value_kept_after_exit():
    out = _one(LIFECYCLE)
    assert np.isnan(out["cur_move"][0, 0])
    np.testing.assert_allclose(out["cur_move"][1:, 0], [0, 2, 1, 0, 3, 3, 0, 4], atol=1e-9)


def test_double_divergence_only_on_third_leg_big_move():
    out = _one(LIFECYCLE)
    assert out["last_dbl"][:, 0].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1]


def test_lock_follows_up_down_move_sums():
    out = _one(LIFECYCLE)
    assert out["lock"][:, 0].tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1]


def test_eligible_only_when_double_divergence_and_lock_agree():
    out = _one(LIFECYCLE)
    assert out["eligible"][:, 0].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert out["t_rank"][:, 0].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1]


def test_without_double_requirement_active_top_symbol_is_eligible():
    out = _one(LIFECYCLE, need_dbl=False)
    assert out["eligible"][:, 0].tolist() == [0, 0, 1, 0, 1, 1, 0, 1, 1]


def test_reversal_while_top_starts_new_short_divergence():
    rows = [(100.0, True, 1.0), (102.0, True, 1.0), (101.0, True, -1.0), (99.0, True, -1.0)]
    out = _one(rows)
    assert out["count"][:, 0].tolist() == [1, 1, 2, 2]
    np.testing.assert_allclose(out["cur_move"][:, 0], [0, 2, 0, (99.0 / 101.0 - 1) * 100])


def test_short_divergence_counts_price_fall_as_favourable_move():
    legs = [(100.0, True, -1.0), (98.0, True, -1.0), (98.0, False, -1.0)]
    legs += [(100.0, True, -1.0), (99.0, True, -1.0), (99.0, False, -1.0)]
    legs += [(100.0, True, -1.0), (90.0, True, -1.0)]
    out = _one(legs)
    assert out["last_dbl"][-1, 0] == 1
    assert out["lock"][-1, 0] == -1


def test_mixed_up_and_down_moves_give_undecided_lock():
    rows = [
        (100.0, True, 1.0),
        (103.0, True, 1.0),
        (103.0, False, 1.0),
        (100.0, True, 1.0),
        (97.0, True, 1.0),
    ]
    out = _one(rows, need_dbl=False)
    assert out["lock"][-1, 0] == 2
    assert out["active"][-1, 0] == 1
    assert not out["eligible"][-1, 0]


def test_ring_buffer_keeps_only_last_max_div_divergences():
    rows = _legs(-50.0, 1.0, 1.0, 1.0)
    assert _one(rows, max_div=3)["lock"][-1, 0] == 1
    assert _one(rows, max_div=10)["lock"][-1, 0] == -1


def test_positive_only_filter_ignores_negative_history():
    rows = [
        (100.0, True, 1.0),
        (98.0, True, 1.0),
        (98.0, False, 1.0),
        (100.0, True, 1.0),
        (101.0, True, 1.0),
        (101.0, False, 1.0),
        (100.0, True, 1.0),
        (103.0, True, 1.0),
    ]
    assert _one(rows)["last_dbl"][-1, 0] == 1
    assert _one(rows, dbl_pos_only=True)["last_dbl"][-1, 0] == 0


def test_direction_not_allowed_blocks_eligibility():
    assert not _one(LIFECYCLE, allow_long=False)["eligible"].any()


def test_slot_rank_follows_sorted_index_lists():
    tracker = DivergenceTracker(3, DivergenceParams(need_dbl=False))
    types = np.array([1.0, 1.0, 1.0])
    top = np.array([True, False, True])
    idx = np.array([2, 0, 1])
    tracker.step(np.array([100.0, 100.0, 100.0]), types, top, idx, idx)
    step = tracker.step(np.array([101.0, 100.0, 102.0]), types, top, idx, idx)
    assert step.eligible.tolist() == [True, False, True]
    assert step.t_rank.tolist() == [2, 0, 1]


def test_events_report_new_divergence_and_double_divergence():
    tracker = DivergenceTracker(1)
    events = []
    closes = [100, 100, 102, 101, 100, 103, 103, 100, 104]
    tops = [0, 1, 1, 0, 1, 1, 0, 1, 1]
    for close, top in zip(closes, tops):
        step = tracker.step(
            np.array([float(close)]),
            np.array([1.0]),
            np.array([bool(top)]),
            np.array([0]),
            np.array([0]),
        )
        events += step.events
    kinds = [event[0] for event in events]
    assert kinds.count("ayrisma") == 2
    assert kinds.count("cift_ayrisma") == 1


def test_nan_close_does_not_start_or_update():
    out = _one([(NAN, True, 1.0), (100.0, True, 1.0)])
    assert out["count"][:, 0].tolist() == [0, 1]


def test_unconfirmed_bar_evaluates_without_advancing_state():
    tracker = DivergenceTracker(1)
    for close, top, side in LIFECYCLE[:8]:
        tracker.step(
            np.array([close]), np.array([side]), np.array([top]), np.array([0]), np.array([0])
        )
    last = LIFECYCLE[8]
    args = (
        np.array([last[0]]),
        np.array([last[2]]),
        np.array([last[1]]),
        np.array([0]),
        np.array([0]),
    )
    preview = tracker.step(*args, track=False)
    assert preview.count[0] == 3
    assert preview.cur_move[0] == 0.0
    assert not preview.last_dbl[0]
    confirmed = tracker.step(*args)
    assert confirmed.cur_move[0] == pytest.approx(4.0)
    assert confirmed.last_dbl[0]
    assert confirmed.eligible[0]


def test_run_divergence_track_last_false_leaves_final_bar_untracked():
    rows = LIFECYCLE
    tracked = _one(rows)
    close = np.array([[r[0]] for r in rows], dtype=float)
    top = np.array([[r[1]] for r in rows], dtype=bool)
    types = np.array([[r[2]] for r in rows], dtype=float)
    idx = np.zeros((len(rows), 1), dtype=int)
    untracked = run_divergence(close, types, top, idx, idx, track_last=False)
    assert untracked["cur_move"][-1, 0] == 0.0
    assert tracked["cur_move"][-1, 0] == pytest.approx(4.0)
    np.testing.assert_array_equal(untracked["count"][:-1], tracked["count"][:-1])
