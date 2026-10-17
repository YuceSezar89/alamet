"""C3 — kendi içinde ayrışma: elle adım adım izlenmiş bacak senaryoları."""

import math

import numpy as np

from alamet.own_divergence import (
    MODE_PREV,
    MODE_SIGMA,
    OwnDivergence,
    own_gate_series,
)

TYPES = np.array([[1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0], [1.0]])
PCTS = np.array([[1.0], [3.0], [2.0], [-1.0], [-4.0], [-2.0], [0.5]])


def _run(types=TYPES, pcts=PCTS, **params):
    return own_gate_series(types, pcts, **params)


def test_previous_move_mode_hand_traced():
    ok, ref, n, cur = _run(own_n=3)[MODE_PREV]
    assert ok[:, 0].tolist() == [False, False, False, False, True, True, False]
    np.testing.assert_allclose(cur[:, 0], [1.0, 3.0, 3.0, 1.0, 4.0, 4.0, 0.5])
    assert np.isnan(ref[:3, 0]).all()
    np.testing.assert_allclose(ref[3:, 0], [3.0, 3.0, 3.0, 4.0])
    assert n[:, 0].tolist() == [0, 0, 0, 1, 1, 1, 2]


def test_sigma_mode_hand_traced():
    ok, ref, n, _ = _run(own_n=3)[MODE_SIGMA]
    assert ok[:, 0].tolist() == [False] * 7
    np.testing.assert_allclose(ref[3:6, 0], [3.0, 3.0, 3.0])
    assert ref[6, 0] == 4.0
    assert n[6, 0] == 2


def test_sigma_mode_passes_with_enough_history_and_big_move():
    types = np.array([[1.0], [-1.0], [1.0], [-1.0]])
    pcts = np.array([[2.0], [-2.0], [2.0], [-9.0]])
    ok, ref, n, _ = _run(types, pcts, own_n=3)[MODE_SIGMA]
    assert n[3, 0] == 3
    assert ref[3, 0] < 9.0
    assert ok[3, 0]


def test_latch_off_uses_current_move_only():
    ok, _, _, cur = _run(own_n=3, latch=False)[MODE_PREV]
    assert cur[5, 0] == 2.0
    assert not ok[5, 0]
    assert ok[4, 0]


def test_allow_no_history_passes_positive_first_leg():
    ok, _, n, _ = _run(own_n=3, allow_no_hist=True)[MODE_PREV]
    assert ok[0, 0] and n[0, 0] == 0
    assert ok[1, 0]


def test_use_own_first_off_always_passes_with_direction():
    ok, _, _, _ = _run(own_n=3, use_own_first=False)[MODE_PREV]
    assert ok.all()


def test_no_direction_rows_do_not_change_state():
    types = np.array([[0.0], [math.nan], [1.0]])
    pcts = np.array([[5.0], [5.0], [1.0]])
    ok, ref, n, cur = _run(types, pcts)[MODE_PREV]
    assert not ok[:2, 0].any()
    assert n[:2, 0].tolist() == [0, 0]
    assert np.isnan(cur[:2, 0]).all() and np.isnan(ref[:2, 0]).all()


def test_ring_buffer_wraps_and_reads_last_written_leg():
    machine = OwnDivergence(1, own_n=2)
    for direction, pct in ((1, 1.0), (-1, -2.0), (1, 3.0), (-1, -4.0)):
        machine.update(np.array([float(direction)]), np.array([pct]))
    assert machine.count[0] == 3
    assert machine.ring[0].tolist() == [3.0, 2.0]
    machine.update(np.array([1.0]), np.array([0.1]))
    ok, ref, n, _ = machine.gate(np.array([1.0]), np.array([0.1]))
    assert n[0] == 2
    assert ref[0] == 4.0
    assert not ok[0]


def test_symbols_are_independent():
    types = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])
    pcts = np.array([[1.0, 1.0], [2.0, -3.0], [-5.0, -4.0]])
    ok, _, n, _ = _run(types, pcts, own_n=3)[MODE_PREV]
    assert n[2].tolist() == [1, 1]
    assert ok[2].tolist() == [True, True]
