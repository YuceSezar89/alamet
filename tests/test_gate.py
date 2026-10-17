"""A8 — onay makinesi (f_gate): elle adım adım izlenmiş senaryolar."""

import numpy as np

from alamet.gate import NO_SIGNAL_BAR, ConfirmGate, run_gate

N = 45
HIGH = np.arange(N, dtype=float) + 100.0
LOW = np.arange(N, dtype=float) + 50.0


def _bools(n, true_from=None, true_to=None, only=None):
    arr = np.zeros(n, dtype=bool)
    if only is not None:
        arr[list(only)] = True
    elif true_from is not None:
        arr[true_from:true_to] = True
    return arr


def _run(raw_long, raw_short, tf_long, tf_short, strength=1, **kwargs):
    n = len(raw_long)
    return run_gate(
        raw_long,
        raw_short,
        tf_long,
        tf_short,
        np.full(n, strength),
        HIGH[:n],
        LOW[:n],
        **kwargs,
    )


def test_direct_signal_when_tf_confirmed_on_same_bar():
    n = 15
    res = _run(_bools(n, 10, n), _bools(n), _bools(n, 0, n), _bools(n))
    assert (res.last_type[:10] == 0).all()
    assert (res.last_type[10:] == 1).all()
    assert res.last_start[10] == LOW[10]
    assert res.last_str[10] == 1
    assert res.last_sig[10] == 10
    assert res.last_sig[9] == NO_SIGNAL_BAR
    assert (res.pend_dir == 0).all()


def test_second_signal_adds_three_to_strength():
    n = 21
    res = _run(_bools(n, 10, n), _bools(n), _bools(n, 12, n), _bools(n))
    assert res.pend_dir[10] == 1
    assert res.pend_dir[11] == 1
    assert res.pend_dir[12] == 0
    assert res.last_type[11] == 0
    assert res.last_type[12] == 1
    assert res.last_str[12] == 4
    assert res.last_sig[12] == 12


def test_pending_cancelled_when_raw_signal_disappears():
    n = 20
    res = _run(_bools(n, 10, 15), _bools(n), _bools(n), _bools(n))
    assert res.pend_dir[14] == 1
    assert res.pend_dir[15] == 0
    assert (res.last_type == 0).all()


def test_pending_expires_after_max_wait_and_reopens_while_raw_stays():
    gate = ConfirmGate()
    for i in range(10, 23):
        gate.step(i, True, False, False, False, 1, 1, 1.0, 1.0)
    assert gate.pend_bar == 10
    gate.step(23, True, False, False, False, 1, 1, 1.0, 1.0)
    assert gate.pend_bar == 23
    assert gate.pend_dir == 1
    assert gate.last_type == 0.0


def test_cooldown_blocks_early_signal_until_bar_ten():
    n = 15
    res = _run(_bools(n, 5, n), _bools(n), _bools(n, 0, n), _bools(n))
    assert res.last_type[9] == 0
    assert res.last_type[10] == 1
    assert res.last_sig[10] == 10


def test_same_direction_not_repeated_until_opposite_confirmed():
    n = 30
    res = _run(_bools(n, 10, n), _bools(n), _bools(n, 0, n), _bools(n))
    assert (res.last_sig[10:] == 10).all()
    assert (res.pend_dir == 0).all()


def test_reversal_and_cooldown_after_signal():
    raw_long = _bools(N, only=[10]) | _bools(N, 30, N)
    raw_short = _bools(N, 25, 31)
    res = _run(raw_long, raw_short, _bools(N, 0, N), _bools(N, 0, N))
    assert res.last_type[10] == 1
    assert res.last_type[25] == -1
    assert res.last_start[25] == HIGH[25]
    assert res.last_type[30] == -1
    assert res.last_type[34] == -1
    assert res.last_type[35] == 1
    assert res.last_sig[35] == 35


def test_short_signal_uses_high_as_start():
    n = 15
    res = _run(_bools(n), _bools(n, 10, n), _bools(n), _bools(n, 0, n))
    assert res.last_type[10] == -1
    assert res.last_start[10] == HIGH[10]


def test_two_confirmation_bars_need_consecutive_tf_ok():
    n = 20
    tf_long = _bools(n, only=[10, 12, 13])
    res = _run(_bools(n, 10, n), _bools(n), tf_long, _bools(n), strength=2, conf_bars=2)
    assert res.last_type[12] == 0
    assert res.last_type[13] == 1
    assert res.last_sig[13] == 13
    assert res.last_str[13] == 5


def test_initial_outputs_before_any_signal():
    n = 5
    res = _run(_bools(n), _bools(n), _bools(n), _bools(n))
    assert np.isnan(res.last_start).all()
    assert (res.last_sig == NO_SIGNAL_BAR).all()
    assert (res.last_str == 0).all()
