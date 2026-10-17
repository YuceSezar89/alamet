"""C5 — sinyal modu seçici: elle izlenmiş senaryolar."""

import numpy as np
import pytest

from alamet.selector import SelectorParams, SignalSelector, parse_seq_order, run_selector

NAN = np.nan


def _run(rows, params, track_last=True):
    """rows: bar başına 4 yöntemin (yön, başlangıç, güç, bekleyen, sinyal barı) listesi."""
    arr = np.array(rows, dtype=float)[:, None, :, :]
    return run_selector(
        arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3], arr[..., 4], params, track_last
    )


def _method(direction=0, start=NAN, strength=0, pend=0, sig=-100000):
    return (direction, start, strength, pend, sig)


FLAT = _method()


def test_parse_seq_order_dedups_and_validates():
    assert parse_seq_order("1,2,4,3") == (0, 1, 3, 2)
    assert parse_seq_order("1, 1,2") == (0, 1)
    with pytest.raises(ValueError):
        parse_seq_order("1,5")
    with pytest.raises(ValueError):
        parse_seq_order("a")


def test_params_require_an_enabled_method_and_valid_mode():
    with pytest.raises(ValueError):
        SelectorParams(enabled=(False,) * 4)
    with pytest.raises(ValueError):
        SelectorParams(mode=7)


def test_single_mode_follows_chosen_method_every_bar():
    rows = [
        [FLAT, _method(1, 10.0, 2, 0, 5), FLAT, FLAT],
        [FLAT, _method(-1, 20.0, 1, 0, 6), FLAT, FLAT],
        [FLAT, _method(0, 20.0, 1, 0, 6), FLAT, FLAT],
    ]
    out = _run(rows, SelectorParams(mode=1))
    assert out["sig_type"][:, 0].tolist() == [1, -1, 0]
    assert out["start"][0, 0] == 10.0 and out["start"][1, 0] == 20.0
    assert np.isnan(out["start"][2, 0])
    assert out["strength"][:, 0].tolist() == [2, 1, 0]
    assert out["method"][:, 0].tolist() == [1, 1, 1]


def test_concurrent_needs_enough_methods_and_keeps_last_pick_when_unmet():
    rows = [
        [_method(1, 1.0, 1, 0, 3), FLAT, FLAT, FLAT],
        [_method(1, 1.0, 1, 0, 3), _method(1, 2.0, 2, 0, 4), FLAT, FLAT],
        [_method(1, 1.0, 1, 0, 3), FLAT, FLAT, FLAT],
    ]
    out = _run(rows, SelectorParams(mode=4, conf_n=2))
    assert out["sig_type"][:, 0].tolist() == [0, 1, 1]
    assert out["method"][:, 0].tolist() == [0, 1, 1]
    assert out["start"][1, 0] == 2.0


def test_concurrent_prefers_latest_signal_and_blocks_on_opposite_majority():
    rows = [[_method(1, 1.0, 0, 0, 5), _method(1, 2.0, 0, 0, 5), FLAT, FLAT]]
    assert _run(rows, SelectorParams(mode=4))["method"][0, 0] == 1
    split = [[_method(1, 1.0, 0, 0, 5), _method(1, 2.0, 0, 0, 6), _method(-1, 3.0, 0, 0, 7), FLAT]]
    assert _run(split, SelectorParams(mode=4, conf_n=2))["sig_type"][0, 0] == 1
    tie = [[_method(1, 1.0, 0, 0, 5), _method(-1, 2.0, 0, 0, 6), FLAT, FLAT]]
    assert _run(tie, SelectorParams(mode=4, conf_n=1))["sig_type"][0, 0] == 0


def test_concurrent_need_is_capped_by_enabled_count():
    rows = [[_method(1, 1.0, 0, 0, 5), FLAT, FLAT, FLAT]]
    params = SelectorParams(mode=4, enabled=(True, False, False, False), conf_n=4)
    assert _run(rows, params)["sig_type"][0, 0] == 1


def test_sequential_requires_order_direction_and_window():
    ok = [[_method(1, 1.0, 1, 0, 10), _method(1, 2.0, 1, 0, 12), _method(1, 3.0, 2, 0, 15), FLAT]]
    params = SelectorParams(mode=5, seq_order=(0, 1, 2, 3), seq_n=3, seq_win=24)
    out = _run(ok, params)
    assert out["sig_type"][0, 0] == 1 and out["method"][0, 0] == 2
    wrong_order = [
        [_method(1, 1.0, 1, 0, 12), _method(1, 2.0, 1, 0, 10), _method(1, 3.0, 1, 0, 15), FLAT]
    ]
    assert _run(wrong_order, params)["sig_type"][0, 0] == 0
    too_wide = [
        [_method(1, 1.0, 1, 0, 10), _method(1, 2.0, 1, 0, 12), _method(1, 3.0, 1, 0, 40), FLAT]
    ]
    assert _run(too_wide, params)["sig_type"][0, 0] == 0
    opposite = [
        [_method(1, 1.0, 1, 0, 10), _method(-1, 2.0, 1, 0, 12), _method(1, 3.0, 1, 0, 15), FLAT]
    ]
    assert _run(opposite, params)["sig_type"][0, 0] == 0


def test_sequential_skips_disabled_methods_and_uses_next_in_order():
    rows = [[_method(1, 1.0, 1, 0, 10), _method(1, 2.0, 1, 0, 11), _method(1, 3.0, 1, 0, 12), FLAT]]
    params = SelectorParams(
        mode=5, enabled=(True, False, True, True), seq_order=(0, 1, 2, 3), seq_n=2, seq_win=5
    )
    assert _run(rows, params)["method"][0, 0] == 2


def test_pending_in_conf_modes_is_first_enabled_nonzero_and_not_sticky():
    rows = [
        [FLAT, _method(0, NAN, 0, -1), _method(0, NAN, 0, 1), FLAT],
        [FLAT, FLAT, FLAT, FLAT],
    ]
    out = _run(rows, SelectorParams(mode=4))
    assert out["pending"][:, 0].tolist() == [-1, 0]


def test_pending_in_single_mode_follows_selected_method():
    rows = [[_method(0, NAN, 0, 1), _method(0, NAN, 0, -1), FLAT, FLAT]]
    assert _run(rows, SelectorParams(mode=1))["pending"][0, 0] == -1


def test_unconfirmed_last_bar_does_not_advance_selection():
    rows = [
        [_method(1, 1.0, 1, 0, 3), FLAT, FLAT, FLAT],
        [_method(-1, 9.0, 2, 0, 4), FLAT, FLAT, FLAT],
    ]
    out = _run(rows, SelectorParams(mode=0), track_last=False)
    assert out["sig_type"][:, 0].tolist() == [1, 1]
    assert out["start"][1, 0] == 1.0


def test_symbols_are_independent():
    selector = SignalSelector(2, SelectorParams(mode=0))
    dirs = np.array([[1.0, 0, 0, 0], [-1.0, 0, 0, 0]])
    zeros = np.zeros((2, 4))
    step = selector.step(dirs, np.full((2, 4), 5.0), zeros, zeros, zeros)
    assert step.sig_type.tolist() == [1.0, -1.0]


def _auto(params, cnt, net, dirs=(1, 1, 0, 0)):
    selector = SignalSelector(1, params)
    d = np.array([dirs], dtype=float)
    starts = np.array([[1.0, 2.0, 3.0, 4.0]])
    zeros = np.zeros((1, 4))
    return selector.step(
        d, starts, zeros, zeros, zeros, True, np.array([cnt]), np.array([net], dtype=float)
    )


def test_auto_picks_best_damped_net_among_methods_with_enough_trades():
    params = SelectorParams(mode=6)
    step = _auto(params, [3, 5, 2, 9], [4.0, 7.0, 99.0, 1.0])
    assert step.method[0] == 1 and step.auto[0] == 1
    assert step.start[0] == 2.0


def test_auto_blocks_when_no_method_is_positive_and_falls_back_without_data():
    params = SelectorParams(mode=6)
    blocked = _auto(params, [3, 3, 3, 3], [-1.0, -2.0, 0.0, -3.0])
    assert blocked.sig_type[0] == 0 and blocked.auto[0] == -1
    open_ = _auto(SelectorParams(mode=6, auto_block=False), [3, 3, 3, 3], [-1.0] * 4)
    assert open_.sig_type[0] == 1 and open_.method[0] == 0
    cold = _auto(params, [0, 1, 2, 0], [5.0, 5.0, 5.0, 5.0])
    assert cold.sig_type[0] == 1 and cold.method[0] == 0


def test_auto_respects_disabled_methods_and_min_net():
    params = SelectorParams(mode=6, enabled=(False, True, True, True), auto_min_net=5.0)
    step = _auto(params, [3, 3, 3, 3], [100.0, 4.0, 6.0, 5.0])
    assert step.method[0] == 2
