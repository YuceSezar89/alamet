"""A1 — pine_tr / pine_rma / pine_atr / pine_sma elle hesaplanmış değerlerle."""

import numpy as np

from alamet.primitives import pine_atr, pine_rma, pine_sma, pine_tr

HIGH = np.array([10.0, 12.0, 11.0, 13.0])
LOW = np.array([8.0, 9.0, 9.0, 10.0])
CLOSE = np.array([9.0, 11.0, 10.0, 12.0])


def test_tr_first_bar_is_high_minus_low():
    assert pine_tr(HIGH, LOW, CLOSE)[0] == 2.0


def test_tr_values():
    np.testing.assert_allclose(pine_tr(HIGH, LOW, CLOSE), [2.0, 3.0, 2.0, 3.0])


def test_tr_uses_gap_against_previous_close():
    tr = pine_tr(np.array([10.0, 20.0]), np.array([9.0, 19.0]), np.array([9.5, 19.5]))
    assert tr[1] == 10.5


def test_rma_seed_is_sma_then_wilder():
    out = pine_rma(np.array([2.0, 3.0, 2.0, 3.0]), 2)
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1:], [2.5, 2.25, 2.625])


def test_atr_matches_hand_calculation():
    out = pine_atr(HIGH, LOW, CLOSE, 2)
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1:], [2.5, 2.25, 2.625])


def test_atr_short_series_all_nan():
    assert np.isnan(pine_atr(HIGH[:2], LOW[:2], CLOSE[:2], 3)).all()


def test_sma_warmup_and_values():
    out = pine_sma(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert np.isnan(out[:2]).all()
    np.testing.assert_allclose(out[2:], [2.0, 3.0, 4.0])
