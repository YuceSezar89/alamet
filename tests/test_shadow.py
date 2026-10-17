"""C6 — gölge kasa: elle hesaplanmış işlemler."""

import numpy as np
import pytest

from alamet.selector import SelectorParams
from alamet.shadow import ShadowBook, ShadowParams, run_with_shadow, trade_r, trade_usd

NO_FEE = ShadowParams(fee_in=0.0, fee_out=0.0)


def _step(book, idx, close, atr, dirs):
    book.step(idx, np.array([close]), np.array([atr]), np.array([dirs]))


def test_trade_usd_and_r_include_fees_and_slippage():
    params = ShadowParams(fee_in=0.02, fee_out=0.02, slippage=0.01)
    assert trade_usd(params, 1000.0, 1, 100.0, 110.0) == pytest.approx(
        1000 * 0.1 - 1000 * 0.0003 - 1000 * 1.1 * 0.0003
    )
    assert trade_r(params, -1, 100.0, 2.0, 96.0) == pytest.approx(
        (4.0 - (0.0003 * 100 + 0.0003 * 96)) / 2.0
    )
    assert trade_r(params, 1, 100.0, 0.0, 110.0) == 0.0


def test_first_valid_bar_only_records_previous_direction():
    book = ShadowBook(1, NO_FEE)
    _step(book, 1, 100.0, 2.0, [1, 1, 0, 0, 1])
    assert book.side[0].tolist() == [0] * 5
    assert book.prev[0].tolist() == [1, 1, 0, 0, 1]


def test_entry_exit_and_risk_sizing():
    book = ShadowBook(1, NO_FEE)
    _step(book, 1, 100.0, 2.0, [0, 0, 0, 0, 0])
    _step(book, 2, 100.0, 2.0, [1, 0, 0, 0, 0])
    assert book.side[0, 0] == 1 and book.entry[0, 0] == 100.0
    assert book.risk[0, 0] == pytest.approx(3.0)
    assert book.notional[0, 0] == pytest.approx(min(1000 * 0.005 / 0.03, 5000))
    _step(book, 3, 106.0, 2.0, [0, 0, 0, 0, 0])
    assert book.side[0, 0] == 0 and book.count[0, 0] == 1 and book.wins[0, 0] == 1
    assert book.net[0, 0] == pytest.approx(book.notional[0, 0] * 0.06)
    assert book.sum_r[0, 0] == pytest.approx(2.0)


def test_reversal_closes_then_opens_opposite():
    book = ShadowBook(1, NO_FEE)
    _step(book, 1, 100.0, 2.0, [1, 0, 0, 0, 0])
    _step(book, 2, 100.0, 2.0, [-1, 0, 0, 0, 0])
    _step(book, 3, 95.0, 2.0, [1, 0, 0, 0, 0])
    assert book.count[0, 0] == 1 and book.wins[0, 0] == 1
    assert book.side[0, 0] == 1 and book.entry[0, 0] == 95.0


def test_notional_is_capped_by_max_leverage_and_fixed_margin_mode():
    tight = ShadowBook(1, ShadowParams(risk_pct=100.0))
    _step(tight, 1, 100.0, 0.1, [0] * 5)
    _step(tight, 2, 100.0, 0.1, [1, 0, 0, 0, 0])
    assert tight.notional[0, 0] == pytest.approx(5000.0)
    fixed = ShadowBook(1, ShadowParams(risk_mode=False, fix_margin=50.0, leverage=3.0))
    _step(fixed, 1, 100.0, 2.0, [0] * 5)
    _step(fixed, 2, 100.0, 2.0, [1, 0, 0, 0, 0])
    assert fixed.notional[0, 0] == pytest.approx(150.0)


def test_invalid_price_or_atr_skips_bar():
    book = ShadowBook(1, NO_FEE)
    _step(book, 1, np.nan, 2.0, [1] * 5)
    _step(book, 2, 100.0, 0.0, [1] * 5)
    assert (book.prev == 9).all()


def test_decay_applies_to_damped_net_only_on_interval():
    book = ShadowBook(1, ShadowParams(fee_in=0.0, fee_out=0.0, decay=0.5, decay_bars=2))
    _step(book, 1, 100.0, 2.0, [0] * 5)
    _step(book, 2, 100.0, 2.0, [1, 0, 0, 0, 0])
    _step(book, 3, 110.0, 2.0, [0] * 5)
    net = book.net[0, 0]
    assert book.net_d[0, 0] == pytest.approx(net)
    _step(book, 4, 110.0, 2.0, [0] * 5)
    assert book.net_d[0, 0] == pytest.approx(net * 0.5)
    assert book.net[0, 0] == pytest.approx(net)


def test_run_with_shadow_auto_mode_unlocks_after_enough_profitable_trades():
    bars = 60
    close = np.linspace(100, 160, bars)[:, None]
    atr = np.full((bars, 1), 1.0)
    direction = np.zeros((bars, 1, 4))
    direction[:, 0, 0] = np.where(np.arange(bars) % 10 < 5, 1.0, 0.0)
    gates = {
        "dirs": direction,
        "starts": np.full((bars, 1, 4), 100.0),
        "strs": np.zeros((bars, 1, 4)),
        "pends": np.zeros((bars, 1, 4)),
        "sigs": np.tile(np.arange(bars, dtype=float)[:, None, None], (1, 1, 4)),
    }
    out = run_with_shadow(
        gates, close, atr, (SelectorParams(mode=6, auto_min_n=2),), NO_FEE, track_last=True
    )
    assert out["book_count"][-1, 0, 0] >= 2
    assert out["s0_auto"][0, 0] == -1
    assert out["s0_auto"][-1, 0] == 0
    assert out["s0_sig_type"].shape == (bars, 1)


def test_last_loss_records_bar_index_of_losing_close_only():
    book = ShadowBook(1, NO_FEE)
    assert (book.last_loss == -100000).all()
    _step(book, 1, 100.0, 2.0, [0] * 5)
    _step(book, 2, 100.0, 2.0, [1, -1, 0, 0, 0])
    _step(book, 3, 105.0, 2.0, [0] * 5)
    assert book.last_loss[0, 0] == -100000
    assert book.last_loss[0, 1] == 3
