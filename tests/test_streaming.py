"""Adım adım gösterge zinciri: toplu `symbol_signals` ile bit düzeyinde aynı olmalı."""

from pathlib import Path

import numpy as np
import pytest

from alamet.golden import load_window
from alamet.levels import daily_open
from alamet.primitives import pine_atr
from alamet.streaming import DailyOpen, SymbolStream
from alamet.symbol import SymbolParams, symbol_signals

GOLDEN = Path(__file__).parent / "data" / "golden" / "window.npz"


def random_series(bars, seed, flat_every=0, zero_volume_every=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, bars)))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.001, bars))
    spread = close * rng.uniform(0.0005, 0.006, bars)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(50, 500, bars)
    if flat_every:
        flat = np.arange(bars) % flat_every == 7
        high[flat], low[flat], open_[flat] = close[flat], close[flat], close[flat]
    if zero_volume_every:
        volume[np.arange(bars) % zero_volume_every == 3] = 0.0
    time = 1_767_571_200 + np.arange(bars) * 900
    return open_, high, low, close, volume, time


def compare(series, params):
    open_, high, low, close, volume, time = series
    batch = symbol_signals(open_, high, low, close, volume, time, params)
    atr = pine_atr(high, low, close, params.atr_len)
    stream = SymbolStream(params)
    steps = [
        stream.update(open_[i], high[i], low[i], close[i], volume[i], int(time[i]))
        for i in range(len(close))
    ]
    for m, gate in enumerate(batch.methods):
        got = np.array([step.methods[m] for step in steps])
        for column, name in enumerate(
            ("last_type", "last_start", "last_str", "pend_dir", "last_sig")
        ):
            assert np.array_equal(got[:, column], getattr(gate, name), equal_nan=True), (m, name)
    assert np.array_equal([s.final_pct for s in steps], batch.mtf.final_pct)
    assert np.array_equal([s.atr for s in steps], atr, equal_nan=True)
    return batch, steps


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_stream_equals_batch_on_random_series(seed):
    compare(random_series(2500, seed), SymbolParams())


def test_stream_equals_batch_with_flat_bars_and_zero_volume():
    compare(random_series(2500, 9, flat_every=11, zero_volume_every=17), SymbolParams())


@pytest.mark.parametrize(
    "params",
    [
        SymbolParams(tf_on=False),
        SymbolParams(conf_bars=2, cool=4, pend_max=6),
        SymbolParams(ha_multipliers=(2, 3, 5, 8), tf_need=2),
        SymbolParams(td_len=14, mo_len=8, mo_z_len=30, hv_len=9, va_slow=20, va_fast=6),
        SymbolParams(signal_tf_seconds=300),
    ],
)
def test_stream_equals_batch_for_other_parameters(params):
    compare(random_series(2200, 5), params)


def test_the_series_actually_produces_signals_so_the_comparison_is_meaningful():
    batch, _ = compare(random_series(2500, 1), SymbolParams())
    assert all((gate.last_type != 0).sum() > 20 for gate in batch.methods)


@pytest.mark.skipif(not GOLDEN.exists(), reason="altın referans penceresi yok")
def test_stream_equals_batch_on_real_market_data_for_every_symbol():
    window = load_window(GOLDEN)
    for k in range(window.n):
        series = tuple(
            getattr(window, name)[:, k] for name in ("open", "high", "low", "close", "volume")
        ) + (window.time,)
        compare(series, SymbolParams())


@pytest.mark.parametrize("hour", [0, 5, 21])
def test_daily_open_stream_equals_batch(hour):
    open_, _, _, _, _, time = random_series(2000, 3)
    want = daily_open(time, open_, hour)
    stream = DailyOpen(hour)
    got = np.array([stream.update(int(time[i]), open_[i]) for i in range(len(time))])
    assert np.array_equal(got, want, equal_nan=True)
