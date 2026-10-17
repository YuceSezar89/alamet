"""Günlük bağlamın adım adım hali: toplu paket ve iki eşleme (geçmiş, canlı) ile bit düzeyinde aynı."""

from pathlib import Path

import numpy as np
import pytest

from alamet.daily import (
    DailyUniverse,
    _symbol_pack,
    daily_from_bars,
    daily_pack,
    map_live,
    map_to_bars,
    window_daily,
)
from alamet.daily_stream import DailyContext, SymbolDaily
from alamet.golden import load_window
from alamet.universe import Universe

DAY = 86400
BASE = 1_767_571_200 - (1_767_571_200 % DAY)
GOLDEN = Path(__file__).parent / "data" / "golden" / "window.npz"


def market(n, days, seconds=900, seed=3, start_offset=0):
    per_day = DAY // seconds
    bars = days * per_day
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, (bars, n)), axis=0))
    open_ = np.vstack([close[:1], close[:-1]])
    spread = close * 0.002
    time = BASE + start_offset * seconds + np.arange(bars) * seconds
    return Universe(
        tuple(f"S{i}" for i in range(n)), time, open_, np.maximum(open_, close) + spread,
        np.minimum(open_, close) - spread, close, rng.uniform(100, 200, (bars, n)),
    )  # fmt: skip


def run_stream(universe, tf=900, first_day=None):
    context = DailyContext(universe.n, tf, first_day=first_day)
    history, live = [], []
    for i in range(universe.bars):
        context.on_bar(
            int(universe.time[i]), universe.open[i], universe.high[i], universe.low[i],
            universe.close[i],
        )  # fmt: skip
        history.append(context.history())
        live.append(context.live())
    return np.array(history), np.array(live)


def run_batch(universe, tf=900, first_day=None):
    daily = daily_from_bars(universe, tf)
    if first_day is not None:
        daily = window_daily(daily, first_day - 1, tf_seconds=tf)
    pack = daily_pack(daily)
    history = map_to_bars(pack, daily.time, universe.time, tf_seconds=tf)
    return history, map_live(pack, daily, universe)


def check(universe, tf=900, first_day=None):
    got_history, got_live = run_stream(universe, tf, first_day)
    want_history, want_live = run_batch(universe, tf, first_day)
    assert got_history.shape == want_history.shape
    assert np.array_equal(got_history, want_history, equal_nan=True), "geçmiş eşleme"
    assert np.array_equal(got_live, want_live, equal_nan=True), "canlı eşleme"
    return got_history, got_live


def test_stream_equals_batch_on_random_universe():
    history, live = check(market(5, 45))
    assert np.isfinite(history[-1]).sum() > 50 and np.isfinite(live[-1]).sum() > 50


def test_live_and_history_agree_on_the_last_bar_of_every_day():
    history, live = check(market(4, 40, seed=8))
    last = np.arange(95 + 96, len(history), 96)
    assert np.array_equal(history[last], live[last], equal_nan=True)


def test_stream_equals_batch_when_the_data_starts_in_the_middle_of_a_day_and_ends_partial():
    universe = market(4, 40, seed=5, start_offset=37)
    universe = Universe(
        universe.symbols, universe.time[:-50], universe.open[:-50], universe.high[:-50],
        universe.low[:-50], universe.close[:-50], universe.volume[:-50],
    )  # fmt: skip
    check(universe)


def test_stream_equals_batch_with_the_calc_window_day_cut():
    universe = market(4, 50, seed=2)
    check(universe, first_day=BASE + 12 * DAY)


@pytest.mark.parametrize("seconds", [300, 60])
def test_stream_equals_batch_at_other_timeframes(seconds):
    check(market(3, 24, seconds=seconds, seed=4), tf=seconds)


def with_nan(universe, symbol, start, stop):
    arrays = {
        name: getattr(universe, name).copy() for name in ("open", "high", "low", "close", "volume")
    }
    for array in arrays.values():
        array[start:stop, symbol] = np.nan
    return Universe(universe.symbols, universe.time, **arrays)


def test_stream_equals_batch_for_late_listed_and_gapped_symbols():
    universe = market(5, 50, seed=6)
    universe = with_nan(universe, 3, 0, 96 * 14 + 20)
    universe = with_nan(universe, 1, 96 * 30 + 10, 96 * 30 + 40)
    check(universe)


@pytest.mark.skipif(not GOLDEN.exists(), reason="altın referans penceresi yok")
def test_stream_equals_batch_on_the_real_tradingview_window():
    check(load_window(GOLDEN))


def test_symbol_daily_equals_the_batch_pack_for_random_closes_with_gaps():
    rng = np.random.default_rng(1)
    days = 200
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, days)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high, low = np.maximum(open_, close) * 1.01, np.minimum(open_, close) * 0.99
    time = BASE + np.arange(days) * DAY
    for blank in (slice(0, 0), slice(0, 9), slice(60, 63)):
        c, o, h, l = close.copy(), open_.copy(), high.copy(), low.copy()
        for arr in (c, o, h, l):
            arr[blank] = np.nan
        want = _symbol_pack(time, o, h, l, c)
        sym = SymbolDaily()
        got = np.array([sym.close_day(int(time[i]), o[i], h[i], l[i], c[i]) for i in range(days)])
        assert np.array_equal(got, want, equal_nan=True), blank


def test_next_devisso_is_what_the_next_days_pack_will_start_with():
    rng = np.random.default_rng(4)
    days = 120
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, days)))
    sym = SymbolDaily()
    time = BASE + np.arange(days + 1) * DAY
    close = np.append(close, 100.0)
    packs = []
    for i in range(days + 1):
        if i:
            expected = sym.next_devisso()
        row = sym.close_day(int(time[i]), close[i], close[i], close[i], close[i])
        packs.append(row)
        if i:
            assert (row[18], row[19]) == pytest.approx(expected, nan_ok=True)


def test_empty_universe_of_unlisted_symbols_gives_nan_rows():
    universe = with_nan(market(2, 20, seed=7), 0, 0, 96 * 20)
    history, live = check(universe)
    assert np.isnan(history[:, 0]).all() and np.isnan(live[:, 0]).all()
