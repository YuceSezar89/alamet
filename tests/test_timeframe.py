"""Sinyal zaman dilimi: 15 dk dışındaki TF'ler (5 dk, 1 dk) için bar sayıları, gün sınırları,
HA kovaları, veri katmanı ve uçtan uca çalışma."""

import warnings

import numpy as np
import pandas as pd
import pytest

from alamet import binance_data as bd
from alamet.daily import (
    DailyUniverse,
    daily_from_bars,
    daily_pack,
    day_inputs,
    map_to_bars,
    window_daily,
)
from alamet.engine import TradeParams
from alamet.stages import panel_params, with_signal_tf
from alamet.symbol import SymbolParams, symbol_signals
from alamet.system import run_system
from alamet.timeframe import (
    SignalTf,
    check_daily_open,
    enforce_chart_tf,
    from_interval,
    ss_lag,
)
from alamet.universe import Universe, resample_universe
from alamet.validate import validate_timeframe

DAY = 86400
BASE = 1_767_571_200 - (1_767_571_200 % DAY)


def universe(seconds, bars, n=3, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, (bars, n)), axis=0))
    spread = close * 0.001
    open_ = np.vstack([close[:1], close[:-1]])
    time = BASE + np.arange(bars) * seconds
    return Universe(
        tuple(f"S{i}" for i in range(n)), time, open_, np.maximum(open_, close) + spread,
        np.minimum(open_, close) - spread, close, rng.uniform(100, 200, (bars, n)),
    )  # fmt: skip


def test_signal_tf_derives_bar_counts_and_intervals():
    five, minute, default = SignalTf(300), SignalTf(60), SignalTf()
    assert (default.bars_per_day, five.bars_per_day, minute.bars_per_day) == (96, 288, 1440)
    assert (default.interval, five.interval, minute.interval) == ("15m", "5m", "1m")
    assert five.millis == 300_000 and five.last_bar_offset == DAY - 300
    assert minute.bars_per_year == 1440 * 365 and from_interval("5m") == five
    for bad in (0, 7 * 60, 1000, 90000):
        with pytest.raises(ValueError):
            SignalTf(bad)
    with pytest.raises(ValueError):
        from_interval("7m")


def test_chart_period_rules_follow_pine():
    assert ss_lag(300, 900) == 1 and ss_lag(900, 900) == 0
    enforce_chart_tf(300, 900)
    enforce_chart_tf(900, 900)
    with pytest.raises(ValueError):
        enforce_chart_tf(1800, 900)
    enforce_chart_tf(1800, 900, enforce=False)
    assert check_daily_open(0, 14400) and check_daily_open(21, 3600)
    assert not check_daily_open(21, 7200)


def test_daily_candles_need_every_bar_of_the_day_at_any_tf():
    for seconds in (900, 300, 60):
        per_day = DAY // seconds
        data = universe(seconds, 2 * per_day + per_day // 2)
        daily = daily_from_bars(data, seconds)
        assert list(daily.time) == [BASE, BASE + DAY]
        assert np.array_equal(daily.open[0], data.open[0])
        assert np.array_equal(daily.close[1], data.close[2 * per_day - 1])
        assert np.array_equal(daily.high[0], data.high[:per_day].max(axis=0))
        assert np.array_equal(daily.low[1], data.low[per_day : 2 * per_day].min(axis=0))


def test_history_mapping_hands_the_closed_day_to_the_last_bar_only():
    seconds = 300
    per_day = DAY // seconds
    data = universe(seconds, 3 * per_day)
    daily = daily_from_bars(data, seconds)
    pack = np.arange(len(daily.time) * 3 * 20, dtype=float).reshape(len(daily.time), 3, 20)
    mapped = map_to_bars(pack, daily.time, data.time, tf_seconds=seconds)
    day1 = slice(per_day, 2 * per_day)
    assert np.isnan(mapped[: per_day - 1]).all()
    assert np.array_equal(mapped[per_day - 1], pack[0])
    assert np.array_equal(mapped[day1][0], pack[0]) and np.array_equal(mapped[day1][-2], pack[0])
    assert np.array_equal(mapped[2 * per_day - 1], pack[1])


def test_calc_window_keeps_calc_bars_over_bars_per_day_days():
    days = np.arange(60) * DAY
    daily = DailyUniverse(("X",), days, *(np.ones((60, 1)) for _ in range(4)))
    last = 55 * DAY + 11 * 3600
    kept15 = window_daily(daily, 0, last, 4000, 900)
    kept5 = window_daily(daily, 0, last, 4000, 300)
    kept1 = window_daily(daily, 0, last, 4000, 60)
    assert kept15.time[0] == (55 - 40) * DAY and kept5.time[0] == (55 - 12) * DAY
    assert kept1.time[0] == (55 - 1) * DAY


def test_heikin_buckets_scale_with_the_signal_tf():
    five = universe(300, 900, n=1)
    ref = Universe(
        five.symbols, five.time * 3, five.open, five.high, five.low, five.close, five.volume
    )
    cols = (five.open[:, 0], five.high[:, 0], five.low[:, 0], five.close[:, 0], five.volume[:, 0])
    a = symbol_signals(*cols, five.time, SymbolParams(signal_tf_seconds=300))
    b = symbol_signals(*cols, ref.time, SymbolParams(signal_tf_seconds=900))
    c = symbol_signals(*cols, five.time, SymbolParams(signal_tf_seconds=900))
    assert np.array_equal(a.mtf.final_pct, b.mtf.final_pct, equal_nan=True)
    assert not np.array_equal(a.mtf.final_pct, c.mtf.final_pct, equal_nan=True)


def test_resample_builds_only_complete_aligned_buckets():
    fine = universe(60, 5 * 40 + 2)
    five = resample_universe(fine, 300)
    assert five.bar_seconds == 300 and five.bars == 40
    assert np.array_equal(five.open[3], fine.open[15])
    assert np.array_equal(five.close[3], fine.close[19])
    assert np.array_equal(five.high[3], fine.high[15:20].max(axis=0))
    assert np.array_equal(five.low[3], fine.low[15:20].min(axis=0))
    assert np.allclose(five.volume[3], fine.volume[15:20].sum(axis=0))
    shifted = Universe(
        fine.symbols, fine.time[2:], fine.open[2:], fine.high[2:], fine.low[2:], fine.close[2:],
        fine.volume[2:],
    )  # fmt: skip
    assert resample_universe(shifted, 300).bars == 39
    with pytest.raises(ValueError):
        resample_universe(fine, 150)
    with pytest.raises(ValueError):
        resample_universe(five, 60)


def test_validate_timeframe_matches_pine_enforce_and_requires_resampling():
    params = panel_params()
    assert validate_timeframe(params, universe(900, 10)).seconds == 900
    with pytest.raises(ValueError, match="BÜYÜK"):
        validate_timeframe(params, universe(1800, 10))
    with pytest.raises(ValueError, match="resample"):
        validate_timeframe(params, universe(300, 10))
    five = with_signal_tf(params, 300)
    assert validate_timeframe(five, universe(300, 10)).seconds == 300
    with pytest.raises(ValueError):
        validate_timeframe(with_signal_tf(params, 420), universe(900, 10))
    from dataclasses import replace  # pylint: disable=import-outside-toplevel

    late = replace(with_signal_tf(params, 14400), daily_open_hour=21)
    with pytest.warns(UserWarning):
        validate_timeframe(late, universe(14400, 10))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_timeframe(replace(params, daily_open_hour=21), universe(900, 10))


@pytest.mark.parametrize("seconds", [300, 60])
def test_whole_system_runs_at_other_signal_tfs(seconds):
    per_day = DAY // seconds
    data = universe(seconds, per_day * 3, n=6, seed=7)
    panel = with_signal_tf(panel_params(n_top=3, max_slots=2), seconds)
    daily = daily_from_bars(data, seconds)
    pack = daily_pack(daily, panel.dr_rsi_len, panel.dr_look)
    day = day_inputs(map_to_bars(pack, daily.time, data.time, tf_seconds=seconds))
    result = run_system(
        data, panel, TradeParams(max_slots=2), True, day=day, collect_stats=True,
        collect_events=True,
    )  # fmt: skip
    assert len(result.trades) > 0 and len(result.events) > 0
    with pytest.raises(ValueError, match="ince"):
        run_system(data, panel_params(n_top=3, max_slots=2), TradeParams(max_slots=2), True)


def test_one_minute_data_resampled_to_five_runs_like_native_five_minute_data():
    fine = universe(60, 5 * 600, n=6, seed=11)
    five = resample_universe(fine, 300)
    panel = with_signal_tf(panel_params(n_top=3, max_slots=2), 300)
    result = run_system(five, panel, TradeParams(max_slots=2), True)
    assert result.sides.shape == (five.bars, 6)


def cache_frame(seconds, bars, offset=0.0):
    times = (BASE + np.arange(bars) * seconds) * 1000
    base = 100.0 + offset + np.arange(bars, dtype=float)
    return pd.DataFrame(
        {
            "open_time": times.astype(np.int64), "open": base, "high": base + 1, "low": base - 1,
            "close": base + 0.5, "volume": np.full(bars, 5.0), "quote_volume": base * 5,
        }
    )  # fmt: skip


def test_binance_cache_loads_infers_and_checks_gaps_at_the_cache_tf(tmp_path):
    for base, offset in (("BTC", 0.0), ("ETH", 1.0)):
        bd.save_cache(tmp_path, base, cache_frame(300, 50, offset))
    five = bd.infer_tf(tmp_path, ("BTC", "ETH"))
    assert five == SignalTf(300) and bd.default_root(five).name == "binance_5m"
    data = bd.load_universe_binance(tmp_path, ("BTC", "ETH"), tf=five)
    assert data.bars == 50 and data.bar_seconds == 300
    with pytest.raises(ValueError, match="eksik"):
        bd.load_universe_binance(tmp_path, ("BTC", "ETH"))
    assert bd.integrity("BTC", bd.load_cache(tmp_path, "BTC"), five).ok
    holed = bd.load_cache(tmp_path, "BTC").drop(index=10).reset_index(drop=True)
    assert bd.integrity("BTC", holed, five).gaps == [(int(holed.open_time[9]), 1)]
