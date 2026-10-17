"""Günlük paket: seviyeler, Devisso, 15 dk barlara iniş ve panel puanları."""

import numpy as np
import pytest

from alamet.daily import (
    HISTORY,
    DailyUniverse,
    daily_from_bars,
    daily_pack,
    day_inputs,
    map_live,
    map_to_bars,
    window_daily,
)
from alamet.panel import PanelParams, _extra_scores
from alamet.universe import Universe

DAY = 86400
# 2026-01-05 bir Pazartesi (UTC 00:00)
T0 = 1767571200


def _daily(closes, n_symbols=1):
    closes = np.array(closes, dtype=float)
    cols = np.column_stack([closes] * n_symbols)
    time = T0 + np.arange(len(closes)) * DAY
    return DailyUniverse(
        tuple(f"S{i}" for i in range(n_symbols)), time, cols, cols + 1, cols - 1, cols
    )


def test_pack_has_twenty_columns_and_weekly_open_resets_on_monday():
    daily = _daily(np.arange(100.0, 120.0))
    pack = daily_pack(daily)
    assert pack.shape == (20, 1, 20)
    assert pack[0, 0, 0] == 100.0 and pack[6, 0, 0] == 100.0
    assert pack[7, 0, 0] == 107.0
    assert pack[7, 0, 3] == pytest.approx(107.0)
    assert pack[7, 0, 5] == pytest.approx(106.0)


def test_pack_leading_nan_rows_are_ignored_like_unlisted_symbol():
    closes = [np.nan, np.nan, 10.0, 11.0, 12.0]
    pack = daily_pack(_daily(closes))
    assert np.isnan(pack[:2, 0, :]).all()
    assert pack[2, 0, 0] == 10.0


def test_history_mapping_uses_previous_day_except_last_bar_of_day():
    pack = np.arange(5, dtype=float)[:, None, None] * np.ones((5, 1, 20))
    daily_time = T0 + np.arange(5) * DAY
    bars = np.array([T0 + 2 * DAY, T0 + 2 * DAY + 900, T0 + 2 * DAY + DAY - 900, T0 + 3 * DAY])
    mapped = map_to_bars(pack, daily_time, bars, HISTORY)
    assert mapped[:, 0, 0].tolist() == [1.0, 1.0, 2.0, 2.0]


def test_mapping_gives_nan_for_missing_days_and_rejects_live_mode():
    pack = np.ones((2, 1, 20))
    daily_time = T0 + np.arange(2) * DAY
    far = np.array([T0 + 10 * DAY])
    assert np.isnan(map_to_bars(pack, daily_time, far)).all()
    with pytest.raises(NotImplementedError):
        map_to_bars(pack, daily_time, far, "live")
    with pytest.raises(ValueError):
        map_to_bars(pack, daily_time, far, "x")


def test_extra_scores_add_target_and_ease_bonus_with_own_gate():
    n = 3
    base = np.array([[1.0, 1.0, 1.0]])
    types = np.array([[1.0, 1.0, 1.0]])
    close = np.array([[100.0, 100.0, 100.0]])
    atr = np.array([[2.0, 2.0, 2.0]])
    own_ok = np.array([[True, True, False]])
    levels = np.full((1, n, 18), np.nan)
    levels[0, :, 0] = 110.0
    mapped = np.full((1, n, 20), np.nan)
    mapped[..., :18] = levels
    mapped[0, :, 18] = [1.0, 2.0, 3.0]
    mapped[0, :, 19] = [0.0, 50.0, 100.0]
    day = day_inputs(mapped)
    params = PanelParams(use_target=True, dr_rank=True)
    scores, dr_pct = _extra_scores(base, types, close, atr, own_ok, day, params)
    target = 0.25 * min(5.0, 8.0)
    assert scores[0, 0] == pytest.approx(1.0 + target + 1.0 * (1 - dr_pct[0, 0] / 100))
    assert scores[0, 2] == pytest.approx(1.0 + 1.0 * (1 - dr_pct[0, 2] / 100))
    assert dr_pct[0, 0] < dr_pct[0, 2]


def test_extra_scores_without_day_inputs_leave_base_untouched():
    base = np.array([[1.0, 2.0]])
    scores, dr_pct = _extra_scores(base, base, base, base, base > 0, None, PanelParams())
    assert (scores == base).all() and np.isnan(dr_pct).all()


def test_daily_from_bars_aggregates_full_utc_days_only():
    bars = 96 * 2 + 10
    time = T0 - 900 * 5 + np.arange(bars) * 900
    base = np.arange(bars, dtype=float)[:, None] + 100.0
    uni = Universe(("S0",), time, base, base + 2, base - 3, base + 0.5, base * 0 + 1)
    daily = daily_from_bars(uni)
    assert list(daily.time) == [T0, T0 + DAY]
    first = 5
    assert daily.open[0, 0] == base[first, 0]
    assert daily.close[0, 0] == base[first + 95, 0] + 0.5
    assert daily.high[1, 0] == base[first + 96 + 95, 0] + 2
    assert daily.low[1, 0] == base[first + 96, 0] - 3


def _live_universe(days=17, extra_bars=40, seed=11):
    rng = np.random.default_rng(seed)
    bars = days * 96 + extra_bars
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, (bars, 2)), axis=0))
    spread = close * 0.004
    open_ = np.vstack([close[:1], close[:-1]])
    time = T0 + np.arange(bars) * 900
    return Universe(
        ("A", "B"), time, open_, np.maximum(open_, close) + spread,
        np.minimum(open_, close) - spread, close, rng.uniform(1, 2, (bars, 2)),
    )  # fmt: skip


def test_live_pack_equals_history_pack_on_the_last_bar_of_each_day():
    uni = _live_universe(extra_bars=0)
    daily = daily_from_bars(uni)
    pack = daily_pack(daily)
    live = map_live(pack, daily, uni)
    history = map_to_bars(pack, daily.time, uni.time)
    last = (uni.time % DAY) == DAY - 900
    both = ~np.isnan(live[last]) & ~np.isnan(history[last])
    assert both.sum() > 0
    assert np.allclose(live[last][both], history[last][both], atol=1e-9)


def test_live_pack_uses_partial_day_high_low_and_resets_on_new_period():
    uni = _live_universe()
    daily = daily_from_bars(uni)
    pack = daily_pack(daily)
    live = map_live(pack, daily, uni)
    day = uni.time // DAY * DAY
    # Pazartesi 12 Ocak (yeni hafta): açılış günün açılışı, önceki hafta high = 5 Ocak haftası
    monday = T0 + 7 * DAY
    first_bar = int(np.argmax(day == monday))
    midday = first_bar + 20
    assert live[midday, 0, 0] == uni.open[first_bar, 0]
    assert live[midday, 0, 1] == uni.high[first_bar : midday + 1, 0].max()
    week_one = (day >= T0) & (day < monday)
    assert live[midday, 0, 3] == uni.high[week_one, 0].max()
    # Salı: hafta high'ı = max(pazartesi tamamı, salının kısmi high'ı)
    tuesday = monday + DAY
    mid = int(np.argmax(day == tuesday)) + 30
    expected = max(
        uni.high[day == monday, 0].max(),
        uni.high[int(np.argmax(day == tuesday)) : mid + 1, 0].max(),
    )
    assert live[mid, 0, 1] == expected


def test_live_pack_handles_the_unfinished_current_day():
    partial = _live_universe(extra_bars=40)
    daily = daily_from_bars(partial)
    assert daily.time[-1] == T0 + 16 * DAY
    live = map_live(daily_pack(daily), daily, partial)
    assert not np.isnan(live[-1, :, 18]).any()
    full = _live_universe(extra_bars=96)
    full_daily = daily_from_bars(full)
    assert full_daily.time[-1] == T0 + 17 * DAY
    history = map_to_bars(daily_pack(full_daily), full_daily.time, full.time)
    # Aynı gün tamamlanınca günün son barındaki Devisso sütunları gelişen günle aynıdır.
    assert np.allclose(live[-1, :, 18:], history[-1, :, 18:], equal_nan=True)


def test_window_daily_keeps_the_last_calc_bars_over_96_days_like_tradingview():
    day = 86400
    daily = DailyUniverse(("X",), np.arange(60) * day, *(np.ones((60, 1)) for _ in range(4)))
    last = 55 * day + 11 * 3600
    for start_hour in (7, 19):
        kept = window_daily(daily, 14 * day + start_hour * 3600, last, 4000)
        assert kept.time[0] == (55 - 40) * day
    assert window_daily(daily, 14 * day + 19 * 3600).time[0] == 15 * day
