"""B1 — periyot seviyeleri (hafta/ay/yıl) ve günlük open."""

import numpy as np

from alamet.levels import daily_open, period_ids, period_levels

JAN_1_2024 = 1704067200
DAY = 86400


def test_week_starts_monday():
    ids = period_ids(np.array([JAN_1_2024, JAN_1_2024 + 6 * DAY, JAN_1_2024 + 7 * DAY]), "W")
    assert ids[0] == ids[1]
    assert ids[2] == ids[0] + 1


def test_week_boundary_sunday_to_monday_not_saturday():
    ids = period_ids(np.array([JAN_1_2024 - DAY, JAN_1_2024]), "W")
    assert ids[1] == ids[0] + 1


def test_month_and_year_boundaries():
    dec_31 = JAN_1_2024 - DAY
    jan_31 = JAN_1_2024 + 30 * DAY
    feb_1 = JAN_1_2024 + 31 * DAY
    months = period_ids(np.array([dec_31, JAN_1_2024, jan_31, feb_1]), "M")
    assert months[0] + 1 == months[1] == months[2]
    assert months[3] == months[2] + 1
    years = period_ids(np.array([dec_31, JAN_1_2024]), "Y")
    assert years[1] == years[0] + 1


def test_unknown_kind_raises():
    try:
        period_ids(np.array([JAN_1_2024]), "X")
    except ValueError:
        return
    raise AssertionError("ValueError bekleniyordu")


def _ten_days():
    time_s = JAN_1_2024 + np.arange(10) * DAY
    open_ = np.arange(10.0) + 10.0
    return time_s, open_, open_ + 2.0, open_ - 1.0, open_ + 1.0


def test_week_levels_hand_calculated():
    time_s, open_, high, low, close = _ten_days()
    lv = period_levels(open_, high, low, close, period_ids(time_s, "W"))
    assert lv.open[6] == 10.0 and lv.high[6] == 18.0 and lv.low[6] == 9.0
    assert np.isnan(lv.prev_high[:7]).all()
    assert lv.open[7] == 17.0
    assert lv.prev_high[7] == 18.0
    assert lv.prev_low[7] == 9.0
    assert lv.prev_close[7] == 17.0
    assert lv.high[9] == 21.0
    assert lv.low[9] == 16.0
    assert lv.prev_high[9] == 18.0


def test_data_starting_mid_period_initialises_first_bar():
    time_s, open_, high, low, close = _ten_days()
    lv = period_levels(open_[3:], high[3:], low[3:], close[3:], period_ids(time_s[3:], "W"))
    assert lv.open[0] == 13.0
    assert np.isnan(lv.prev_high[0])


def test_month_levels_carry_previous_month():
    time_s = JAN_1_2024 + np.arange(40) * DAY
    base = np.arange(40.0) + 100.0
    lv = period_levels(base, base + 1.0, base - 1.0, base + 0.5, period_ids(time_s, "M"))
    assert lv.open[30] == 100.0
    assert lv.open[31] == 131.0
    assert lv.prev_high[31] == 131.0
    assert lv.prev_close[31] == 130.5


def test_daily_open_utc_day_change():
    t = JAN_1_2024 + np.arange(0, 3 * DAY, 900)
    open_ = np.arange(len(t), dtype=float)
    out = daily_open(t, open_)
    assert (out[:96] == 0.0).all()
    assert (out[96:192] == 96.0).all()
    assert (out[192:] == 192.0).all()


def test_daily_open_custom_hour():
    t = JAN_1_2024 + np.arange(0, 2 * DAY, 3600)
    open_ = np.arange(len(t), dtype=float)
    out = daily_open(t, open_, hour_utc=3)
    assert (out[:3] == 0.0).all()
    assert (out[3:27] == 3.0).all()
    assert (out[27:] == 27.0).all()
