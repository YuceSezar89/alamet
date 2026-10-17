"""D1 — giriş kalitesi filtreleri: elle hesaplanmış durumlar."""

import numpy as np

from alamet.entry_quality import (
    COOLDOWN,
    DAILY_OPEN,
    DAILY_OPEN_FAR,
    EXTENDED,
    MONTHLY_OPEN,
    POWER,
    WEEKLY_OPEN,
    EntryQualityParams,
    describe,
    entry_flags,
)

NAN = np.nan
FAR = 1000


def _flags(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    side=1, start=100.0, close=102.0, atr=1.0, power=50.0, d=101.0, w=99.0, m=98.0, since=FAR, **kw
):
    params = EntryQualityParams(**kw)
    return int(entry_flags(side, start, close, atr, power, d, w, m, since, params))


def test_clean_long_entry_has_no_flags():
    assert _flags() == 0


def test_extension_blocks_both_overstretched_and_wrong_side():
    assert _flags(close=105.0) & EXTENDED
    assert _flags(close=99.0) & EXTENDED
    assert not _flags(close=104.0) & EXTENDED
    assert _flags(side=-1, start=100.0, close=98.0, d=99.0) & EXTENDED == 0
    assert _flags(side=-1, start=100.0, close=101.0, d=102.0) & EXTENDED


def test_extension_ignored_without_valid_inputs_or_when_disabled():
    assert not _flags(start=NAN) & EXTENDED
    assert not _flags(atr=0.0) & EXTENDED
    assert not _flags(close=105.0, use_ext=False) & EXTENDED


def test_cooldown_counts_bars_since_loss():
    assert _flags(since=11) & COOLDOWN
    assert not _flags(since=12) & COOLDOWN
    assert not _flags(since=5, cool_bars=0) & COOLDOWN


def test_daily_open_direction_for_long_and_short():
    assert _flags(close=102.0, d=102.0) & DAILY_OPEN
    assert _flags(close=102.0, d=103.0) & DAILY_OPEN
    assert not _flags(close=102.0, d=101.0) & DAILY_OPEN
    assert _flags(side=-1, close=98.0, start=100.0, d=98.0) & DAILY_OPEN
    assert not _flags(use_daily_open=False, d=110.0) & DAILY_OPEN
    assert not _flags(d=NAN) & DAILY_OPEN


def test_daily_open_distance_filter():
    assert _flags(max_daily_open_dist=0.5, d=101.0) & DAILY_OPEN_FAR
    assert not _flags(max_daily_open_dist=2.0, d=101.0) & DAILY_OPEN_FAR
    assert not _flags(max_daily_open_dist=0.0, d=50.0) & DAILY_OPEN_FAR


def test_weekly_and_monthly_open_only_when_enabled():
    assert not _flags(w=110.0, m=110.0) & (WEEKLY_OPEN | MONTHLY_OPEN)
    on = _flags(w=110.0, m=110.0, use_weekly_open=True, use_monthly_open=True)
    assert on & WEEKLY_OPEN and on & MONTHLY_OPEN
    assert not _flags(w=NAN, use_weekly_open=True) & WEEKLY_OPEN


def test_min_power_by_direction_and_nan():
    assert _flags(power=10.0, min_power=20.0) & POWER
    assert not _flags(power=30.0, min_power=20.0) & POWER
    assert _flags(side=-1, start=100.0, close=98.0, d=99.0, power=-10.0, min_power=20.0) & POWER
    assert not _flags(power=NAN, min_power=20.0) & POWER
    assert not _flags(power=-90.0) & POWER


def test_flags_combine_and_describe():
    flags = _flags(close=110.0, since=1, power=0.0, min_power=10.0)
    assert flags == POWER | EXTENDED | COOLDOWN
    assert describe(flags) == ["MTF güç zayıf", "uzama", "soğuma"]


def test_vectorised_matches_scalar():
    close = np.array([102.0, 105.0, 99.0])
    out = entry_flags(1, 100.0, close, 1.0, 50.0, 101.0, 99.0, 98.0, FAR)
    assert out.tolist() == [0, EXTENDED, EXTENDED | DAILY_OPEN]
