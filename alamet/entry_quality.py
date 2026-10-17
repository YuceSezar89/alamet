"""
D1 — Giriş kalitesi filtreleri (DEPV2 f_qual).

Kaynak Pine: pine/D1_entry_quality.pine (DEPV2 1588-1617). Filtrelerden biri bile
tetiklenirse o sembole o barda girilmez. Sonuç bit maskesidir (0 = giriş serbest).
Bitler: 1 MTF güç · 2 uzama · 4 soğuma · 8 günlük open ters · 16 open'dan uzak ·
32 haftalık open ters · 64 aylık open ters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POWER = 1
EXTENDED = 2
COOLDOWN = 4
DAILY_OPEN = 8
DAILY_OPEN_FAR = 16
WEEKLY_OPEN = 32
MONTHLY_OPEN = 64

LABELS = {
    POWER: "MTF güç zayıf",
    EXTENDED: "uzama",
    COOLDOWN: "soğuma",
    DAILY_OPEN: "günlük open ters",
    DAILY_OPEN_FAR: "open'dan uzak",
    WEEKLY_OPEN: "haftalık open ters",
    MONTHLY_OPEN: "aylık open ters",
}


@dataclass(frozen=True)
class EntryQualityParams:  # pylint: disable=too-many-instance-attributes
    min_power: float = 0.0
    use_ext: bool = True
    max_ext: float = 4.0
    cool_bars: int = 12
    use_daily_open: bool = True
    use_weekly_open: bool = False
    use_monthly_open: bool = False
    max_daily_open_dist: float = 0.0


def _against(side: np.ndarray, close: np.ndarray, level: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.where(side == 1, close <= level, close >= level)


def entry_flags(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    side,
    start,
    close,
    atr,
    power,
    daily_open,
    weekly_open,
    monthly_open,
    bars_since_loss,
    params: EntryQualityParams = EntryQualityParams(),
) -> np.ndarray:
    """Girdiler skaler ya da dizi olabilir; `bars_since_loss = ss_idx - son_zarar`."""
    side, start, close, atr, power = (
        np.asarray(x, dtype=float) for x in (side, start, close, atr, power)
    )
    d_open, w_open, m_open = (
        np.asarray(x, dtype=float) for x in (daily_open, weekly_open, monthly_open)
    )
    since = np.asarray(bars_since_loss, dtype=float)
    flags = np.zeros(np.broadcast(side, close).shape, dtype=int)
    with np.errstate(invalid="ignore", divide="ignore"):
        if params.min_power > 0:
            weak = np.where(side == 1, power < params.min_power, power > -params.min_power)
            flags |= np.where(weak & ~np.isnan(power), POWER, 0)
        if params.use_ext:
            ok = ~(np.isnan(start) | np.isnan(close) | np.isnan(atr)) & (atr > 0)
            ext = np.where(side == 1, close - start, start - close) / atr
            flags |= np.where(ok & ((ext <= 0) | (ext > params.max_ext)), EXTENDED, 0)
        flags |= np.where(since < params.cool_bars, COOLDOWN, 0)
        has_open = ~(np.isnan(d_open) | np.isnan(close))
        if params.use_daily_open:
            flags |= np.where(has_open & _against(side, close, d_open), DAILY_OPEN, 0)
        if params.max_daily_open_dist > 0:
            far = (
                (~np.isnan(atr))
                & (atr > 0)
                & (np.abs(close - d_open) / atr > params.max_daily_open_dist)
            )
            flags |= np.where(has_open & far, DAILY_OPEN_FAR, 0)
        if params.use_weekly_open:
            valid = ~(np.isnan(w_open) | np.isnan(close))
            flags |= np.where(valid & _against(side, close, w_open), WEEKLY_OPEN, 0)
        if params.use_monthly_open:
            valid = ~(np.isnan(m_open) | np.isnan(close))
            flags |= np.where(valid & _against(side, close, m_open), MONTHLY_OPEN, 0)
    return flags


def describe(flags: int) -> list[str]:
    return [label for bit, label in LABELS.items() if int(flags) & bit]
