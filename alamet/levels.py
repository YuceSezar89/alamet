"""
B1 — Haftalık / aylık / yıllık OHLC seviyeleri ve günlük open.

Kaynak Pine: pine/B1_levels.pine (DEPV2 f_per 617-637) ve B1b_daily_open.pine
(DEPV2 f_sym 574-580).

Seviye başına 6 değer: periyodun şimdiye kadarki open/high/low'u + önceki periyodun
high/low/close'u. Önceki periyot değerleri ilk periyot değişimine kadar NaN'dır. Veri
başlangıcı bir periyodun ortasına denk gelirse ilk bar yeni periyot gibi başlatılır
(Pine `na(cO)`).

Periyot sınırları (UTC): hafta Pazartesi başlar, ay takvim ayı, yıl takvim yılı
(Pine "W", "M", "12M").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class PeriodLevels:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    prev_high: np.ndarray
    prev_low: np.ndarray
    prev_close: np.ndarray


def period_ids(time_s: np.ndarray, kind: str) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=np.int64)
    if kind == "W":
        return (time_s // SECONDS_PER_DAY + 3) // 7
    stamps = time_s.astype("datetime64[s]")
    if kind == "M":
        return stamps.astype("datetime64[M]").astype(np.int64)
    if kind == "Y":
        return stamps.astype("datetime64[Y]").astype(np.int64)
    raise ValueError(f"bilinmeyen periyot: {kind}")


def period_levels(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ids: np.ndarray,
) -> PeriodLevels:
    n = len(close)
    out = np.full((n, 6), np.nan)
    c_o = c_h = c_l = p_h = p_l = p_c = np.nan
    for i in range(n):
        if (i > 0 and ids[i] != ids[i - 1]) or np.isnan(c_o):
            if not np.isnan(c_o):
                p_h, p_l, p_c = c_h, c_l, close[i - 1]
            c_o, c_h, c_l = open_[i], high[i], low[i]
        else:
            c_h = max(c_h, high[i])
            c_l = min(c_l, low[i])
        out[i] = (c_o, c_h, c_l, p_h, p_l, p_c)
    return PeriodLevels(*(out[:, k] for k in range(6)))


def daily_open(time_s: np.ndarray, open_: np.ndarray, hour_utc: int = 0) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=np.int64)
    open_ = np.asarray(open_, dtype=float)
    n = len(open_)
    out = np.full(n, np.nan)
    hours = (time_s % SECONDS_PER_DAY) // 3600
    days = time_s // SECONDS_PER_DAY
    current = np.nan
    for i in range(n):
        if hour_utc == 0:
            new_day = i > 0 and days[i] != days[i - 1]
        else:
            new_day = i > 0 and hours[i] == hour_utc and hours[i - 1] != hour_utc
        if new_day or np.isnan(current):
            current = open_[i]
        out[i] = current
    return out
