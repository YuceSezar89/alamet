"""
A2 — ① TREND yöntemi (seviye): (HLC3 - SMA) / ATR.

Kaynak Pine: pine/A2_trend.pine (DEPV2 f_sym 500-511).
Pine'ın na davranışı korunur: ATR hazır değilken z = 0.0, ATR hazır ama SMA hazır
değilken z = NaN (NaN karşılaştırmaları False olduğundan durum değişmez, güç 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alamet.primitives import pine_atr, pine_sma
from alamet.threshold import threshold_state, threshold_strength

ATR_LEN = 10
TD_LEN = 24
TD_THR = 0.5


@dataclass(frozen=True)
class TrendResult:
    z: np.ndarray
    state: np.ndarray
    strength: np.ndarray


def trend_signal(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_len: int = ATR_LEN,
    length: int = TD_LEN,
    thr: float = TD_THR,
) -> TrendResult:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    atr = pine_atr(high, low, close, atr_len)
    ma = pine_sma((high + low + close) / 3.0, length)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(atr > 0, (close - ma) / atr, 0.0)
    return TrendResult(z=z, state=threshold_state(z, thr), strength=threshold_strength(z, thr))
