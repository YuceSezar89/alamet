"""
A4 — ③ VOLATİLİTE yöntemi (gürültü): TR / önceki yavaş ATR genişlemesi.
Yön, genişleme barının gövde yönüdür.

Kaynak Pine: pine/A4_volatility.pine (DEPV2 f_sym 528-544).
Durum makinesi TREND'den farklıdır: yalnızca genişleme (yukarı/aşağı gövde) yön verir,
sıfırlama yalnızca hızlı ATR / yavaş ATR < vaCalm iken olur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alamet.primitives import pine_atr, pine_tr

VA_SLOW = 12
VA_FAST = 5
VA_THR = 1.3
VA_CALM = 0.8


@dataclass(frozen=True)
class VolatilityResult:
    x: np.ndarray
    ratio: np.ndarray
    state: np.ndarray
    strength: np.ndarray


def volatility_state(
    up: np.ndarray, down: np.ndarray, ratio: np.ndarray, calm: float
) -> np.ndarray:
    state = np.zeros(len(up), dtype=int)
    st = 0
    for i, (u, d, r) in enumerate(zip(up, down, ratio)):
        if u:
            st = 1
        elif d:
            st = -1
        elif r < calm:
            st = 0
        state[i] = st
    return state


def volatility_strength(x: np.ndarray, thr: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.where(x >= thr * 1.5, 2, np.where(x >= thr * 1.2, 1, 0)).astype(int)


def volatility_signal(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    slow: int = VA_SLOW,
    fast: int = VA_FAST,
    thr: float = VA_THR,
    calm: float = VA_CALM,
) -> VolatilityResult:
    open_ = np.asarray(open_, dtype=float)
    close = np.asarray(close, dtype=float)
    tr = pine_tr(high, low, close)
    atr_slow = pine_atr(high, low, close, slow)
    atr_fast = pine_atr(high, low, close, fast)
    prev_slow = np.concatenate([[np.nan], atr_slow[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(np.nan_to_num(prev_slow) > 0, tr / prev_slow, 1.0)
        ratio = np.where(atr_slow > 0, atr_fast / atr_slow, 1.0)
    up = (x >= thr) & (close > open_)
    down = (x >= thr) & (close < open_)
    return VolatilityResult(
        x=x,
        ratio=ratio,
        state=volatility_state(up, down, ratio, calm),
        strength=volatility_strength(x, thr),
    )
