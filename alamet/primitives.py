"""
A1 — Pine ta.tr / ta.rma / ta.atr / ta.sma birebir portu.

Kaynak Pine: pine/A1_atr.pine
  ta.tr(true) : ilk bar high - low, sonrası max(h-l, |h-c1|, |l-c1|)
  ta.rma      : ilk değer ilk `length` verinin SMA'sı, sonra (x + (length-1)*önceki) / length
  ta.atr      : rma(tr, length)
  ta.sma      : yeterli veri yoksa NaN
  ta.stdev    : popülasyon std (biased); penceredeki NaN sonucu NaN yapar
  math.sum    : pencere toplamı; yeterli veri yoksa NaN
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

EPS = 1e-10


def gt(a: float, b: float) -> bool:
    """Pine kayan nokta `a > b`: TradingView farkı 1e-10'dan küçük olan sayıları eşit sayar."""
    return a - b > EPS


def lt(a: float, b: float) -> bool:
    return b - a > EPS


def ge(a: float, b: float) -> bool:
    return not lt(a, b)


def le(a: float, b: float) -> bool:
    return not gt(a, b)


def pine_tr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    tr = high - low
    if len(tr) > 1:
        prev_close = close[:-1]
        tr[1:] = np.maximum.reduce(
            [high[1:] - low[1:], np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)]
        )
    return tr


def pine_rma(src: np.ndarray, length: int) -> np.ndarray:
    src = np.asarray(src, dtype=float)
    n = len(src)
    out = np.full(n, np.nan)
    if length < 1 or n < length:
        return out
    value = src[:length].mean()
    out[length - 1] = value
    for i in range(length, n):
        value = (src[i] + (length - 1) * value) / length
        out[i] = value
    return out


def pine_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    return pine_rma(pine_tr(high, low, close), length)


def pine_sma(src: np.ndarray, length: int) -> np.ndarray:
    src = np.asarray(src, dtype=float)
    n = len(src)
    out = np.full(n, np.nan)
    if length < 1 or n < length:
        return out
    out[length - 1 :] = sliding_window_view(src, length).mean(axis=1)
    return out


def pine_sum(src: np.ndarray, length: int) -> np.ndarray:
    src = np.asarray(src, dtype=float)
    n = len(src)
    out = np.full(n, np.nan)
    if length < 1 or n < length:
        return out
    out[length - 1 :] = sliding_window_view(src, length).sum(axis=1)
    return out


def pine_stdev(src: np.ndarray, length: int) -> np.ndarray:
    src = np.asarray(src, dtype=float)
    n = len(src)
    out = np.full(n, np.nan)
    if length < 1 or n < length:
        return out
    out[length - 1 :] = sliding_window_view(src, length).std(axis=1)
    return out
