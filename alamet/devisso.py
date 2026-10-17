"""
B2 — Devisso ratio, DEPV2 günlük sürümü.

Kaynak Pine: pine/B2_devisso.pine (DEPV2 f_dayPack 643-665).
totalAmount ve RSI (Pine `ta.rsi`) bu dosyadadır. Ratio kısmı
"label" sürümünden FARKLIDIR:
  - label sürümü: işaretli oran, |Δtoplam| <= 0.0001 ise 0.0
  - DEPV2 sürümü: mutlak değer, eşik yok; Δtoplam == 0 veya oran NaN iken bir önceki değer taşınır
drB = bir önceki KAPANMIŞ günün ratio'su; drPr = drB'nin önceki `look` gündeki yüzdeliği
(0 = en küçük = en kolay), en az `min_n` geçerli veri yoksa NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RSI_LEN = 3
LOOK = 60
MIN_N = 10


def total_amount_series(closes: np.ndarray, prev_close: float | None) -> np.ndarray:
    """Kapanışlardan birikimli totalAmount serisi. prev_close, ilk mumun değişimini
    hesaplamak için bir önceki kapanış (yoksa None — ilk mumun değişimi 0 sayılır)."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if n == 0:
        return np.zeros(0)
    before = np.empty(n)
    before[0] = np.nan if prev_close is None else float(prev_close)
    before[1:] = closes[:-1]

    delta = np.zeros(n)
    valid = ~np.isnan(before) & (before != 0.0)
    up = valid & (closes > before)
    down = valid & (closes < before)
    delta[up] = (closes[up] - before[up]) / before[up] * 100.0
    delta[down] = -(before[down] - closes[down]) / closes[down] * 100.0
    return np.cumsum(delta)


def pine_rsi(src: np.ndarray, length: int = RSI_LEN) -> np.ndarray:
    """Pine ta.rsi: RMA ile, ilk RMA değeri ilk `length` değişimin SMA'sı.
    İlk `length` bar NaN, serinin ilk değişimi bar 1'de oluştuğu için ilk RSI
    index `length`'te çıkar."""
    src = np.asarray(src, dtype=float)
    n = len(src)
    rsi = np.full(n, np.nan)
    if n <= length:
        return rsi
    change = np.diff(src)
    gain = np.maximum(change, 0.0)
    loss = -np.minimum(change, 0.0)

    up = gain[:length].mean()
    down = loss[:length].mean()
    rsi[length] = _rsi_value(up, down)
    for i in range(length, len(change)):
        up = (gain[i] + (length - 1) * up) / length
        down = (loss[i] + (length - 1) * down) / length
        rsi[i + 1] = _rsi_value(up, down)
    return rsi


def _rsi_value(up: float, down: float) -> float:
    if down == 0.0:
        return 100.0
    if up == 0.0:
        return 0.0
    return 100.0 - 100.0 / (1.0 + up / down)


@dataclass(frozen=True)
class DevissoDaily:
    total: np.ndarray
    rsi: np.ndarray
    ratio: np.ndarray
    prev_ratio: np.ndarray
    percentile: np.ndarray


def ratio_series(total: np.ndarray, rsi_len: int = RSI_LEN) -> tuple[np.ndarray, np.ndarray]:
    total = np.asarray(total, dtype=float)
    n = len(total)
    rsi = pine_rsi(total, rsi_len)
    d_tot = np.full(n, np.nan)
    d_rsi = np.full(n, np.nan)
    if n > 1:
        d_tot[1:] = np.diff(total)
        d_rsi[1:] = np.diff(rsi)
    ratio = np.full(n, np.nan)
    current = np.nan
    for i in range(n):
        if not np.isnan(d_rsi[i]) and not np.isnan(d_tot[i]) and d_tot[i] != 0.0:
            current = abs(d_rsi[i] / d_tot[i])
        ratio[i] = current
    return rsi, ratio


def ratio_percentile(ratio: np.ndarray, look: int = LOOK, min_n: int = MIN_N) -> np.ndarray:
    ratio = np.asarray(ratio, dtype=float)
    n = len(ratio)
    out = np.full(n, np.nan)
    for t in range(1, n):
        base = ratio[t - 1]
        if np.isnan(base):
            continue
        window = ratio[max(0, t - look - 1) : max(0, t - 1)]
        window = window[~np.isnan(window)]
        if len(window) >= min_n:
            below = (window < base).sum() + 0.5 * (window == base).sum()
            out[t] = below / len(window) * 100.0
    return out


def devisso_daily(
    closes: np.ndarray, rsi_len: int = RSI_LEN, look: int = LOOK, min_n: int = MIN_N
) -> DevissoDaily:
    total = total_amount_series(closes, None)
    rsi, ratio = ratio_series(total, rsi_len)
    prev_ratio = np.concatenate([[np.nan], ratio[:-1]]) if len(ratio) else ratio
    return DevissoDaily(
        total=total,
        rsi=rsi,
        ratio=ratio,
        prev_ratio=prev_ratio,
        percentile=ratio_percentile(ratio, look, min_n),
    )
