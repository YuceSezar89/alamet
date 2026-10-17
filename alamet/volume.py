"""
A5 — ④ HACİM yöntemi (katılım): V dikey (bağıl hacim) / V yatay (fiyat bölgesindeki hacim payı).

Kaynak Pine: pine/A5_volume.pine (DEPV2 f_sym 546-572).
Pine'ın na/nz davranışı korunur:
  - hacim SMA'sı veya toplamı hazır değilken relV = 1.0, akış = 0.0
  - pencere başlangıçta eksikse eksik barlar nz(volume[i]) = 0 sayılır ve fiyat bölgesi koşulunda
    NaN karşılaştırması False olur (payda yalnızca mevcut barlar)
  - ATR hazır değilken hiçbir bar fiyat bölgesinde sayılmaz (pay 0, oran = relV / 0.1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from alamet.primitives import pine_atr, pine_sma, pine_sum

ATR_LEN = 10
HV_LEN = 12
HV_BAND = 0.5
HV_THR = 2.0
HV_FLOW = 0.1
MIN_SHARE = 0.1


@dataclass(frozen=True)
class VolumeResult:
    rel_v: np.ndarray
    flow: np.ndarray
    share: np.ndarray
    ratio: np.ndarray
    state: np.ndarray
    strength: np.ndarray


def volume_state(up: np.ndarray, down: np.ndarray, flow: np.ndarray) -> np.ndarray:
    state = np.zeros(len(up), dtype=int)
    st = 0
    for i, (u, d, f) in enumerate(zip(up, down, flow)):
        if u:
            st = 1
        elif d:
            st = -1
        elif (st == 1 and f < 0) or (st == -1 and f > 0):
            st = 0
        state[i] = st
    return state


def volume_strength(ratio: np.ndarray, thr: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.where(ratio >= thr * 1.5, 2, np.where(ratio >= thr * 1.2, 1, 0)).astype(int)


def price_zone_share(
    hlc3: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    atr: np.ndarray,
    length: int,
    band: float,
) -> np.ndarray:
    pad = length - 1
    hlc3_w = sliding_window_view(np.concatenate([np.full(pad, np.nan), hlc3]), length)
    vol_w = sliding_window_view(np.concatenate([np.zeros(pad), volume]), length)
    with np.errstate(invalid="ignore"):
        inside = np.abs(hlc3_w - close[:, None]) <= band * atr[:, None]
    sum_all = vol_w.sum(axis=1)
    sum_in = (vol_w * inside).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sum_all > 0, sum_in / sum_all, 0.0)


def volume_signal(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    atr_len: int = ATR_LEN,
    length: int = HV_LEN,
    band: float = HV_BAND,
    thr: float = HV_THR,
    flow_thr: float = HV_FLOW,
) -> VolumeResult:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    atr = pine_atr(high, low, close, atr_len)
    rng = high - low
    vol_sma = pine_sma(volume, length)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_v = np.where(vol_sma > 0, volume / vol_sma, 1.0)
        delta = np.where(rng > 0, volume * ((close - low) - (high - close)) / rng, 0.0)
        sum_v = pine_sum(volume, length)
        flow = np.where(sum_v > 0, pine_sum(delta, length) / sum_v, 0.0)
    share = price_zone_share((high + low + close) / 3.0, close, volume, atr, length, band)
    ratio = rel_v / np.maximum(share, MIN_SHARE)
    up = (ratio >= thr) & (flow >= flow_thr)
    down = (ratio >= thr) & (flow <= -flow_thr)
    return VolumeResult(
        rel_v=rel_v,
        flow=flow,
        share=share,
        ratio=ratio,
        state=volume_state(up, down, flow),
        strength=volume_strength(ratio, thr),
    )
