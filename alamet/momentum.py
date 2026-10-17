"""
A3 — ② MOMENTUM yöntemi (hız): n barlık log getiri / (σ_1bar × √n).

Kaynak Pine: pine/A3_momentum.pine (DEPV2 f_sym 513-526).
σ = ta.stdev(1 barlık log getiri, moZLen) (popülasyon std). σ hazır değilken veya 0 iken z = 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alamet.primitives import pine_stdev
from alamet.threshold import threshold_state, threshold_strength

MO_LEN = 12
MO_Z_LEN = 48
MO_THR = 1.0


@dataclass(frozen=True)
class MomentumResult:
    z: np.ndarray
    state: np.ndarray
    strength: np.ndarray


def momentum_signal(
    close: np.ndarray,
    length: int = MO_LEN,
    z_len: int = MO_Z_LEN,
    thr: float = MO_THR,
) -> MomentumResult:
    close = np.asarray(close, dtype=float)
    n = len(close)
    lr1 = np.full(n, np.nan)
    lr_n = np.full(n, np.nan)
    if n > 1:
        lr1[1:] = np.log(close[1:] / close[:-1])
    if n > length:
        lr_n[length:] = np.log(close[length:] / close[:-length])
    lr_sd = pine_stdev(lr1, z_len)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(lr_sd > 0, lr_n / (lr_sd * np.sqrt(length)), 0.0)
    return MomentumResult(z=z, state=threshold_state(z, thr), strength=threshold_strength(z, thr))
