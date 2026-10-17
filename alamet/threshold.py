"""
Eşik tabanlı durum makinesi ve güç kodu — TREND (A2) ve MOMENTUM (A3) ortak kullanır.

Pine karşılığı:
  if z >= thr: st := 1 / else if z <= -thr: st := -1
  else if (st == 1 and z < 0) or (st == -1 and z > 0): st := 0
  güç = |z| >= thr*3 ? 2 : (|z| >= thr*2 ? 1 : 0)
NaN karşılaştırmaları False olduğundan NaN barda durum değişmez, güç 0 olur.
"""

from __future__ import annotations

import numpy as np


def threshold_state(z: np.ndarray, thr: float) -> np.ndarray:
    state = np.zeros(len(z), dtype=int)
    st = 0
    for i, zi in enumerate(z):
        if zi >= thr:
            st = 1
        elif zi <= -thr:
            st = -1
        elif (st == 1 and zi < 0) or (st == -1 and zi > 0):
            st = 0
        state[i] = st
    return state


def threshold_strength(z: np.ndarray, thr: float) -> np.ndarray:
    a = np.abs(z)
    with np.errstate(invalid="ignore"):
        return np.where(a >= thr * 3, 2, np.where(a >= thr * 2, 1, 0)).astype(int)
