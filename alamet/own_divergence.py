"""
C3 — Kendi içinde ayrışma (ÖNCE): bir emtia, kendi önceki hareketlerinden ayrışıyor mu?

Kaynak Pine: pine/C3_own_divergence.pine
(DEPV2 f_ownUpdate/f_ownGate/f_ownAll 1054-1125).

"Bacak": aynı yönlü sinyal dizisi. Yön lehine hareket rv = yön × (sinyalden beri % değişim).
Her barda önce durum güncellenir (bacak zirvesi, bacak bitince halka tampona yazılır),
sonra geçerli hareket önceki bacakların zirveleriyle karşılaştırılır:
  "Önceki hareket": cur > son bacak zirvesi × mult
  "Ort + σ"       : n >= min_prev ve cur > ortalama + σ_katsayısı × std
ownLatch açıkken cur = max(rv, bu bacağın zirvesi).
"""

from __future__ import annotations

import math

import numpy as np

MODE_PREV = "Önceki hareket"
MODE_SIGMA = "Ort + σ"
OWN_N = 8


class OwnDivergence:
    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        n_symbols: int,
        own_n: int = OWN_N,
        mult: float = 1.0,
        sigma: float = 1.0,
        min_prev: int = 2,
        latch: bool = True,
        allow_no_hist: bool = False,
        use_own_first: bool = True,
    ) -> None:
        self.n_symbols = n_symbols
        self.own_n = own_n
        self.mult = mult
        self.sigma = sigma
        self.min_prev = min_prev
        self.latch = latch
        self.allow_no_hist = allow_no_hist
        self.use_own_first = use_own_first
        self.ring = np.full((n_symbols, own_n), np.nan)
        self.count = np.zeros(n_symbols, dtype=int)
        self.leg_max = np.zeros(n_symbols)
        self.leg_dir = np.zeros(n_symbols, dtype=int)

    @staticmethod
    def _direction(value: float) -> int:
        if math.isnan(value):
            return 0
        return 1 if value == 1.0 else (-1 if value == -1.0 else 0)

    def update(self, types: np.ndarray, pcts: np.ndarray) -> None:
        for k in range(self.n_symbols):
            direction = self._direction(types[k])
            if direction == 0 or math.isnan(pcts[k]):
                continue
            move = direction * pcts[k]
            if self.leg_dir[k] != direction:
                if self.leg_dir[k] != 0:
                    self.ring[k, self.count[k] % self.own_n] = self.leg_max[k]
                    self.count[k] += 1
                self.leg_dir[k] = direction
                self.leg_max[k] = move
            else:
                self.leg_max[k] = max(self.leg_max[k], move)

    def _gate_one(self, k: int, direction: int, move: float, mode: str) -> tuple:
        seen = int(self.count[k])
        n = min(seen, self.own_n)
        cur = move
        if self.latch and self.leg_dir[k] == direction:
            cur = max(move, self.leg_max[k])
        ref = math.nan
        if n == 0:
            return self.allow_no_hist and cur > 0, ref, n, cur
        if mode == MODE_PREV:
            ref = self.ring[k, (seen - 1) % self.own_n] * self.mult
            return cur > ref and cur > 0, ref, n, cur
        total = square = 0.0
        for j in range(n):
            value = self.ring[k, j]
            total += value
            square += value * value
        mean = total / n
        std = math.sqrt(max(square / n - mean * mean, 0.0)) if n > 1 else 0.0
        ref = mean + self.sigma * std
        return n >= self.min_prev and cur > ref and cur > 0, ref, n, cur

    def gate(self, types: np.ndarray, pcts: np.ndarray, mode: str = MODE_PREV):
        ok = np.zeros(self.n_symbols, dtype=bool)
        ref = np.full(self.n_symbols, np.nan)
        count = np.zeros(self.n_symbols, dtype=int)
        cur = np.full(self.n_symbols, np.nan)
        for k in range(self.n_symbols):
            direction = self._direction(types[k])
            if direction == 0 or math.isnan(pcts[k]):
                continue
            passed, ref[k], count[k], cur[k] = self._gate_one(
                k, direction, direction * pcts[k], mode
            )
            ok[k] = (not self.use_own_first) or passed
        return ok, ref, count, cur


def own_gate_series(
    types: np.ndarray,
    pcts: np.ndarray,
    modes: tuple[str, ...] = (MODE_PREV, MODE_SIGMA),
    **params,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    types = np.asarray(types, dtype=float)
    pcts = np.asarray(pcts, dtype=float)
    bars, n_symbols = types.shape
    machine = OwnDivergence(n_symbols, **params)
    out = {
        m: [np.zeros((bars, n_symbols), dtype=t) for t in (bool, float, int, float)] for m in modes
    }
    for i in range(bars):
        machine.update(types[i], pcts[i])
        for mode in modes:
            for target, values in zip(out[mode], machine.gate(types[i], pcts[i], mode)):
                target[i] = values
    return {mode: tuple(arrays) for mode, arrays in out.items()}
