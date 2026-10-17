"""
E2 — Kazanan havuzu: emtia başına kapanan işlem sayısı ve sönümlü net K/Z ile sıralamaya girebilen
emtiaları sınırlar; 1. sıradaki emtiayı net K/Z çok kötü değilse havuzda tutar.

Kaynak Pine: DEPV2 f_poolCalc (1032-1049), f_top1 (1157-1182), f_close (1759-1760), sönümleme
(f_blockB 1889-1891). Havuz maskesi her barın başında, bir önceki barın motor adımından sonraki
duruma göre hesaplanır ve yalnızca sıralamaya (f_rankAll) girer. f_mRank ve idxNet orijinalde
kullanılmıyor, taşınmadı.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alamet.primitives import gt
from alamet.ranking import SENTINEL, sort_descending


@dataclass(frozen=True)
class PoolParams:  # pylint: disable=too-many-instance-attributes
    on: bool = False
    size: int = 8
    min_n: float = 4.0
    min_net: float = 0.0
    prob: bool = True
    decay_bars: int = 48
    decay: float = 0.85
    forgive: float = 0.2
    top1: bool = True
    top1_floor: float = -2.0


class PoolState:
    def __init__(self, n_symbols: int, params: PoolParams, start_cap: float) -> None:
        self.p = params
        self.start_cap = start_cap
        self.count = np.zeros(n_symbols)
        self.net = np.zeros(n_symbols)

    def add(self, k: int, usd: float) -> None:
        self.count[k] += 1.0
        self.net[k] += usd

    def decay_step(self, step_i: int) -> None:
        if self.p.on and step_i % self.p.decay_bars == 0:
            self.net *= self.p.decay

    def mask(self) -> np.ndarray:
        """Havuzun izin verdiği emtialar (f_poolCalc)."""
        allowed = np.ones(len(self.net), dtype=bool)
        if not self.p.on:
            return allowed
        p = self.p
        judged_net = np.where(self.count >= p.min_n, self.net, SENTINEL)
        order = sort_descending(judged_net[None, :], ties="pine")[0]
        forgive = p.forgive / 100.0 * self.start_cap
        for position, k in enumerate(order):
            if self.count[k] >= p.min_n:
                net = self.net[k]
                good = gt(net, p.min_net) and position < p.size
                allowed[k] = good or (p.prob and gt(net, p.min_net - forgive))
            else:
                allowed[k] = p.prob
        return allowed

    def protect_top1(
        self, allowed: np.ndarray, scores: np.ndarray, types: np.ndarray, own_ok: np.ndarray
    ) -> np.ndarray:
        """Ham skoru en yüksek LONG ve SHORT emtia, net K/Z çok kötü değilse havuzda kalır."""
        if not (self.p.on and self.p.top1):
            return allowed
        out = allowed.copy()
        floor = self.p.top1_floor / 100.0 * self.start_cap
        for side in (1.0, -1.0):
            best, pick = SENTINEL, -1
            for k, (score, kind) in enumerate(zip(scores, types)):
                valid = not (np.isnan(score) or np.isnan(kind)) and own_ok[k]
                if valid and kind == side and score > best:
                    best, pick = score, k
            if pick >= 0 and (self.count[pick] < self.p.min_n or gt(self.net[pick], floor)):
                out[pick] = True
        return out
