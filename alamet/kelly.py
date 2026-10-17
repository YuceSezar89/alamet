"""
E1 — Kasa motoru (Kelly / beklenen-R): kapanan işlemlerden öğrenen büyüklük çarpanı ve BLOK.

Kaynak Pine: DEPV2 f_engIdx, f_engAcc, f_engSd, f_engine (1638-1690), sönümleme (1893-1897).
Her kapanan işlem 9 özellik kovasına (yön, yöntem, güç, hedef sırası, ayrışma no, giriş tipi,
günlük open yönü, Devisso kolaylık dilimi, emtia) kırpılmış R olarak yazılır. Yeni aday için
kovaların küçültülmüş ortalamaları güvenle ağırlıklandırılır; E[R] ve σ'dan çeyrek Kelly çarpanı ve
BLOK (E[R] ≤ −eşik) üretilir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alamet.primitives import ge, gt, le

DIRECTION_BASE = 9
METHOD_BASE = 11
STRENGTH_BASE = 15
RANK_FIRST, RANK_OTHER = 18, 19
DIV_BASE = 20
ENTRY_DIRECT, ENTRY_SECOND = 31, 32
OPEN_RIGHT, OPEN_WRONG = 33, 34
SYMBOL_BASE = 36
EASE_TIERS = 4
DEFAULT_SYMBOLS = 20


def ease_base(n_symbols: int) -> int:
    """Devisso dilimi kovalarının ilk indeksi: sembol kovalarından hemen sonra gelir."""
    return SYMBOL_BASE + n_symbols


def bucket_count(n_symbols: int) -> int:
    return ease_base(n_symbols) + EASE_TIERS


BUCKETS = bucket_count(DEFAULT_SYMBOLS)
EASE_BASE = ease_base(DEFAULT_SYMBOLS)
MIN_SD_VARIANCE = 0.25


@dataclass(frozen=True)
class KellyParams:  # pylint: disable=too-many-instance-attributes
    on: bool = False
    replace: bool = True
    kelly: float = 0.25
    prior: float = 4.0
    z: float = 0.5
    min_n: float = 6.0
    explore: float = 0.5
    min_mult: float = 0.25
    max_mult: float = 3.0
    block: float = 0.15
    clip: float = 3.0
    decay_bars: int = 96
    decay: float = 0.98
    dd_scale: bool = True


def ease_tier(percentile: float, base: int = EASE_BASE) -> int:
    if math.isnan(percentile):
        return base + 2
    if percentile <= 25:
        return base
    if percentile <= 50:
        return base + 1
    return base + 2 if percentile <= 75 else base + 3


def open_side(side: int, price: float, daily_open: float) -> int:
    if math.isnan(daily_open) or math.isnan(price):
        return 0
    return 1 if (price > daily_open if side == 1 else price < daily_open) else -1


def bucket_indices(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    symbol: int, side: int, rank: int, div_n: int, strength: int, method: int, op_side: int,
    tier: int,
) -> list[int]:  # fmt: skip
    if div_n <= 2:
        div_bucket = DIV_BASE
    elif div_n == 3:
        div_bucket = DIV_BASE + 1
    else:
        div_bucket = DIV_BASE + 2 if div_n <= 5 else DIV_BASE + 3
    return [
        DIRECTION_BASE + (0 if side == 1 else 1),
        METHOD_BASE + method,
        STRENGTH_BASE + strength % 3,
        RANK_FIRST if rank == 1 else RANK_OTHER,
        div_bucket,
        ENTRY_SECOND if strength >= 3 else ENTRY_DIRECT,
        OPEN_WRONG if op_side == -1 else OPEN_RIGHT,
        tier,
        SYMBOL_BASE + symbol,
    ]


class KellyEngine:
    def __init__(self, params: KellyParams, n_symbols: int = DEFAULT_SYMBOLS) -> None:
        self.p = params
        self.n = np.zeros(bucket_count(n_symbols))
        self.total = np.zeros(bucket_count(n_symbols))
        self.total_sq = np.zeros(bucket_count(n_symbols))

    def accumulate(self, indices: list[int], r: float) -> None:
        clipped = max(-self.p.clip, min(self.p.clip, r))
        for bucket in indices:
            self.n[bucket] += 1.0
            self.total[bucket] += clipped
            self.total_sq[bucket] += clipped * clipped

    def deviation(self) -> float:
        count = self.n[DIRECTION_BASE] + self.n[DIRECTION_BASE + 1]
        if gt(count, 0.0):
            mean = (self.total[DIRECTION_BASE] + self.total[DIRECTION_BASE + 1]) / count
            var = (self.total_sq[DIRECTION_BASE] + self.total_sq[DIRECTION_BASE + 1]) / count
            var -= mean * mean
        else:
            var = 1.0
        return math.sqrt(max(var, MIN_SD_VARIANCE))

    def estimate(self, indices: list[int]) -> tuple[float, float]:
        """Kovaların güvenle ağırlıklı ortalama R'si ve etkin örnek sayısı."""
        p = self.p
        weight_sum = weighted = seen = 0.0
        last = len(indices) - 1
        for j, bucket in enumerate(indices):
            count = self.n[bucket]
            weight = count / (count + p.prior)
            weight_sum += weight
            weighted += weight * self.total[bucket] / (count + p.prior)
            if j < last:
                seen += count
        mean = weighted / weight_sum if gt(weight_sum, 0.0) else 0.0
        return mean, seen / last

    def evaluate(
        self, indices: list[int], drawdown: float, dd_hard_pct: float, risk_pct: float
    ) -> tuple[float, bool]:
        """(çarpan, blok) döner; motor kapalıysa (1.0, False)."""
        p = self.p
        if not p.on:
            return 1.0, False
        mean, effective = self.estimate(indices)
        sd = self.deviation()
        block = False
        if ge(effective, p.min_n):
            safe = mean - p.z * sd / math.sqrt(max(effective, 1.0))
            kelly = p.kelly * (safe / (sd * sd)) / (risk_pct / 100.0) if gt(safe, 0.0) else 0.0
            mult = max(p.min_mult, min(p.max_mult, kelly))
            block = le(mean, -p.block)
        else:
            mult = p.explore
        if p.dd_scale and drawdown > 0:
            mult *= max(0.25, 1.0 - drawdown / dd_hard_pct)
        return mult, block

    def decay_step(self, step_i: int) -> None:
        if self.p.on and step_i % self.p.decay_bars == 0:
            self.n *= self.p.decay
            self.total *= self.p.decay
            self.total_sq *= self.p.decay
