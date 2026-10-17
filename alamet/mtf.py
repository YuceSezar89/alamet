"""
A7 — MTF güç birleşimi + TF onayı.

Kaynak Pine: pine/A7_mtf_confirm.pine (DEPV2 f_sym 485-495).
finalPct = ağırlıklı ortalama(4 HA skoru). TF onayı: LONG için skoru >0 olan HA sayısı >= need,
SHORT için <0 olan sayı >= need. Onay kapalıysa (tf_on=False) iki yön de her zaman onaylıdır.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

WEIGHTS = (2.0, 3.0, 4.0, 6.0)
TF_NEED = 3


@dataclass(frozen=True)
class MtfResult:
    final_pct: np.ndarray
    n_bull: np.ndarray
    n_bear: np.ndarray
    ok_long: np.ndarray
    ok_short: np.ndarray


def mtf_confirm(
    scores: Sequence[np.ndarray],
    weights: Sequence[float] = WEIGHTS,
    need: int = TF_NEED,
    tf_on: bool = True,
) -> MtfResult:
    arrs = [np.asarray(s, dtype=float) for s in scores]
    weighted = arrs[0] * weights[0]
    total = weights[0]
    for arr, w in zip(arrs[1:], weights[1:]):
        weighted = weighted + arr * w
        total = total + w
    n_bull = sum((a > 0).astype(int) for a in arrs)
    n_bear = sum((a < 0).astype(int) for a in arrs)
    if tf_on:
        ok_long = n_bull >= need
        ok_short = n_bear >= need
    else:
        ok_long = np.ones(len(arrs[0]), dtype=bool)
        ok_short = np.ones(len(arrs[0]), dtype=bool)
    return MtfResult(
        final_pct=weighted / total,
        n_bull=n_bull,
        n_bear=n_bear,
        ok_long=ok_long,
        ok_short=ok_short,
    )
