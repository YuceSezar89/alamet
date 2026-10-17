"""
C2 — Yön bazlı sıralama ve ilk N ("market hedefi") seçimi. Havuz kapalı varsayılır.

Kaynak Pine: pine/C2_ranking.pine (DEPV2 f_rankAll 1187-1215).
Sıralamaya yalnızca skoru, yönü ve kendi-içi kapısı geçerli semboller girer (LONG ve
SHORT ayrı listelenir); girmeyenler -1e10 nöbetçi değeriyle sona gider. İlk N sembol,
(N+1). sıradaki skordan en az `gap_min` üstteyse ya da (N+1). sırada geçerli skor yoksa
topArr'a girer.

Eşit skorlarda sıra (Pine array.sort_indices, ölçüldü: P1 sondası + C2, 87.820 satır):
eşitler indeks sırasının TERSİNE dizilir ("last"); tek istisna, dizinin bütün elemanları
başlangıç dolgu değerine eşitse (dizi hiç değişmemişse) sıra olduğu gibi kalır. DEPV2'de
rL/rS -1e10 ile doldurulduğundan geçerli sembol yoksa sıra 0..N-1'dir. Mekanizma doğrulanmış
değil, kural gözleme dayanır; `ties` yalnızca "first"/"last"/"pine" karşılaştırması içindir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SENTINEL = -1e10
NEAR_SENTINEL = -1e9
N_TOP = 2


@dataclass(frozen=True)
class RankResult:
    r_long: np.ndarray
    r_short: np.ndarray
    idx_long: np.ndarray
    idx_short: np.ndarray
    cut_long: np.ndarray
    cut_short: np.ndarray
    top: np.ndarray


def sort_descending(values: np.ndarray, ties: str = "first", fill: float = SENTINEL) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = values.shape[1]
    first = np.argsort(-values, axis=1, kind="stable")
    last = n - 1 - np.argsort(-values[:, ::-1], axis=1, kind="stable")
    if ties == "last":
        return last
    if ties == "pine":
        untouched = (values == fill).all(axis=1)
        return np.where(untouched[:, None], np.arange(n)[None, :], last)
    return first


def _mark_top(top, idx, ranks, cut, n_top_c, gap_min) -> None:
    rows = np.arange(idx.shape[0])
    for r in range(n_top_c):
        k = idx[:, r]
        value = ranks[rows, k]
        ok = (value > NEAR_SENTINEL) & ((cut <= NEAR_SENTINEL) | (value - cut >= gap_min))
        top[rows[ok], k[ok]] = True


def rank_all(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    scores: np.ndarray,
    types: np.ndarray,
    own_ok: np.ndarray,
    n_top: int = N_TOP,
    gap_min: float = 0.0,
    pool_ok: np.ndarray | None = None,
    ties: str = "first",
) -> RankResult:
    scores = np.asarray(scores, dtype=float)
    types = np.asarray(types, dtype=float)
    allowed = np.asarray(own_ok, dtype=bool)
    if pool_ok is not None:
        allowed = allowed & np.asarray(pool_ok, dtype=bool)
    valid = ~np.isnan(scores) & ~np.isnan(types) & allowed
    r_long = np.where(valid & (types == 1.0), scores, SENTINEL)
    r_short = np.where(valid & (types == -1.0), scores, SENTINEL)
    idx_long = sort_descending(r_long, ties)
    idx_short = sort_descending(r_short, ties)
    n_top_c = min(n_top, scores.shape[1] - 1)
    rows = np.arange(scores.shape[0])
    cut_long = r_long[rows, idx_long[:, n_top_c]]
    cut_short = r_short[rows, idx_short[:, n_top_c]]
    top = np.zeros(scores.shape, dtype=bool)
    _mark_top(top, idx_long, r_long, cut_long, n_top_c, gap_min)
    _mark_top(top, idx_short, r_short, cut_short, n_top_c, gap_min)
    return RankResult(r_long, r_short, idx_long, idx_short, cut_long, cut_short, top)
