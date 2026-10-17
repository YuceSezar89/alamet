"""
C1 — Skor: temel skor, hedef mesafesi puanı ve Devisso kolaylık puanı.

Kaynak Pine: pine/C1a_base_score.pine, C1b_target_distance.pine, C1c_devisso_ease.pine
(DEPV2 f_selOut, f_calcSc, f_near, f_tgtD, f_tgtAll, f_drXsAll, f_drPr).

  temel skor   = yön × (kapanış - başlangıç) / başlangıç × 100, ATR modunda / ATR%
  hedef puanı  = tgtW × min(hedef mesafesi, tgtCap)
                 (mesafe: yönündeki en yakın HTF seviyesi, ATR cinsinden)
  kolaylık     = drRankW × (1 - birleşik Devisso yüzdeliği / 100)
Toplam skor: temel + hedef puanı (yalnızca kendi-içi ayrışma kapısını geçenlere) + kolaylık puanı.
"""

from __future__ import annotations

import numpy as np

RANK_ATR = "ATR cinsinden (volatilite normalize)"
RANK_RAW = "Ham %"
TGT_NEAR = "Yakın hedef"
TGT_BIG = "Büyük hedef (aylık+yıllık)"
TGT_MIX = "Yakın + büyük ortalaması"
DR_OWN = "Kendi geçmişi"
DR_XS = "Emtialar arası"
DR_MIX = "Karma"

TGT_W = 0.25
TGT_CAP = 8.0
DR_RANK_W = 1.0
ALL_LEVELS = (0, 17)
BIG_LEVELS = (6, 17)


def atr_percent(atr: np.ndarray, close: np.ndarray) -> np.ndarray:
    atr = np.asarray(atr, dtype=float)
    close = np.asarray(close, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            ~np.isnan(atr) & ~np.isnan(close) & (close > 0), atr / close * 100.0, np.nan
        )


def signal_pct(close: np.ndarray, start: np.ndarray) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    start = np.asarray(start, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = (close - start) / start * 100.0
    bad = np.isnan(start) | np.isnan(close) | (start == 0)
    return np.where(bad, 0.0, pct)


def base_score(
    types: np.ndarray, pcts: np.ndarray, atr_p: np.ndarray, rank_mode: str = RANK_ATR
) -> np.ndarray:
    types = np.asarray(types, dtype=float)
    pcts = np.asarray(pcts, dtype=float)
    atr_p = np.asarray(atr_p, dtype=float)
    valid = ~np.isnan(pcts) & ~np.isnan(types) & (types != 0)
    raw = types * pcts
    if rank_mode == RANK_RAW:
        return np.where(valid, raw, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(valid & ~np.isnan(atr_p) & (atr_p > 0), raw / atr_p, np.nan)


def near_level(levels: np.ndarray, px: np.ndarray, up, first: int, last: int) -> np.ndarray:
    cand = np.asarray(levels, dtype=float)[:, first : last + 1]
    px = np.asarray(px, dtype=float)[:, None]
    with np.errstate(invalid="ignore"):
        above = np.where(cand > px, cand, np.inf).min(axis=1)
        below = np.where(cand < px, cand, -np.inf).max(axis=1)
    above = np.where(np.isinf(above), np.nan, above)
    below = np.where(np.isinf(below), np.nan, below)
    return np.where(up, above, below)


def target_distances(
    levels: np.ndarray, px: np.ndarray, atr: np.ndarray, up
) -> tuple[np.ndarray, np.ndarray]:
    px = np.asarray(px, dtype=float)
    atr = np.asarray(atr, dtype=float)
    valid = ~np.isnan(px) & ~np.isnan(atr) & (atr > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        d_near = np.abs(near_level(levels, px, up, *ALL_LEVELS) - px) / atr
        d_big = np.abs(near_level(levels, px, up, *BIG_LEVELS) - px) / atr
    return np.where(valid, d_near, np.nan), np.where(valid, d_big, np.nan)


def target_distance(
    levels: np.ndarray, px: np.ndarray, atr: np.ndarray, up, mode: str = TGT_MIX
) -> np.ndarray:
    d_near, d_big = target_distances(levels, px, atr, up)
    px = np.asarray(px, dtype=float)
    atr = np.asarray(atr, dtype=float)
    valid = ~np.isnan(px) & ~np.isnan(atr) & (atr > 0)
    if mode == TGT_NEAR:
        res = d_near
    elif mode == TGT_BIG:
        res = d_big
    else:
        both = ~np.isnan(d_near) & ~np.isnan(d_big)
        res = np.where(both, (d_near + d_big) / 2.0, np.where(np.isnan(d_big), d_near, d_big))
    return np.where(valid, res, np.nan)


def target_bonus(
    dist: np.ndarray,
    atr_p: np.ndarray,
    weight: float = TGT_W,
    cap: float = TGT_CAP,
    rank_mode: str = RANK_ATR,
) -> np.ndarray:
    dist = np.asarray(dist, dtype=float)
    scale = np.nan_to_num(np.asarray(atr_p, dtype=float), nan=0.0) if rank_mode == RANK_RAW else 1.0
    return np.where(np.isnan(dist), np.nan, weight * np.minimum(dist, cap) * scale)


def cross_percentile(ratios: np.ndarray, min_symbols: int = 3) -> np.ndarray:
    ratios = np.asarray(ratios, dtype=float)
    valid = ~np.isnan(ratios)
    count = valid.sum(axis=1)
    with np.errstate(invalid="ignore"):
        less = ratios[:, None, :] < ratios[:, :, None]
        equal = ratios[:, None, :] == ratios[:, :, None]
    others = valid[:, None, :] & ~np.eye(ratios.shape[1], dtype=bool)[None, :, :]
    points = ((less * 1.0 + equal * 0.5) * others).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = points / (count[:, None] - 1) * 100.0
    return np.where(valid & (count[:, None] >= min_symbols), pct, np.nan)


def combine_percentile(own: np.ndarray, xs: np.ndarray, source: str = DR_MIX) -> np.ndarray:
    own = np.asarray(own, dtype=float)
    xs = np.asarray(xs, dtype=float)
    if source == DR_OWN:
        return np.where(np.isnan(own), xs, own)
    if source == DR_XS:
        return np.where(np.isnan(xs), own, xs)
    both = ~np.isnan(own) & ~np.isnan(xs)
    return np.where(both, (own + xs) / 2.0, np.where(np.isnan(own), xs, own))


def ease_bonus(
    percentile: np.ndarray,
    weight: float = DR_RANK_W,
    atr_p: np.ndarray | None = None,
    rank_mode: str = RANK_ATR,
) -> np.ndarray:
    pr = np.asarray(percentile, dtype=float)
    if rank_mode == RANK_RAW and atr_p is not None:
        scale = np.nan_to_num(np.asarray(atr_p, dtype=float), nan=1.0)
    else:
        scale = 1.0
    return np.where(np.isnan(pr), np.nan, weight * (1.0 - pr / 100.0) * scale)


def combine_score(
    base: np.ndarray,
    tgt_bonus: np.ndarray,
    dr_bonus: np.ndarray,
    own_ok: np.ndarray | None = None,
) -> np.ndarray:
    base = np.asarray(base, dtype=float)
    tgt = np.asarray(tgt_bonus, dtype=float)
    if own_ok is not None:
        tgt = np.where(own_ok, tgt, np.nan)
    total = base + np.nan_to_num(tgt, nan=0.0) + np.nan_to_num(np.asarray(dr_bonus), nan=0.0)
    return np.where(np.isnan(base), np.nan, total)
