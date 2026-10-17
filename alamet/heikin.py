"""
A6 — MTF Heikin Ashi gücü (tek zaman dilimi): (haC - haO) / (haH - haL) * 100.

Kaynak Pine: pine/A6_ha_score.pine (DEPV2 f_haScore 376-401).
HTF mumu sinyal TF barlarından birleştirilir. HTF bar sınırı timeframe.change(tf) ile
belirlenir; burada bu, `bucket` dizisindeki değişimdir (zaman damgası // HTF saniye).
Gün uzunluğu 30/45/60/90 dk'ya tam bölündüğü için 00:00 UTC hizalı borsa seanslarında
TradingView hizalamasıyla aynıdır. Veri başlangıcı bir HTF barının ortasına denk gelirse
ilk bar yeni bar gibi başlatılır (Pine `na(cO)`).
"""

from __future__ import annotations

import numpy as np


def bucket_ids(time_s: np.ndarray, tf_seconds: int) -> np.ndarray:
    return np.asarray(time_s, dtype=np.int64) // int(tf_seconds)


def ha_score(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    bucket: np.ndarray,
) -> np.ndarray:
    n = len(close)
    out = np.zeros(n)
    c_o = c_h = c_l = p_o = p_c = np.nan
    for i in range(n):
        if (i > 0 and bucket[i] != bucket[i - 1]) or np.isnan(c_o):
            if not np.isnan(c_o):
                last_c = (c_o + c_h + c_l + close[i - 1]) / 4.0
                last_o = (c_o + close[i - 1]) / 2.0 if np.isnan(p_o) else (p_o + p_c) / 2.0
                p_o, p_c = last_o, last_c
            c_o, c_h, c_l = open_[i], high[i], low[i]
        else:
            c_h = max(c_h, high[i])
            c_l = min(c_l, low[i])
        ha_c = (c_o + c_h + c_l + close[i]) / 4.0
        ha_o = (c_o + close[i]) / 2.0 if np.isnan(p_o) else (p_o + p_c) / 2.0
        rg = max(c_h, ha_o, ha_c) - min(c_l, ha_o, ha_c)
        out[i] = (ha_c - ha_o) / rg * 100.0 if rg != 0 else 0.0
    return out
