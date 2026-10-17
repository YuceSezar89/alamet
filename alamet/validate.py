"""
Girdi doğrulamaları (DEPV2 `barstate.isfirst` bloğu, 302-317): geçersiz ayar birleşimleri
Pine'da `runtime.error` verir; burada `ValueError` olarak fırlatılır.

Yöntemlerin en az birinin aktif olması (`SelectorParams`) ve sıralı conf sırası
(`parse_seq_order`) kendi modüllerinde zaten doğrulanır. `enforceTf` (grafik periyodu)
`validate_timeframe` ile yapılır.
"""

from __future__ import annotations

import warnings

from alamet.engine import TradeParams
from alamet.own_divergence import MODE_SIGMA
from alamet.panel import PanelParams
from alamet.timeframe import SignalTf, check_daily_open, enforce_chart_tf
from alamet.universe import Universe


def validate_params(panel: PanelParams, trade: TradeParams) -> None:
    div = panel.divergence
    if div.min_prev > div.max_div - 1:
        raise ValueError("minPrev, maxDiv - 1 değerinden büyük olamaz")
    if panel.own_mode == MODE_SIGMA and panel.own.get("min_prev", div.min_prev) > panel.own.get(
        "own_n", 8
    ):
        raise ValueError("Ort + σ modunda minPrev, ownN değerinden büyük olamaz")
    if trade.dr_max_m < trade.dr_min_m:
        raise ValueError("Devisso: en kolay çarpan, en zor çarpandan küçük olamaz")


def validate_timeframe(panel: PanelParams, universe: Universe, enforce: bool = True) -> SignalTf:
    """Veri aralığı sinyal TF'sine eşit olmalı. Daha kalın veri Pine'daki `enforceTf` hatasıdır;
    daha ince veri önce `resample_universe` ile sinyal TF'sine çevrilmelidir (botta `ssLag` yolu
    yoktur, ince grafikle çalışma TradingView'a özgü bir okuma gecikmesidir)."""
    tf = SignalTf(panel.symbol.signal_tf_seconds)
    seconds = universe.bar_seconds
    if seconds is not None:
        enforce_chart_tf(seconds, tf.seconds, enforce)
        if seconds < tf.seconds:
            raise ValueError(
                f"Veri aralığı ({seconds} sn) sinyal TF'sinden ({tf.seconds} sn) ince: "
                "önce resample_universe ile sinyal TF'sine çevir"
            )
    if not check_daily_open(panel.daily_open_hour, tf.seconds):
        warnings.warn(
            "Günlük open saati ≠ 0 yalnızca sinyal TF ≤ 60 dk iken doğru çalışır", stacklevel=2
        )
    return tf
