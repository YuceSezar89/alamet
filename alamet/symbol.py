"""
Sembol başına sinyal hattı (DEPV2 f_sym): 4 dik yöntem + MTF Heikin Ashi onayı + onay makinesi.

Sıra: 0 TREND · 1 MOMENTUM · 2 VOLATİLİTE · 3 HACİM (DEPV2 yöntem indeksiyle aynı).
Kaynak Pine: pine/A8_gate.pine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alamet.gate import VP_CONF_BARS, VP_COOL, VP_PEND_MAX, GateResult, run_gate
from alamet.heikin import bucket_ids, ha_score
from alamet.momentum import MO_LEN, MO_THR, MO_Z_LEN, momentum_signal
from alamet.mtf import TF_NEED, WEIGHTS, MtfResult, mtf_confirm
from alamet.trend import ATR_LEN, TD_LEN, TD_THR, trend_signal
from alamet.volatility import VA_CALM, VA_FAST, VA_SLOW, VA_THR, volatility_signal
from alamet.volume import HV_BAND, HV_FLOW, HV_LEN, HV_THR, volume_signal

SIGNAL_TF_SECONDS = 900
HA_MULTIPLIERS = (2, 3, 4, 6)
METHOD_NAMES = ("TREND", "MOMENTUM", "VOLATİLİTE", "HACİM")


@dataclass(frozen=True)
class SymbolParams:  # pylint: disable=too-many-instance-attributes
    tf_on: bool = True
    tf_need: int = TF_NEED
    atr_len: int = ATR_LEN
    td_len: int = TD_LEN
    td_thr: float = TD_THR
    mo_len: int = MO_LEN
    mo_z_len: int = MO_Z_LEN
    mo_thr: float = MO_THR
    va_slow: int = VA_SLOW
    va_fast: int = VA_FAST
    va_thr: float = VA_THR
    va_calm: float = VA_CALM
    hv_len: int = HV_LEN
    hv_band: float = HV_BAND
    hv_thr: float = HV_THR
    hv_flow: float = HV_FLOW
    ha_multipliers: tuple = HA_MULTIPLIERS
    ha_weights: tuple = WEIGHTS
    cool: int = VP_COOL
    pend_max: int = VP_PEND_MAX
    conf_bars: int = VP_CONF_BARS
    signal_tf_seconds: int = SIGNAL_TF_SECONDS


@dataclass(frozen=True)
class SymbolSignals:
    mtf: MtfResult
    methods: tuple[GateResult, ...]


def symbol_signals(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    time_s: np.ndarray,
    params: SymbolParams = SymbolParams(),
) -> SymbolSignals:
    scores = [
        ha_score(open_, high, low, close, bucket_ids(time_s, params.signal_tf_seconds * mult))
        for mult in params.ha_multipliers
    ]
    mtf = mtf_confirm(scores, params.ha_weights, params.tf_need, params.tf_on)
    trend = trend_signal(high, low, close, params.atr_len, params.td_len, params.td_thr)
    momentum = momentum_signal(close, params.mo_len, params.mo_z_len, params.mo_thr)
    volat = volatility_signal(
        open_, high, low, close, params.va_slow, params.va_fast, params.va_thr, params.va_calm
    )
    vol = volume_signal(
        high, low, close, volume, params.atr_len, params.hv_len, params.hv_band, params.hv_thr,
        params.hv_flow,
    )  # fmt: skip
    raw = (
        (trend.state, trend.strength),
        (momentum.state, momentum.strength),
        (volat.state, volat.strength),
        (vol.state, vol.strength),
    )
    methods = tuple(
        run_gate(
            state == 1,
            state == -1,
            mtf.ok_long,
            mtf.ok_short,
            strength,
            high,
            low,
            params.cool,
            params.pend_max,
            params.conf_bars,
        )  # fmt: skip
        for state, strength in raw
    )
    return SymbolSignals(mtf=mtf, methods=methods)
