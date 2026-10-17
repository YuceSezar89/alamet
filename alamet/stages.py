"""
Bütün-sistem doğrulama basamakları: aynı ayar kümesi hem orijinal DEPV2 debug kopyasının girdi
varsayılanlarını (Pine) hem de Python motorunun parametrelerini (Python) belirler.

Her basamak bir öncekine ayar ekler; bir basamak tutmazsa sorun yalnızca eklenen kuralda aranır.
Öğrenen parçalar (havuz, Kelly motoru, dinamik kasa, Devisso çarpanı, hedef puanı) S17'ye kadar
kademeli açılır; S17 orijinalin varsayılan ayarlarının tamamıdır (hiç debug geçersiz kılması yok).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from alamet.divergence import DivergenceParams
from alamet.engine import TradeParams
from alamet.entry_quality import EntryQualityParams
from alamet.kelly import KellyParams
from alamet.own_divergence import MODE_SIGMA
from alamet.panel import PanelParams
from alamet.pool import PoolParams
from alamet.score import DR_OWN, DR_XS, RANK_RAW, TGT_BIG, TGT_NEAR
from alamet.selector import SelectorParams
from alamet.shadow import ShadowParams
from alamet.symbol import SymbolParams

BASE_PINE = {
    "sigMode": "Aynı anda conf",
    "usePool": False,
    "engOn": False,
    "useDyn": False,
    "drOn": False,
    "useTgt": False,
}

S0_PINE = {
    "maxLongSlots": 1,
    "swapEdge": 1000.0,
    "useGive": False,
    "ddHardOn": False,
    "dayLossOn": False,
    "lockOn": False,
    "capOn": False,
    "sizeMode": "Sabit marj × kaldıraç",
}


AUTO_MODE = "Otomatik (emtia başına en iyi)"
CONF_SELECTOR = SelectorParams(mode=4, conf_n=2)
AUTO_SELECTOR = SelectorParams(mode=6)


@dataclass(frozen=True)
class Stage:
    pine: dict
    n_top: int
    trade: TradeParams | None
    selector: SelectorParams = CONF_SELECTOR
    extra: dict = field(default_factory=dict)
    div: dict = field(default_factory=dict)


def _trade(**kw) -> TradeParams:
    base = {
        "max_slots": 1, "swap_edge": 1000.0, "use_give": False, "dd_hard_on": False,
        "day_loss_on": False, "lock_on": False, "cap_on": False, "risk_mode": False,
    }  # fmt: skip
    base.update(kw)
    return TradeParams(**base)


def _stage(parent: Stage | None, pine: dict, n_top: int | None = None, **trade) -> Stage:
    if parent is None:
        merged_pine = {**BASE_PINE, **S0_PINE, **pine}
        return Stage(merged_pine, n_top or 2, _trade(**trade))
    merged_pine = {**parent.pine, **pine}
    params = replace(parent.trade, **trade)
    return Stage(merged_pine, n_top or parent.n_top, params)


_S0 = _stage(None, {})
_S1 = _stage(_S0, {"sizeMode": "Risk %", "capOn": True}, risk_mode=True, cap_on=True)
_S2 = _stage(
    _S1, {"maxLongSlots": 4, "swapEdge": 0.3, "nTop": 4}, n_top=4, max_slots=4, swap_edge=0.3
)
_S3 = _stage(_S2, {"useGive": True}, use_give=True)
_S4 = _stage(
    _S3,
    {"useStop": True, "stopR": 2.0, "useTimeStop": True, "maxHold": 48, "useTp": True, "tpR": 3.0},
    use_stop=True, stop_r=2.0, use_time_stop=True, max_hold=48, use_tp=True, tp_r=3.0,
)  # fmt: skip
_S5 = _stage(
    _S4,
    {
        "ddHardOn": True, "ddHardPct": 6.0, "dayLossOn": True, "dayLossPct": 1.5,
        "lockOn": True, "lockArm": 1.0,
    },
    dd_hard_on=True, dd_hard_pct=6.0, day_loss_on=True, day_loss_pct=1.5,
    lock_on=True, lock_arm=1.0,
)  # fmt: skip

_S6 = _stage(
    _S5,
    {
        "ddHardPct": 1.0, "ddHardBars": 24, "dayLossPct": 0.5, "dayLossClose": True, "maxExp": 0.5,
    },
    dd_hard_pct=1.0, dd_hard_bars=24, day_loss_pct=0.5, day_loss_close=True, max_exp=0.5,
)  # fmt: skip

_S7 = replace(_S5, pine={**_S5.pine, "sigMode": AUTO_MODE}, selector=AUTO_SELECTOR)

_SCORES_ON = {"useTgt": True, "drOn": True, "drMinM": 1.0, "drMaxM": 1.0}
_S8 = replace(
    _S5,
    pine={**_S5.pine, **_SCORES_ON},
    trade=replace(_S5.trade, dr_on=True, dr_min_m=1.0, dr_max_m=1.0),
    extra={"use_target": True, "dr_rank": True},
)
_S9 = replace(
    _S8,
    pine={**_S8.pine, "drMinM": 0.4, "drMaxM": 2.0, "drBlockOn": True},
    trade=replace(_S8.trade, dr_min_m=0.4, dr_max_m=2.0, dr_block_on=True),
)
_S10 = replace(
    _S9,
    pine={**_S9.pine, "useWOp": True, "useMOp": True, "maxDOpDist": 3.0},
    trade=replace(
        _S9.trade,
        quality=EntryQualityParams(
            use_weekly_open=True, use_monthly_open=True, max_daily_open_dist=3.0
        ),
    ),
)

_S11 = replace(_S9, pine={**_S9.pine, "useDyn": True}, trade=replace(_S9.trade, use_dyn=True))
_DYN_TIGHT = {
    "dynMinN": 1.0, "dynGain": 3.0, "dynDecay": 0.9, "dynDecayBars": 24, "killN": 2.0,
    "killR": 0.1, "ddThrottle": 0.5, "ddHalt": 1.5, "haltBars": 24, "streakN": 1,
}  # fmt: skip
_S12 = replace(
    _S11,
    pine={**_S11.pine, **_DYN_TIGHT},
    trade=replace(
        _S11.trade, dyn_min_n=1.0, dyn_gain=3.0, dyn_decay=0.9, dyn_decay_bars=24, kill_n=2.0,
        kill_r=0.1, dd_throttle=0.5, dd_halt=1.5, halt_bars=24, streak_n=1,
    ),
)  # fmt: skip

_S13 = replace(
    _S9,
    pine={**_S9.pine, "engOn": True},
    trade=replace(_S9.trade, kelly=KellyParams(on=True)),
)
_KELLY_TIGHT = {"engPrior": 2.0, "engMinN": 2.0, "engBlock": 0.05, "engDecayBars": 24}
_S14 = replace(
    _S11,
    pine={**_S11.pine, "engOn": True, **_KELLY_TIGHT},
    trade=replace(
        _S11.trade,
        kelly=KellyParams(on=True, prior=2.0, min_n=2.0, block=0.05, decay_bars=24),
    ),
)

_S15 = replace(
    _S9,
    pine={**_S9.pine, "usePool": True},
    trade=replace(_S9.trade, pool=PoolParams(on=True)),
)
_POOL_TIGHT = {
    "poolSize": 4, "poolMinN": 1.0, "poolMinNet": 0.0, "poolDecayBars": 24, "poolForgive": 0.05,
    "top1Floor": -0.05,
}  # fmt: skip
_S16 = replace(
    _S13,
    pine={**_S13.pine, "usePool": True, **_POOL_TIGHT},
    trade=replace(
        _S13.trade,
        pool=PoolParams(
            on=True, size=4, min_n=1.0, min_net=0.0, decay_bars=24, forgive=0.05, top1_floor=-0.05
        ),
    ),
)

_S17 = Stage(
    {}, 2, TradeParams(
        dr_on=True, use_dyn=True, kelly=KellyParams(on=True), pool=PoolParams(on=True),
    ), AUTO_SELECTOR, {"use_target": True, "dr_rank": True},
)  # fmt: skip

_S9_EXTRA = _S9.extra
_S18 = replace(
    _S9,
    pine={**_S9.pine, "maxExp": 0.5, "swapEdge": 0.0},
    trade=replace(_S9.trade, max_exp=0.5, swap_edge=0.0),
)
_S19 = replace(
    _S9,
    pine={**_S9.pine, "vpMode": "Sadece SHORT", "reEntry": False},
    trade=replace(_S9.trade, allow_long=False, re_entry=False),
    div={"allow_long": False},
)
_S20 = replace(
    _S9,
    pine={**_S9.pine, "vpMinPow": 30.0, "useExt": False, "useDOp": False},
    trade=replace(
        _S9.trade,
        quality=EntryQualityParams(min_power=30.0, use_ext=False, use_daily_open=False),
    ),
)
_S21 = replace(
    _S9,
    pine={**_S9.pine, "vpTfOn": False, "sigMode": "MOMENTUM (tek)"},
    selector=SelectorParams(mode=1),
    extra={**_S9_EXTRA, "symbol": SymbolParams(tf_on=False)},
)
_S22 = replace(
    _S9,
    pine={**_S9.pine, "tgtMode": TGT_NEAR, "drSrc": DR_OWN},
    extra={**_S9_EXTRA, "tgt_mode": TGT_NEAR, "dr_src": DR_OWN},
)
_S23 = replace(
    _S9,
    pine={**_S9.pine, "tgtMode": TGT_BIG, "drSrc": DR_XS},
    extra={**_S9_EXTRA, "tgt_mode": TGT_BIG, "dr_src": DR_XS},
)
_S24 = replace(
    _S9,
    pine={**_S9.pine, "ownMode": MODE_SIGMA, "dblPosOnly": True},
    extra={**_S9_EXTRA, "own_mode": MODE_SIGMA},
    div={"dbl_pos_only": True},
)
_S25 = replace(_S9, pine={**_S9.pine, "sigMode": "Sıralı conf"}, selector=SelectorParams(mode=5))

_S26 = replace(
    _S9,
    pine={
        **_S9.pine,
        "ownLatch": False,
        "ownAllowNoHist": True,
        "needDbl": False,
        "capMode": "Marj (notional / maks. kaldıraç)",
        "capPct": 5.0,
    },
    trade=replace(_S9.trade, cap_risk_mode=False, cap_pct=5.0),
    extra={**_S9_EXTRA, "own": {"latch": False, "allow_no_hist": True}},
    div={"need_dbl": False},
)
_S27 = replace(
    _S9,
    pine={**_S9.pine, "useOwnFirst": False, "rankMode": "Ham %", "drRankOn": False},
    extra={
        **_S9_EXTRA,
        "own": {"use_own_first": False},
        "rank_mode": RANK_RAW,
        "dr_rank": False,
    },
)
_S28 = replace(
    _S11,
    pine={
        **_S11.pine,
        "engOn": True,
        "engReplace": False,
        "poolProb": False,
        "usePool": True,
        "useKill": False,
        "useM3": False,
    },
    trade=replace(
        _S11.trade,
        use_kill=False,
        kelly=KellyParams(on=True, replace=False),
        pool=PoolParams(on=True, prob=False),
    ),
    selector=SelectorParams(mode=4, conf_n=2, enabled=(True, True, True, False)),
)
_S29 = replace(
    _S13,
    pine={
        **_S13.pine,
        "startCap": 5000.0,
        "riskPct": 1.0,
        "maxLev": 3.0,
        "capPct": 0.5,
        "feeIn": 0.05,
        "feeOut": 0.05,
        "slipPct": 0.02,
    },
    trade=replace(
        _S13.trade,
        start_cap=5000.0,
        risk_pct=1.0,
        max_lev=3.0,
        cap_pct=0.5,
        fee_in=0.05,
        fee_out=0.05,
        slippage=0.02,
    ),
)
_S31 = replace(
    _S15,
    pine={**_S15.pine, "poolTop1": False},
    trade=replace(_S15.trade, pool=PoolParams(on=True, top1=False)),
)
_S30 = replace(
    _S7, pine={**_S7.pine, "autoBlock": False}, selector=SelectorParams(mode=6, auto_block=False)
)

_S32 = replace(
    _S9,
    pine={**_S9.pine, "tdLen": 18, "tdThr": 0.4, "moLen": 9, "moZLen": 30, "moThr": 0.8},
    extra={
        **_S9_EXTRA,
        "symbol": SymbolParams(td_len=18, td_thr=0.4, mo_len=9, mo_z_len=30, mo_thr=0.8),
    },
)
_S33 = replace(
    _S9,
    pine={
        **_S9.pine,
        "vaSlow": 10,
        "vaFast": 4,
        "vaThr": 1.2,
        "vaCalm": 0.7,
        "hvLen": 9,
        "hvBand": 0.4,
        "hvThr": 1.5,
        "hvFlow": 0.2,
    },
    extra={
        **_S9_EXTRA,
        "symbol": SymbolParams(
            va_slow=10,
            va_fast=4,
            va_thr=1.2,
            va_calm=0.7,
            hv_len=9,
            hv_band=0.4,
            hv_thr=1.5,
            hv_flow=0.2,
        ),
    },
)
_S34 = replace(
    _S9,
    pine={
        **_S9.pine,
        "vpM1": 3,
        "vpM2": 4,
        "vpM3": 6,
        "vpM4": 8,
        "vpW1": 1.0,
        "vpW2": 2.0,
        "vpW3": 3.0,
        "vpW4": 5.0,
        "vpTfNeed": 2,
    },
    extra={
        **_S9_EXTRA,
        "symbol": SymbolParams(
            ha_multipliers=(3, 4, 6, 8), ha_weights=(1.0, 2.0, 3.0, 5.0), tf_need=2
        ),
    },
)
_S35 = replace(
    _S9,
    pine={**_S9.pine, "vpCool": 6, "vpPendMax": 8, "vpConfBars": 2, "atrPeriod": 14},
    extra={
        **_S9_EXTRA,
        "symbol": SymbolParams(cool=6, pend_max=8, conf_bars=2, atr_len=14),
    },
)
_S36 = replace(
    _S9,
    pine={**_S9.pine, "dOpenHr": 3, "drRsiLen": 5, "drLook": 90},
    extra={**_S9_EXTRA, "daily_open_hour": 3, "dr_rsi_len": 5, "dr_look": 90},
)
_S37 = replace(
    _S9,
    pine={**_S9.pine, "confN": 3, "useM0": False},
    selector=SelectorParams(mode=4, conf_n=3, enabled=(False, True, True, True)),
)
_S38 = replace(
    _S9,
    pine={
        **_S9.pine,
        "sigMode": "Sıralı conf",
        "seqOrder": "4,3,2,1",
        "seqN": 2,
        "seqWin": 12,
    },
    selector=SelectorParams(mode=5, seq_order=(3, 2, 1, 0), seq_n=2, seq_win=12),
)
_S39 = replace(
    _S7,
    pine={**_S7.pine, "autoMinN": 1, "autoMinNet": 0.2},
    selector=SelectorParams(mode=6, auto_min_n=1, auto_min_net=0.2),
)
_S40 = replace(
    _S9,
    pine={
        **_S9.pine,
        "gapMin": 0.2,
        "maxDiv": 6,
        "zMin": 0.5,
        "minPrev": 3,
        "minScore": 0.3,
        "minShare": 60.0,
        "ownMode": MODE_SIGMA,
        "ownN": 5,
        "ownMult": 0.8,
        "ownSigma": 1.5,
        "stayResetBars": 5,
        "rank2Mult": 0.3,
        "coolBars": 6,
        "maxExt": 2.5,
        "rUnit": 1.0,
        "tgtW": 0.4,
        "tgtCap": 5.0,
        "drPow": 2.0,
        "drRankW": 2.0,
        "drBlockPct": 80.0,
        "nTop": 3,
        "tradeMinN": 2,
    },
    n_top=3,
    trade=replace(
        _S9.trade,
        stay_reset_bars=5,
        rank2_mult=0.3,
        r_unit=1.0,
        dr_pow=2.0,
        dr_block_pct=80.0,
        quality=EntryQualityParams(cool_bars=6, max_ext=2.5),
    ),
    extra={
        **_S9_EXTRA,
        "gap_min": 0.2,
        "own_mode": MODE_SIGMA,
        "own": {"own_n": 5, "mult": 0.8, "sigma": 1.5, "min_prev": 3},
        "tgt_w": 0.4,
        "tgt_cap": 5.0,
        "dr_rank_w": 2.0,
    },
    div={
        "max_div": 6,
        "z_min": 0.5,
        "min_prev": 3,
        "min_score": 0.3,
        "min_share": 60.0,
        "trade_min_n": 2,
    },
)
_S41 = replace(
    _S11,
    pine={
        **_S11.pine,
        "minMult": 0.5,
        "maxMult": 1.5,
        "ddMult": 0.3,
        "streakMult": 0.3,
        "lockFrac": 0.25,
        "lockBars": 24,
        "negMult": 0.8,
        "giveArm": 1.0,
        "giveFrac": 0.3,
    },
    trade=replace(
        _S11.trade,
        dyn_min_mult=0.5,
        dyn_max_mult=1.5,
        dd_mult=0.3,
        streak_mult=0.3,
        lock_frac=0.25,
        lock_bars=24,
        neg_mult=0.8,
        give_arm=1.0,
        give_frac=0.3,
    ),
)
_S42 = replace(
    _S9,
    pine={
        **_S9.pine,
        "sizeMode": "Sabit marj × kaldıraç",
        "fixMargin": 50.0,
        "leverage": 3.0,
        "maxLev": 4.0,
    },
    trade=replace(_S9.trade, risk_mode=False, fix_margin=50.0, leverage=3.0, max_lev=4.0),
)
_S43 = replace(
    _S16,
    pine={
        **_S16.pine,
        "engKelly": 0.5,
        "engZ": 0.0,
        "engExplore": 1.0,
        "engMin": 0.1,
        "engMax": 2.0,
        "engClip": 2.0,
        "engDecay": 0.9,
        "engDdScale": False,
        "poolDecay": 0.5,
        "poolMinNet": 0.5,
    },
    trade=replace(
        _S16.trade,
        kelly=KellyParams(
            on=True,
            kelly=0.5,
            z=0.0,
            explore=1.0,
            min_mult=0.1,
            max_mult=2.0,
            clip=2.0,
            decay=0.9,
            dd_scale=False,
        ),
        pool=replace(_S16.trade.pool, decay=0.5, min_net=0.5),
    ),
)
_S44 = replace(
    _S7,
    pine={
        **_S7.pine,
        "feeIn": 0.05,
        "feeOut": 0.05,
        "slipPct": 0.02,
        "rUnit": 1.0,
        "riskPct": 1.0,
        "startCap": 5000.0,
        "dynDecay": 0.9,
        "dynDecayBars": 24,
    },
    trade=replace(
        _S7.trade,
        fee_in=0.05,
        fee_out=0.05,
        slippage=0.02,
        r_unit=1.0,
        risk_pct=1.0,
        start_cap=5000.0,
        dyn_decay=0.9,
        dyn_decay_bars=24,
    ),
)

_S45 = replace(
    _S16,
    pine={**_S16.pine, "poolDecay": 0.5, "poolMinNet": 0.5},
    trade=replace(_S16.trade, pool=replace(_S16.trade.pool, decay=0.5, min_net=0.5)),
)
_S46 = replace(
    _S13,
    pine={
        **_S13.pine,
        "engKelly": 0.5,
        "engZ": 0.0,
        "engExplore": 1.0,
        "engMin": 0.1,
        "engMax": 2.0,
        "engClip": 2.0,
        "engDecay": 0.9,
        "engDdScale": False,
    },
    trade=replace(
        _S13.trade,
        kelly=KellyParams(
            on=True,
            kelly=0.5,
            z=0.0,
            explore=1.0,
            min_mult=0.1,
            max_mult=2.0,
            clip=2.0,
            decay=0.9,
            dd_scale=False,
        ),
    ),
)

_S47 = replace(
    _S9,
    pine={**_S9.pine, "useM1": False, "useM2": False},
    selector=SelectorParams(mode=4, conf_n=2, enabled=(True, False, False, True)),
)

_S48 = replace(
    _S7,
    pine={**_S7.pine, "useM0": False},
    selector=SelectorParams(mode=6, enabled=(False, True, True, True)),
)
_S49 = replace(
    _S13,
    pine={**_S13.pine, "needDbl": False},
    div={"need_dbl": False},
)

_S50 = replace(
    _S9,
    pine={**_S9.pine, "holdShort": 2, "holdLong": 6, "logN": 5},
    trade=replace(_S9.trade, hold_short=2, hold_long=6, log_size=5),
)

STAGES: dict[str, Stage] = {
    "S0": _S0, "S1": _S1, "S2": _S2, "S3": _S3, "S4": _S4, "S5": _S5, "S6": _S6, "S7": _S7,
    "S8": _S8, "S9": _S9, "S10": _S10, "S11": _S11, "S12": _S12, "S13": _S13, "S14": _S14,
    "S15": _S15, "S16": _S16, "S17": _S17,
    "S18": _S18, "S19": _S19, "S20": _S20, "S21": _S21, "S22": _S22, "S23": _S23,
    "S24": _S24, "S25": _S25, "S26": _S26, "S27": _S27, "S28": _S28, "S29": _S29,
    "S30": _S30, "S31": _S31, "S32": _S32, "S33": _S33, "S34": _S34, "S35": _S35,
    "S36": _S36, "S37": _S37, "S38": _S38, "S39": _S39, "S40": _S40, "S41": _S41,
    "S42": _S42, "S43": _S43, "S44": _S44, "S45": _S45, "S46": _S46, "S47": _S47, "S48": _S48, "S49": _S49, "S50": _S50,
}  # fmt: skip
SIGNAL_STAGES = {
    "SIG": ({**BASE_PINE}, CONF_SELECTOR, {}),
    "SIGA": ({**BASE_PINE, "sigMode": AUTO_MODE}, AUTO_SELECTOR, {}),
    "SIGT": ({**BASE_PINE, "useTgt": True}, CONF_SELECTOR, {"use_target": True}),
    "SIGD": (
        {**BASE_PINE, **_SCORES_ON}, CONF_SELECTOR, {"use_target": True, "dr_rank": True}
    ),
}  # fmt: skip


def shadow_params(trade: TradeParams) -> ShadowParams:
    """Gölge kasa, Pine'daki gibi motorla aynı girdileri paylaşır (ücret, risk, kasa, sönümleme)."""
    return ShadowParams(
        fee_in=trade.fee_in,
        fee_out=trade.fee_out,
        slippage=trade.slippage,
        r_unit=trade.r_unit,
        risk_mode=trade.risk_mode,
        start_cap=trade.start_cap,
        risk_pct=trade.risk_pct,
        max_lev=trade.max_lev,
        fix_margin=trade.fix_margin,
        leverage=trade.leverage,
        decay=trade.dyn_decay,
        decay_bars=trade.dyn_decay_bars,
    )


def panel_params(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    n_top: int = 2,
    max_slots: int = 4,
    selector: SelectorParams = CONF_SELECTOR,
    div: dict | None = None,
    shadow: ShadowParams = ShadowParams(),
    **extra,
) -> PanelParams:
    return PanelParams(
        **extra,
        selector=selector,
        shadow=shadow,
        divergence=DivergenceParams(max_slots=max_slots, **(div or {})),
        n_top=n_top,
    )


def with_signal_tf(panel: PanelParams, seconds: int) -> PanelParams:
    """Panel parametrelerinde sinyal TF'sini değiştirir (HA katları bu TF'nin çarpanıdır)."""
    return replace(panel, symbol=replace(panel.symbol, signal_tf_seconds=seconds))


def stage_params(name: str, **trade_overrides) -> tuple[PanelParams, TradeParams]:
    """Basamağın (panel, motor) parametreleri; `trade_overrides` motor alanlarını değiştirir ve
    gölge kasa bunlarla birlikte yeniden türetilir."""
    stage = STAGES[name]
    trade = replace(stage.trade, **trade_overrides) if trade_overrides else stage.trade
    panel = panel_params(
        stage.n_top, trade.max_slots, stage.selector, stage.div, shadow_params(trade), **stage.extra
    )
    return panel, trade


QUALITY = EntryQualityParams()
