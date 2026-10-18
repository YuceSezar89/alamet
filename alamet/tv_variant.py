"""
TradingView doğrulaması için DEBUG_S17 kopyası üretir: seçilen girdilerin varsayılanlarını değiştirir
ve aynı ayarların `verify_system --set` karşılığını yazar. Böylece Pine ile Python aynı ayarla koşar.

    python -m alamet.tv_variant V1 --pine .../DEBUG_S17.pine --out .../DEBUG_S17_V1.pine
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SYM = "panel.symbol."
# Pine girdi adı -> Python yolu
PATHS = {
    "tdLen": SYM + "td_len", "tdThr": SYM + "td_thr",
    "moLen": SYM + "mo_len", "moZLen": SYM + "mo_z_len", "moThr": SYM + "mo_thr",
    "vaSlow": SYM + "va_slow", "vaFast": SYM + "va_fast", "vaThr": SYM + "va_thr",
    "vaCalm": SYM + "va_calm",
    "hvLen": SYM + "hv_len", "hvBand": SYM + "hv_band", "hvThr": SYM + "hv_thr",
    "hvFlow": SYM + "hv_flow",
    "atrPeriod": SYM + "atr_len",
    "vpM1": SYM + "ha_multipliers.0", "vpM2": SYM + "ha_multipliers.1",
    "vpM3": SYM + "ha_multipliers.2", "vpM4": SYM + "ha_multipliers.3",
    "vpW1": SYM + "ha_weights.0", "vpW2": SYM + "ha_weights.1",
    "vpW3": SYM + "ha_weights.2", "vpW4": SYM + "ha_weights.3",
    "vpCool": SYM + "cool", "vpPendMax": SYM + "pend_max", "vpConfBars": SYM + "conf_bars",
    "vpTfOn": SYM + "tf_on", "vpTfNeed": SYM + "tf_need",
    "maxExt": "trade.quality.max_ext",
    "useM0": "panel.selector.enabled.0", "useM1": "panel.selector.enabled.1",
    "useM2": "panel.selector.enabled.2", "useM3": "panel.selector.enabled.3",
    "confN": "panel.selector.conf_n", "seqN": "panel.selector.seq_n",
    "seqWin": "panel.selector.seq_win", "autoMinN": "panel.selector.auto_min_n",
    "autoMinNet": "panel.selector.auto_min_net", "autoBlock": "panel.selector.auto_block",
    "dynDecay": "trade.dyn_decay", "dynDecayBars": "trade.dyn_decay_bars",
    "feeIn": "trade.fee_in", "feeOut": "trade.fee_out", "slipPct": "trade.slippage",
    "rUnit": "trade.r_unit",
    "ownMode": "panel.own_mode", "ownMult": "panel.own.mult", "ownSigma": "panel.own.sigma",
    "ownN": "panel.own.own_n", "ownLatch": "panel.own.latch",
    "ownAllowNoHist": "panel.own.allow_no_hist",
    "tgtMode": "panel.tgt_mode", "tgtW": "panel.tgt_w", "tgtCap": "panel.tgt_cap",
    "drRankW": "panel.dr_rank_w", "drSrc": "panel.dr_src", "rankMode": "panel.rank_mode",
    "gapMin": "panel.gap_min",
    "maxDiv": "panel.divergence.max_div", "dblPosOnly": "panel.divergence.dbl_pos_only",
    "minScore": "panel.divergence.min_score", "minShare": "panel.divergence.min_share",
    "needDbl": "panel.divergence.need_dbl", "tradeMinN": "panel.divergence.trade_min_n",
    "maxLongSlots": "trade.max_slots",
    "startCap": "trade.start_cap", "riskPct": "trade.risk_pct", "maxLev": "trade.max_lev",
    "fixMargin": "trade.fix_margin", "leverage": "trade.leverage", "rank2Mult": "trade.rank2_mult",
    "useGive": "trade.use_give", "giveArm": "trade.give_arm", "giveFrac": "trade.give_frac",
    "swapEdge": "trade.swap_edge", "stayResetBars": "trade.stay_reset_bars",
    "reEntry": "trade.re_entry", "holdShort": "trade.hold_short", "holdLong": "trade.hold_long",
    "capOn": "trade.cap_on", "capPct": "trade.cap_pct", "maxExp": "trade.max_exp",
    "ddHardOn": "trade.dd_hard_on", "ddHardPct": "trade.dd_hard_pct",
    "ddHardBars": "trade.dd_hard_bars", "dayLossOn": "trade.day_loss_on",
    "dayLossPct": "trade.day_loss_pct", "dayLossClose": "trade.day_loss_close",
    "lockOn": "trade.lock_on", "lockArm": "trade.lock_arm", "lockFrac": "trade.lock_frac",
    "lockBars": "trade.lock_bars", "negMult": "trade.neg_mult",
    "useDyn": "trade.use_dyn", "dynMinN": "trade.dyn_min_n", "dynGain": "trade.dyn_gain",
    "minMult": "trade.dyn_min_mult", "maxMult": "trade.dyn_max_mult",
    "useKill": "trade.use_kill", "killN": "trade.kill_n", "killR": "trade.kill_r",
    "ddThrottle": "trade.dd_throttle", "ddMult": "trade.dd_mult", "ddHalt": "trade.dd_halt",
    "haltBars": "trade.halt_bars", "streakN": "trade.streak_n", "streakMult": "trade.streak_mult",
    "vpMinPow": "trade.quality.min_power", "useWOp": "trade.quality.use_weekly_open",
    "useMOp": "trade.quality.use_monthly_open",
    "maxDOpDist": "trade.quality.max_daily_open_dist", "useExt": "trade.quality.use_ext",
    "drOn": "trade.dr_on", "drMinM": "trade.dr_min_m", "drMaxM": "trade.dr_max_m",
    "zMin": "panel.divergence.z_min", "nTop": "panel.n_top", "useTp": "trade.use_tp",
    "tpR": "trade.tp_r", "useStop": "trade.use_stop", "stopR": "trade.stop_r",
    "useTimeStop": "trade.use_time_stop", "maxHold": "trade.max_hold",
    "drPow": "trade.dr_pow", "drBlockOn": "trade.dr_block_on",
    "drBlockPct": "trade.dr_block_pct", "drRsiLen": "panel.dr_rsi_len", "drLook": "panel.dr_look",
}  # fmt: skip
MODES = (
    "TREND (tek)", "MOMENTUM (tek)", "VOLATİLİTE (tek)", "HACİM (tek)", "Aynı anda conf",
    "Sıralı conf", "Otomatik (emtia başına en iyi)",
)  # fmt: skip
STRING_INPUTS = (
    "sigMode",
    "seqOrder",
    "ownMode",
    "tgtMode",
    "drSrc",
    "rankMode",
    "sizeMode",
    "capMode",
    "vpMode",
    "ssTf",
)
SHADOW_PATHS = {
    "trade.dyn_decay": "panel.shadow.decay", "trade.dyn_decay_bars": "panel.shadow.decay_bars",
    "trade.fee_in": "panel.shadow.fee_in", "trade.fee_out": "panel.shadow.fee_out",
    "trade.slippage": "panel.shadow.slippage", "trade.r_unit": "panel.shadow.r_unit",
}  # fmt: skip

VARIANTS: dict[str, dict[str, str]] = {
    "V1": {
        "tdLen": "18", "tdThr": "0.4",
        "moLen": "10", "moZLen": "36", "moThr": "0.8",
        "vaSlow": "10", "vaFast": "4", "vaThr": "1.2", "vaCalm": "0.7",
        "hvLen": "10", "hvBand": "0.4", "hvThr": "1.6", "hvFlow": "0.08",
        "atrPeriod": "14",
        "vpM3": "5", "vpM4": "8",
        "vpW1": "1.0", "vpW2": "2.0", "vpW3": "3.0", "vpW4": "5.0",
        "vpCool": "6", "vpPendMax": "8", "vpConfBars": "2",
        "maxExt": "6.0",
    },
}  # fmt: skip
VARIANTS["V1B"] = {k: x for k, x in VARIANTS["V1"].items() if k != "vpM3"}
VARIANTS["V1A"] = {"vpM3": "5", "maxExt": "6.0"}
VARIANTS["V2A"] = {  # Otomatik mod: eşikler, kapalı yöntem, gölge kasa ve ücret girdileri
    "autoMinN": "2", "autoMinNet": "0.5", "autoBlock": "false", "useM3": "false",
    "dynDecay": "0.95", "dynDecayBars": "48", "feeIn": "0.04", "feeOut": "0.03",
    "slipPct": "0.02", "rUnit": "2.0", "maxExt": "6.0",
}  # fmt: skip
VARIANTS["V2F"] = {"autoBlock": "false", "useM3": "false", "maxExt": "6.0"}
VARIANTS["V2G"] = {
    "dynDecay": "0.95", "dynDecayBars": "48", "feeIn": "0.04", "feeOut": "0.03",
    "slipPct": "0.02", "rUnit": "2.0", "maxExt": "6.0",
}  # fmt: skip
VARIANTS["V2H"] = {"autoMinN": "2", "autoMinNet": "0.5", "maxExt": "6.0"}
VARIANTS["V2B"] = {
    "sigMode": "Sıralı conf", "seqOrder": "2,1,3,4", "seqN": "2", "seqWin": "12", "maxExt": "6.0",
}  # fmt: skip
VARIANTS["V2C"] = {
    "sigMode": "Aynı anda conf", "confN": "1", "useM1": "false", "maxExt": "6.0",
}  # fmt: skip
VARIANTS["SC"] = {"confN": "1", "useM1": "false"}
VARIANTS["SA"] = {
    "autoMinN": "2", "autoMinNet": "0.5", "autoBlock": "false", "useM3": "false",
    "dynDecay": "0.95", "dynDecayBars": "48", "feeIn": "0.04", "feeOut": "0.03",
    "slipPct": "0.02", "rUnit": "2.0",
}  # fmt: skip
VARIANTS["V3A"] = {
    "ownMode": "Ort + σ", "ownSigma": "0.8", "ownN": "6", "ownAllowNoHist": "true", "maxExt": "6.0",
}  # fmt: skip
VARIANTS["V3B"] = {"ownMult": "1.3", "ownLatch": "false", "ownN": "5", "maxExt": "6.0"}
VARIANTS["V3C"] = {
    "tgtMode": "Yakın hedef", "tgtW": "0.5", "tgtCap": "6.0", "drRankW": "2.0",
    "drSrc": "Kendi geçmişi", "maxExt": "6.0",
}  # fmt: skip
VARIANTS["SG3"] = {
    "tgtMode": "Yakın hedef", "tgtW": "0.5", "tgtCap": "6.0", "drRankW": "2.0",
    "drSrc": "Kendi geçmişi",
}  # fmt: skip
VARIANTS["V3D"] = {"rankMode": "Ham %", "gapMin": "0.3", "maxExt": "6.0"}
VARIANTS["V3E"] = {
    "maxDiv": "6", "dblPosOnly": "true", "minScore": "0.5", "minShare": "60.0", "tradeMinN": "2",
    "maxExt": "6.0",
}  # fmt: skip
VARIANTS["V3F"] = {"needDbl": "false", "maxLongSlots": "2", "maxExt": "6.0"}
VARIANTS["V4A"] = {
    "startCap": "2000.0", "riskPct": "0.8", "maxLev": "3.0", "rank2Mult": "0.3", "maxExt": "6.0",
}  # fmt: skip
VARIANTS["V4B"] = {
    "sizeMode": "Sabit marj × kaldıraç", "fixMargin": "150.0", "leverage": "4.0", "maxLev": "6.0",
    "maxExt": "6.0",
}  # fmt: skip
VARIANTS["V4C"] = {"giveArm": "1.0", "giveFrac": "0.35", "maxExt": "6.0"}
VARIANTS["V4D"] = {
    "useGive": "false", "swapEdge": "0.6", "stayResetBars": "5", "reEntry": "false",
    "maxExt": "6.0",
}  # fmt: skip
VARIANTS["V4E"] = {"holdShort": "2", "holdLong": "8", "maxExt": "6.0"}
VARIANTS["TRK"] = {"maxExt": "6.0"}
VARIANTS["V5A"] = {
    "capPct": "0.5", "capMode": "Marj (notional / maks. kaldıraç)", "maxExp": "1.0", "maxExt": "6.0",
}  # fmt: skip
VARIANTS["V5B"] = {
    "ddHardPct": "1.5", "ddHardBars": "24", "dayLossPct": "0.5", "dayLossClose": "true",
    "maxExt": "6.0",
}  # fmt: skip
VARIANTS["V5C"] = {
    "lockArm": "0.5", "lockFrac": "0.25", "lockBars": "24", "negMult": "0.5", "maxExt": "6.0",
}  # fmt: skip
VARIANTS["V5D"] = {
    "ddThrottle": "1.0", "ddMult": "0.4", "ddHalt": "2.0", "haltBars": "24", "streakN": "2",
    "streakMult": "0.4", "dynMinN": "3.0", "minMult": "0.5", "maxMult": "1.5", "maxExt": "6.0",
}  # fmt: skip
VARIANTS["V5E"] = {
    "vpMinPow": "20.0", "useWOp": "true", "useMOp": "true", "maxDOpDist": "4.0", "maxExt": "6.0",
}  # fmt: skip
VARIANTS["V5F"] = {
    "drMinM": "0.6", "drMaxM": "1.6", "drPow": "1.5", "drBlockOn": "true", "drBlockPct": "80.0",
    "maxExt": "6.0",
}  # fmt: skip
VARIANTS["V5G"] = {"drRsiLen": "5", "drLook": "40", "maxExt": "6.0"}
VARIANTS["V5H"] = {"drOn": "false", "useDyn": "false", "useExt": "false"}
VARIANTS["DEF"] = {}
VARIANTS["V5I"] = {"drOn": "false", "maxExt": "6.0"}
VARIANTS["V5J"] = {"useDyn": "false", "maxExt": "6.0"}
VARIANTS["V5K"] = {"useExt": "false"}
VARIANTS["V6A"] = {"vpMode": "Sadece LONG", "maxExt": "6.0"}
VARIANTS["V6B"] = {"vpMode": "Sadece SHORT", "maxExt": "6.0"}
VARIANTS["V6C"] = {
    "ssTf": "5", "maxExt": "6.0", "zMin": "0.5", "nTop": "3", "tdThr": "0.4", "vpTfNeed": "2",
    "riskPct": "0.8",
}  # fmt: skip
VARIANTS["V6D"] = {
    "ssTf": "5", "useTp": "true", "tpR": "1.0", "useStop": "true", "stopR": "2.0",
    "useTimeStop": "true", "maxHold": "48", "maxExt": "6.0",
}  # fmt: skip
VARIANTS["V7A"] = {"ssTf": "30", "maxExt": "6.0"}
VARIANTS["V7B"] = {"ssTf": "60", "maxExt": "6.0"}
VARIANTS["V7C"] = {"ssTf": "1", "maxExt": "6.0"}
VARIANTS["V8A"] = {"vpTfOn": "false", "maxExt": "6.0"}
VARIANTS["S30"] = {"ssTf": "30"}
VARIANTS["S60"] = {"ssTf": "60"}
VARIANTS["S1"] = {"ssTf": "1"}
VARIANTS["V2D"] = {"sigMode": "MOMENTUM (tek)", "maxExt": "6.0"}
VARIANTS["V2E"] = {"sigMode": "HACİM (tek)", "maxExt": "6.0"}


def _pine_value(name: str, raw: str) -> str:
    return f'"{raw}"' if name in STRING_INPUTS else raw


def apply(source: str, changes: dict[str, str]) -> str:
    """Her girdinin ilk argümanını (varsayılan) değiştirir; bulunamazsa hata verir."""
    for name, raw in changes.items():
        value = '"[^"]*"' if name in STRING_INPUTS else "[^,]+"
        pattern = re.compile(
            rf"^(\s*{re.escape(name)}\s*=\s*input\.(?:int|float|bool|string|timeframe)\()({value})(,)",
            re.MULTILINE,
        )
        source, count = pattern.subn(
            lambda m, v=raw, n=name: m.group(1) + _pine_value(n, v) + m.group(3), source
        )
        if count != 1:
            raise ValueError(f"{name}: Pine'da {count} eşleşme (1 olmalı)")
    return source


def set_args(changes: dict[str, str], signal: bool = False) -> list[str]:
    """`signal=True`: yalnız panel düzeyi (sinyal basamakları); gölge kasa alanları panel yoluna
    çevrilir, motor alanları atılır."""
    args: list[str] = []
    for name, raw in changes.items():
        if name == "sigMode":
            args.append(f"panel.selector.mode={MODES.index(raw)}")
        elif name == "ssTf":
            continue
        elif name == "vpMode":
            if raw == "Sadece LONG":
                args += ["trade.allow_short=false", "panel.divergence.allow_short=false"]
            elif raw == "Sadece SHORT":
                args += ["trade.allow_long=false", "panel.divergence.allow_long=false"]
        elif name == "capMode":
            args.append(f"trade.cap_risk_mode={'false' if raw.startswith('Marj') else 'true'}")
        elif name == "sizeMode":
            args.append(f"trade.risk_mode={'false' if raw.startswith('Sabit') else 'true'}")
        elif name == "seqOrder":
            order = [int(part) - 1 for part in raw.split(",")]
            args += [f"panel.selector.seq_order.{i}={m}" for i, m in enumerate(order)]
        else:
            path = PATHS[name]
            if signal:
                path = SHADOW_PATHS.get(path, path)
                if path.startswith("trade."):
                    continue
            args.append(f"{path}={raw}")
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--pine", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--signal", action="store_true", help="sinyal basamağı (SIG/SIGA) için")
    args = parser.parse_args()
    changes = VARIANTS[args.variant]
    args.out.write_text(apply(args.pine.read_text(encoding="utf-8"), changes), encoding="utf-8")
    print(f"yazıldı: {args.out}")
    print(" ".join(f"--set {item}" for item in set_args(changes, args.signal)))


if __name__ == "__main__":
    main()
