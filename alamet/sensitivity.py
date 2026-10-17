"""
M2 — Ayar duyarlılığı: varsayılan (S17) ayarlara karşı tek ayar değişimlerinin DAVRANIŞ etkisi.

Kâr/zarar ölçülmez; yalnızca gecikme (girişte sinyalden beri hareket), işlem sıklığı, kasa
meşguliyeti ve uygun adayların hangi kapıda elendiği. Sistem yola bağımlı olduğundan her ayar
birden çok pencerede (farklı bitiş noktalarında) çalıştırılır ve varsayılana KARŞI eşleştirilmiş
fark raporlanır. Canlı bota ve `orijinal` profile dokunmaz.

  python -m alamet.sensitivity --windows 8 --workers 6 --out docs/reports/m2_ayar_duyarliligi.md
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

import alamet.engine as eng
from alamet.binance_data import DEFAULT_ROOT, infer_tf, load_universe_binance
from alamet.daily import daily_from_bars, daily_pack, day_inputs, map_to_bars, window_daily
from alamet.entry_quality import LABELS
from alamet.kelly import KellyEngine
from alamet.own_divergence import MODE_SIGMA
from alamet.stages import stage_params, with_signal_tf
from alamet.system import run_system

BARS = 4000
SPACING = 1344  # 14 gün
VARIANTS = {
    "varsayılan": {},
    "maxExt 6": {"max_ext": 6.0},
    "maxExt 8": {"max_ext": 8.0},
    "zMin 0,5": {"z_min": 0.5},
    "nTop 3": {"n_top": 3},
    "ownMode Ort+σ": {"own_mode": MODE_SIGMA},
}
_STATE: dict = {}
_COUNT: collections.Counter = collections.Counter()
_CUR: dict = {}


def _install_probes() -> None:
    """Giriş denemelerini sınıflayan ölçüm kancaları (davranışı değiştirmez)."""
    if _STATE.get("probes"):
        return
    orig_try, orig_flags = eng.TradeEngine.try_entry, eng.entry_flags
    orig_mult, orig_eval = eng.TradeEngine.entry_multiplier, KellyEngine.evaluate
    orig_open = eng.TradeEngine.open_trade

    def p_flags(*args, **kwargs):
        result = orig_flags(*args, **kwargs)
        _CUR["flags"] = int(result)
        return result

    def p_mult(self, *args, **kwargs):
        result = orig_mult(self, *args, **kwargs)
        _CUR["kill"] = bool(result[1])
        return result

    def p_eval(self, *args, **kwargs):
        result = orig_eval(self, *args, **kwargs)
        _CUR["kelly_block"] = bool(result[1])
        return result

    def p_open(self, *args, **kwargs):
        _CUR["opened"] = True
        return orig_open(self, *args, **kwargs)

    def p_try(self, step_i, bar, side, rank, k, halted, drawdown):
        _CUR.clear()
        _CUR["halted"] = bool(halted)
        orig_try(self, step_i, bar, side, rank, k, halted, drawdown)
        _COUNT["deneme"] += 1
        if _CUR.get("opened"):
            _COUNT["açıldı"] += 1
        elif _CUR["halted"]:
            _COUNT["kasa freni"] += 1
        elif _CUR.get("flags"):
            _COUNT["uzama" if _CUR["flags"] & 2 else "diğer filtre"] += 1
        elif _CUR.get("kill") or _CUR.get("kelly_block"):
            _COUNT["blok"] += 1
        else:
            _COUNT["slot/ısı"] += 1

    eng.entry_flags = p_flags
    eng.TradeEngine.entry_multiplier = p_mult
    KellyEngine.evaluate = p_eval
    eng.TradeEngine.open_trade = p_open
    eng.TradeEngine.try_entry = p_try
    _STATE["probes"] = True


def _universe(root: str, symbols: tuple[str, ...]):
    key = ("uni", root)
    if key not in _STATE:
        tf = infer_tf(Path(root))
        _STATE[key] = (load_universe_binance(Path(root), symbols, tf=tf, union=True), tf)
    return _STATE[key]


def run_one(job: tuple) -> dict:
    """Bir (ayar, pencere) koşusu → davranış metrikleri."""
    name, mods, window, root, symbols = job
    _install_probes()
    _COUNT.clear()
    full, tf = _universe(root, symbols)
    step = tf.seconds
    end = full.bars - window * SPACING
    uni = full.slice_to(end)
    start = uni.bars - (BARS - 1)
    live = int(uni.time[-1]) + step
    daily = window_daily(daily_from_bars(uni, step), uni.time[start], live, BARS, step)
    panel, trade = stage_params("S17")
    panel = with_signal_tf(panel, step)
    if "max_ext" in mods:
        trade = replace(trade, quality=replace(trade.quality, max_ext=mods["max_ext"]))
    if "z_min" in mods:
        panel = replace(panel, divergence=replace(panel.divergence, z_min=mods["z_min"]))
    if "own_mode" in mods:
        panel = replace(panel, own_mode=mods["own_mode"])
    if "n_top" in mods:
        panel = replace(panel, n_top=mods["n_top"])
    pack = daily_pack(daily, panel.dr_rsi_len, panel.dr_look)
    day = day_inputs(map_to_bars(pack, daily.time, uni.time, tf_seconds=step))
    result = run_system(uni, panel, trade, True, start=start, day=day)
    trades = result.trades
    moves = np.array([t.side * t.signal_pct for t in trades]) if trades else np.array([np.nan])
    elig = np.asarray(result.panel.divergence["eligible"])[-BARS:]
    busy = (np.asarray(result.sides)[-BARS:] != 0).any(axis=1)
    signal = np.nan_to_num(np.asarray(result.panel.types)[-BARS:]) != 0
    own_ok = np.asarray(result.panel.own_ok)[-BARS:].astype(bool) & signal
    tries = max(_COUNT["deneme"], 1)
    return {
        "ayar": name, "pencere": window, "işlem": len(trades),
        "işlem_gün": len(trades) / (BARS / 96),
        "giriş_hareket_medyan": float(np.nanmedian(moves)),
        "giriş_hareket_p75": float(np.nanpercentile(moves, 75)),
        "meşguliyet": float(busy.mean() * 100),
        "uygun_bar_başı": float(elig.sum(axis=1).mean()),
        "kendi_içi_geçen": float(own_ok.sum() / max(signal.sum(), 1) * 100),
        "deneme": _COUNT["deneme"], "uzama_oranı": _COUNT["uzama"] / tries * 100,
        "açıldı_oranı": _COUNT["açıldı"] / tries * 100,
    }  # fmt: skip


METRICS = [
    ("işlem_gün", "İşlem / gün", 2),
    ("giriş_hareket_medyan", "Girişte sinyalden beri hareket, medyan %", 2),
    ("giriş_hareket_p75", "… %75'lik %", 2),
    ("meşguliyet", "Kasa meşguliyeti % (≥1 açık)", 1),
    ("kendi_içi_geçen", "Kendi içi kapıyı geçen % (sinyalli semboller içinde)", 1),
    ("uygun_bar_başı", "Uygun aday / bar", 2),
    ("uzama_oranı", "Girişi uzama filtresine takılan %", 1),
    ("açıldı_oranı", "Denemelerden açılan %", 1),
]


def report(rows: list[dict], windows: int) -> str:
    base = {r["pencere"]: r for r in rows if r["ayar"] == "varsayılan"}
    lines = [
        "# M2 — Ayar duyarlılığı (S17'ye karşı tek ayar değişimi)",
        "",
        f"168 sembol, 15 dk, her koşu {BARS} bar (≈{BARS // 96} gün), {windows} farklı bitiş noktası "
        f"({SPACING // 96} gün arayla). Kâr ölçülmedi; yalnızca davranış. Hücre: medyan "
        "(en az … en çok), ve varsayılana karşı eşleştirilmiş fark (medyan Δ).",
        "",
        "| Ayar | " + " | ".join(label for _, label, _ in METRICS) + " |",
        "|---|" + "---:|" * len(METRICS),
    ]
    for name in VARIANTS:
        sel = [r for r in rows if r["ayar"] == name]
        cells = []
        for key, _, digits in METRICS:
            vals = np.array([r[key] for r in sel], dtype=float)
            text = f"{np.nanmedian(vals):.{digits}f} ({np.nanmin(vals):.{digits}f}–{np.nanmax(vals):.{digits}f})"
            if name != "varsayılan":
                diff = np.array([r[key] - base[r["pencere"]][key] for r in sel], dtype=float)
                same = int((diff > 0).sum()), int((diff < 0).sum())
                text += f"<br>Δ {np.nanmedian(diff):+.{digits}f} (↑{same[0]} ↓{same[1]})"
            cells.append(text)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "↑/↓: varsayılana göre artan/azalan pencere sayısı (tutarlılık göstergesi)."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ayar duyarlılığı ölçümü (M2)")
    parser.add_argument("--universe", default="data/universe.json")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="docs/reports/m2_ayar_duyarliligi.md")
    parser.add_argument("--json", default="data/sensitivity.json")
    args = parser.parse_args(argv)
    symbols = tuple(s["base"] for s in json.loads(Path(args.universe).read_text())["symbols"])
    jobs = [(n, m, w, args.root, symbols) for w in range(args.windows) for n, m in VARIANTS.items()]
    print(f"{len(jobs)} koşu, {args.workers} işçi", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(run_one, jobs), 1):
            rows.append(row)
            print(
                f"  {i}/{len(jobs)} {row['ayar']} pencere {row['pencere']}: {row['işlem']} işlem",
                flush=True,
            )
    Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    Path(args.out).write_text(report(rows, args.windows), encoding="utf-8")
    print(f"rapor: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
