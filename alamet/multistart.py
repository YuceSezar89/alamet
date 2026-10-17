"""
B3 — Çok başlangıçlı backtest: aynı ayarı farklı başlangıç barlarından sabit süreyle koşturur.

Her koşu `warmup` bar önce başlar (gösterge, öğrenme ve havuz durumu ısınsın; canlıda geçmişi
yeniden oynatan bir botun durumuna karşılık gelir). Getiri ve düşüş yalnızca başlangıç barından
sonraki kasa eğrisinden hesaplanır. Sonuç tek sayı değil dağılımdır.

Kullanım:
  python -m alamet.multistart --stages S17 --costs original realistic --horizon-days 180 \
      --step-days 14 --warmup 4000 --out reports/b3_s17.md
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from alamet.backtest import COSTS
from alamet.binance_data import DEFAULT_ROOT, infer_tf, load_universe_binance
from alamet.daily import daily_from_bars, daily_pack, day_inputs, map_to_bars
from alamet.stages import STAGES, stage_params, with_signal_tf
from alamet.system import run_system
from alamet.universe import SYMBOLS

BARS_PER_DAY = 96  # 15 dk varsayılanı; araçlar gerçek TF'yi `SignalTf` ile türetir


@dataclass
class RunResult:
    stage: str
    cost: str
    start_bar: int
    ret: float
    max_dd: float
    trades: int
    idle_days: float


@lru_cache(maxsize=1)
def _data(root: str):
    tf = infer_tf(Path(root))
    universe = load_universe_binance(Path(root), SYMBOLS, tf=tf)
    return universe, daily_from_bars(universe, tf.seconds)


def start_bars(total: int, warmup: int, horizon: int, step: int) -> list[int]:
    return list(range(warmup, total - horizon + 1, step))


def window_metrics(
    result, begin: int, bars_after: int, bars_per_day: int = BARS_PER_DAY
) -> tuple[float, float, int, float]:
    """(getiri %, azami düşüş %, açılan işlem, en uzun işlemsiz süre gün) başlangıç barı `begin`."""
    realised = np.asarray(result.realised, dtype=float)
    curve = realised[begin:]
    base = curve[0] if curve[0] > 0 else np.nan
    ret = (curve[-1] / base - 1.0) * 100.0
    peak = np.maximum.accumulate(curve)
    max_dd = float(np.max((peak - curve) / np.where(peak > 0, peak, np.nan)) * 100.0)
    opens = sorted(t.open_bar for t in result.trades if t.open_bar >= begin)
    edges = [begin, *opens, begin + bars_after]
    idle = max(b - a for a, b in zip(edges[:-1], edges[1:])) / bars_per_day
    return float(ret), max_dd, len(opens), float(idle)


def run_one(job: tuple) -> RunResult:  # pylint: disable=too-many-locals
    stage, cost, begin, warmup, horizon, root = job
    universe, daily = _data(root)
    first = begin - warmup
    sub = universe.slice_from(first).slice_to(warmup + horizon)
    fee_in, fee_out, slip = COSTS[cost]
    panel_p, params = stage_params(stage, fee_in=fee_in, fee_out=fee_out, slippage=slip)
    tf = sub.bar_seconds
    panel_p = with_signal_tf(panel_p, tf)
    pack = daily_pack(daily, panel_p.dr_rsi_len, panel_p.dr_look)
    day = day_inputs(map_to_bars(pack, daily.time, sub.time, tf_seconds=tf))
    result = run_system(sub, panel_p, params, True, start=0, day=day)
    ret, max_dd, trades, idle = window_metrics(result, warmup, horizon, 86400 // tf)
    return RunResult(stage, cost, begin, ret, max_dd, trades, idle)


def summarise(rows: list[RunResult]) -> str:
    head = (
        "| Set | Maliyet | Başlangıç | Medyan getiri % | P10 | P90 | En kötü | Pozitif % | "
        "Medyan maks DD % | Medyan işlem | Medyan en uzun işlemsiz gün |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    keys = sorted({(r.stage, r.cost) for r in rows})
    for stage, cost in keys:
        group = [r for r in rows if r.stage == stage and r.cost == cost]
        ret = np.array([r.ret for r in group])
        lines.append(
            f"| {stage} | {cost} | {len(group)} | {np.median(ret):+.1f} | "
            f"{np.percentile(ret, 10):+.1f} | {np.percentile(ret, 90):+.1f} | {ret.min():+.1f} | "
            f"{(ret > 0).mean() * 100:.0f} | {np.median([r.max_dd for r in group]):.1f} | "
            f"{np.median([r.trades for r in group]):.0f} | "
            f"{np.median([r.idle_days for r in group]):.1f} |"
        )
    return head + "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DEPV2 çok başlangıçlı backtest (B3)")
    parser.add_argument("--stages", nargs="+", default=["S17"])
    parser.add_argument("--costs", nargs="+", default=["original", "realistic"], choices=COSTS)
    parser.add_argument("--horizon-days", type=int, default=180)
    parser.add_argument("--step-days", type=int, default=14)
    parser.add_argument("--warmup", type=int, default=4000)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    for stage in args.stages:
        if stage.upper() not in STAGES:
            parser.error(f"bilinmeyen set: {stage}")
    universe, _ = _data(args.root)
    per_day = 86400 // universe.bar_seconds
    horizon = args.horizon_days * per_day
    starts = start_bars(universe.bars, args.warmup, horizon, args.step_days * per_day)
    jobs = [
        (stage.upper(), cost, begin, args.warmup, horizon, args.root)
        for stage in args.stages
        for cost in args.costs
        for begin in starts
    ]
    print(
        f"{len(starts)} başlangıç × {len(args.stages)} set × {len(args.costs)} maliyet", flush=True
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_one, jobs, chunksize=1))
    text = summarise(rows)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
