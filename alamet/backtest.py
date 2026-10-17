"""
B1 — Tam geçmiş backtest koşucusu: Binance 15 dk önbelleği → panel → motor, ayar seti × maliyet.

Kullanım:
  python -m alamet.backtest --stages S0 S1 S12 --costs original realistic --out reports/b1.md

Günlük paket 15 dk veriden üretilir ("history" eşlemesi: her bar yalnızca kapanmış günü görür).
Panel her ayar seti için bir kez hesaplanır, maliyet senaryoları aynı panel üzerinde koşar.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alamet.binance_data import DEFAULT_ROOT, infer_tf, load_universe_binance
from alamet.daily import daily_from_bars, daily_pack, day_inputs, map_to_bars
from alamet.engine import Trade
from alamet.panel import build_panel
from alamet.selector import MODE_AUTO
from alamet.stages import STAGES, stage_params, with_signal_tf
from alamet.system import run_system
from alamet.universe import SYMBOLS

COSTS = {
    "zero": (0.0, 0.0, 0.0),
    "original": (0.02, 0.02, 0.0),
    "realistic": (0.05, 0.05, 0.02),
    "bad": (0.05, 0.05, 0.05),
}


@dataclass
class Metrics:  # pylint: disable=too-many-instance-attributes
    stage: str
    cost: str
    trades: int
    start_cap: float
    final: float
    max_dd: float
    win_rate: float
    profit_factor: float
    avg_r: float
    long_share: float
    top10_share: float
    years: float

    @property
    def ret(self) -> float:
        return (self.final / self.start_cap - 1.0) * 100.0


def summarise(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    stage: str, cost: str, trades: list[Trade], result, bars: int, cap: float,
    bars_per_year: int = 96 * 365,
) -> Metrics:  # fmt: skip
    done = [t for t in trades if t.close_bar >= 0]
    usd = np.array([t.usd for t in done], dtype=float)
    wins, losses = usd[usd > 0].sum(), -usd[usd < 0].sum()
    ordered = np.sort(usd)[::-1]
    gain = usd.sum()
    return Metrics(
        stage,
        cost,
        len(done),
        cap,
        float(result.realised[-1]),
        float(result.max_dd[-1]),
        float((usd > 0).mean() * 100.0) if len(usd) else 0.0,
        float(wins / losses) if losses > 0 else float("inf"),
        float(np.nanmean([t.r for t in done])) if done else 0.0,
        float(np.mean([t.side == 1 for t in done]) * 100.0) if done else 0.0,
        float(ordered[:10].sum() / gain * 100.0) if gain > 0 else float("nan"),
        bars / bars_per_year,
    )


def run_stage(args: tuple[str, list[str], str]) -> list[Metrics]:  # pylint: disable=too-many-locals
    stage, costs, root = args
    tf = infer_tf(Path(root))
    universe = load_universe_binance(Path(root), SYMBOLS, tf=tf)
    daily = daily_from_bars(universe, tf.seconds)
    base_panel_p = with_signal_tf(stage_params(stage)[0], tf.seconds)
    pack = daily_pack(daily, base_panel_p.dr_rsi_len, base_panel_p.dr_look)
    day = day_inputs(map_to_bars(pack, daily.time, universe.time, tf_seconds=tf.seconds))
    shared = (
        None
        if base_panel_p.selector.mode == MODE_AUTO
        else build_panel(universe, base_panel_p, True, 0, None, day)
    )
    out = []
    for cost in costs:
        fee_in, fee_out, slip = COSTS[cost]
        panel_p, params = stage_params(stage, fee_in=fee_in, fee_out=fee_out, slippage=slip)
        panel_p = with_signal_tf(panel_p, tf.seconds)
        panel = shared or build_panel(universe, panel_p, True, 0, None, day)
        result = run_system(universe, panel_p, params, True, panel=panel, start=0, day=day)
        out.append(
            summarise(
                stage, cost, result.trades, result, universe.bars, params.start_cap,
                tf.bars_per_year,
            )
        )  # fmt: skip
    return out


def table(rows: list[Metrics]) -> str:
    head = (
        "| Set | Maliyet | İşlem | Getiri % | Maks DD % | Kazanma % | PF | Ort R | Long % | "
        "Top10 pay % |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [
        f"| {m.stage} | {m.cost} | {m.trades} | {m.ret:+.1f} | {m.max_dd:.1f} | {m.win_rate:.1f} | "
        f"{m.profit_factor:.2f} | {m.avg_r:+.3f} | {m.long_share:.0f} | {m.top10_share:.0f} |"
        for m in rows
    ]
    return head + "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DEPV2 tam geçmiş backtest (B1)")
    parser.add_argument("--stages", nargs="+", default=list(STAGES))
    parser.add_argument("--costs", nargs="+", default=["original", "realistic"], choices=COSTS)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    jobs = [(stage.upper(), args.costs, args.root) for stage in args.stages]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = [m for chunk in pool.map(run_stage, jobs) for m in chunk]
    text = table(results)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
