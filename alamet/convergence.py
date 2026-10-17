"""
M1d — Başlangıç / ısınma yakınsama testi: bot geçmişi ne kadar oynatmalı?

Sinyal hattı (yöntemler, seçici, gölge kasa, ayrışma takibi) durum makineleri içerir; çıktı
başlangıç barına bağlıdır. Bu araç, değerlendirme penceresi (son `eval_bars`) için farklı ısınma
uzunluklarıyla (`warmups`) panelin kararlarını çok uzun ısınmalı referansla karşılaştırır:
  - sinyal yönü (`types`), hedef sırası (`t_rank`) ve havuzsuz skor uyuşma oranı,
  - tüm hücrelerin penceresonuna dek tamamen eşitlendiği ilk bar (tam yakınsama).

Kullanım:
  python -m alamet.convergence --stage S17 --warmups 0 200 1000 4000 8000 --eval-bars 4000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alamet.binance_data import DEFAULT_ROOT, infer_tf, load_universe_binance
from alamet.daily import daily_from_bars, daily_pack, day_inputs, map_to_bars
from alamet.panel import PanelParams, SignalPanel, build_panel
from alamet.stages import STAGES, stage_params, with_signal_tf
from alamet.universe import SYMBOLS, Universe


@dataclass
class Agreement:
    warmup: int
    types: float
    t_rank: float
    scores: float
    converged_after: int | None


def agreement(panel: SignalPanel, reference: SignalPanel, window: int, warmup: int) -> Agreement:
    """Son `window` bardaki uyuşma oranları; `converged_after`: sonuna dek hep aynı olan ilk bar."""

    def tail(array: np.ndarray) -> np.ndarray:
        return np.asarray(array)[-window:]

    same_type = np.nan_to_num(tail(panel.types)) == np.nan_to_num(tail(reference.types))
    same_rank = tail(panel.divergence["t_rank"]) == tail(reference.divergence["t_rank"])
    a, b = np.nan_to_num(tail(panel.scores)), np.nan_to_num(tail(reference.scores))
    same_score = np.isclose(a, b, atol=1e-9, rtol=1e-9)
    every = same_type.all(axis=1) & same_rank.all(axis=1) & same_score.all(axis=1)
    bad = np.nonzero(~every)[0]
    return Agreement(
        warmup,
        float(same_type.mean()),
        float(same_rank.mean()),
        float(same_score.mean()),
        0 if len(bad) == 0 else (None if bad[-1] == window - 1 else int(bad[-1]) + 1),
    )


def build(universe: Universe, params: PanelParams, start: int, day) -> SignalPanel:
    return build_panel(universe, params, True, start, None, day)


def run(stage: str, warmups: list[int], eval_bars: int, reference: int, root: str):
    tf = infer_tf(Path(root))
    universe = load_universe_binance(Path(root), SYMBOLS, tf=tf)
    daily = daily_from_bars(universe, tf.seconds)
    params = with_signal_tf(stage_params(stage)[0], tf.seconds)
    pack = daily_pack(daily, params.dr_rsi_len, params.dr_look)
    day = day_inputs(map_to_bars(pack, daily.time, universe.time, tf_seconds=tf.seconds))
    end = universe.bars
    ref = build(universe, params, end - eval_bars - reference, day)
    rows = []
    for warmup in warmups:
        panel = build(universe, params, end - eval_bars - warmup, day)
        rows.append(agreement(panel, ref, eval_bars, warmup))
    return rows


def table(stage: str, rows: list[Agreement], reference: int, eval_bars: int) -> str:
    head = (
        f"Basamak {stage} · referans ısınma {reference} bar · değerlendirme son {eval_bars} bar\n\n"
        "| Isınma (bar) | Sinyal yönü % | Hedef sırası % | Skor % | Tam yakınsama (bar) |\n"
        "|---:|---:|---:|---:|---:|\n"
    )
    lines = [
        f"| {r.warmup} | {r.types * 100:.2f} | {r.t_rank * 100:.2f} | {r.scores * 100:.2f} | "
        f"{'yakınsamadı' if r.converged_after is None else r.converged_after} |"
        for r in rows
    ]
    return head + "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DEPV2 ısınma/yakınsama testi (M1d)")
    parser.add_argument("--stage", default="S17", choices=list(STAGES))
    parser.add_argument("--warmups", type=int, nargs="+", default=[0, 200, 1000, 4000, 8000])
    parser.add_argument("--eval-bars", type=int, default=4000)
    parser.add_argument("--reference", type=int, default=30000)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args(argv)
    rows = run(args.stage, args.warmups, args.eval_bars, args.reference, args.root)
    print(table(args.stage, rows, args.reference, args.eval_bars))
    return 0


if __name__ == "__main__":
    sys.exit(main())
