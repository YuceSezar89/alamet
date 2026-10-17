"""
İstatistik doğrulaması: `DEBUG_STATS.pine` dışa aktarımı ↔ `stats_vector`.

Kullanım:
  python -m alamet.verify_stats debug_stats.csv --binance data/binance_15m

Her blok bir (basamak `STG`, dilim `SLC`) çiftidir; dilim 62 değerlik bir parçadır. Python aynı
basamağı koşturup vektörün o dilimini TradingView'in V00…V61 sütunlarıyla bar bar karşılaştırır.
"""

# pylint: disable=too-many-locals
from __future__ import annotations

import argparse
import re
import sys

import numpy as np

from alamet.stages import STAGES, stage_params, with_signal_tf
from alamet.stats import vector_names
from alamet.system import run_system
from alamet.verify import load_tv_csv
from alamet.verify_system import _prepare

PER_SLICE = 62


def blocks(csv_path: str) -> list[tuple[int, str, int]]:
    raw = load_tv_csv(csv_path)
    names = [c for c in raw.columns if re.fullmatch(r"stg(\.\d+)?", c)]
    order = list(STAGES)
    out = []
    for block, column in enumerate(names):
        suffix = column[3:]
        stage = order[int(raw[column].dropna().iloc[0])]
        out.append((block, stage, int(raw["slc" + suffix].dropna().iloc[0])))
    return out


def compare(
    ctx, vectors: np.ndarray, slice_no: int, log_size: int = 8
) -> list[tuple[str, int, int]]:
    """(değer adı, uyuşmayan bar sayısı, ilk bozuk satır); yalnızca uyuşmayanlar."""
    names = vector_names(log_size)
    picked = vectors[ctx.rows - ctx.start]
    bad = []
    for i in range(PER_SLICE):
        column = slice_no * PER_SLICE + i
        if column >= vectors.shape[1]:
            break
        tv = ctx.df[f"v{i:02d}"].to_numpy(float)
        py = picked[:, column]
        ok = np.isclose(tv, py, atol=1e-9, rtol=1e-9) | (np.isnan(tv) & np.isnan(py))
        if not ok.all():
            bad.append((names[column], int((~ok).sum()), int(np.argmax(~ok))))
    return bad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DEPV2 istatistik doğrulaması")
    parser.add_argument("csv")
    parser.add_argument("--binance", default="data/binance_15m")
    args = parser.parse_args(argv)
    cache: dict[tuple[str, int], np.ndarray] = {}
    ok = True
    for block, stage, slice_no in blocks(args.csv):
        ns = argparse.Namespace(
            stage=stage, debug_csv=args.csv, data1=None, data2=None, binance=args.binance,
            daily=None, state_window=0, block=block,
        )  # fmt: skip
        ctx = _prepare(ns)
        key = (stage, ctx.start)
        if key not in cache:
            panel_p, trade_p = stage_params(stage)
            panel_p = with_signal_tf(panel_p, ctx.universe.bar_seconds)
            result = run_system(
                ctx.universe, panel_p, trade_p, ctx.confirm_last, start=ctx.start, day=ctx.day,
                collect_stats=True,
            )  # fmt: skip
            cache[key] = result.stats
        bad = compare(ctx, cache[key], slice_no, stage_params(stage)[1].log_size)
        print(
            f"== {stage} dilim {slice_no} (blok {block}) · {len(ctx.df)} satır · "
            f"{'TAMAM' if not bad else 'FARK VAR'}"
        )
        for name, count, first in bad[:8]:
            print(f"    {name}: {count} bar, ilk satır {first}")
        ok &= not bad
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
