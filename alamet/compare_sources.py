"""
M1b — TradingView verisi (data1/data2) ile Binance önbelleği (M1a) karşılaştırması.

Kullanım:
  python -m alamet.compare_sources data1.csv data2.csv [--root data/binance_15m]

Ortak zaman aralığında sembol ve alan bazında uyuşmayan bar sayısını, en büyük göreli farkı ve
hacim oranını yazar. Binance'te eksik olan TradingView barları ayrıca sayılır.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from alamet.binance_data import DEFAULT_ROOT, load_universe_binance
from alamet.universe import SYMBOLS, load_universe
from alamet.verify import compare_column

FIELDS = ("open", "high", "low", "close", "volume")


def _symbol_line(tv, bn, tv_idx, bn_idx, k):
    parts, bad = [], 0
    for field in FIELDS:
        expected = getattr(tv, field)[tv_idx, k]
        actual = getattr(bn, field)[bn_idx, k]
        report = compare_column(field, expected, actual, atol=0.0, rtol=1e-9)
        bad += report.mismatches
        parts.append(f"{field[:1].upper()}:{report.mismatches}")
        if field == "volume":
            ratio = float(np.nanmedian(actual / np.where(expected == 0, np.nan, expected)))
            parts.append(f"hacim_oran={ratio:.6f}")
    return " ".join(parts), bad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TradingView ↔ Binance veri karşılaştırması")
    parser.add_argument("data1")
    parser.add_argument("data2")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args(argv)

    full = load_universe(args.data1, args.data2)
    tv = full.slice_to(full.bars - 1)
    window = (int(tv.time[0]) * 1000, int(tv.time[-1]) * 1000)
    bn = load_universe_binance(Path(args.root), SYMBOLS, *window, allow_gaps=True)

    common, tv_idx, bn_idx = np.intersect1d(tv.time, bn.time, return_indices=True)
    missing = tv.bars - len(common)
    print(f"TV bar: {tv.bars}  Binance ortak bar: {len(common)}  Binance'te olmayan: {missing}")
    total_bad = 0
    for k, symbol in enumerate(SYMBOLS):
        line, bad = _symbol_line(tv, bn, tv_idx, bn_idx, k)
        total_bad += bad
        print(f"  {symbol:<5} {line}")
    ok = total_bad == 0 and missing == 0
    print("SONUÇ:", "TAMAM" if ok else f"FARK VAR ({total_bad})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
