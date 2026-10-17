"""
Bütün-sistem doğrulaması: orijinal DEPV2 debug kopyasının CSV'si ↔ Python (panel + motor).

Kullanım:
  python -m alamet.verify_system SIG  debug.csv data1.csv data2.csv
  python -m alamet.verify_system S0   debug.csv data1.csv data2.csv     (S0 … S10, bkz. stages.py)
  python -m alamet.verify_system DAY   debug.csv data1.csv data2.csv --daily datad1.csv datad2.csv
  (SIGT, SIGD, S8-S10 de --daily ister)
  python -m alamet.verify_system S12 debug.csv --binance data/binance_15m  (günlük 15 dk'dan)
  python -m alamet.verify_system AUTO debug_all.csv --binance data/binance_15m  (DEBUG_ALL)

Debug CSV `pine/debug/DEBUG_<ad>.pine` ile üretilir. Python girdisi yalnızca
data1/data2 (20 sembolün 15 dk OHLCV'si) ve gerekirse günlük OHLC'dir.

Hesap penceresi: orijinal `calc_bars_count = 4000` kullanır ve bütün hattı (sembol sinyalleri,
seçici, gölge kasa, sıralama, ayrışma, motor) CSV'nin SON barından 3.999 bar önce başlatır
(canlı bar dahil 4000 bar); grafik daha az bar yüklese bile. CSV'de yalnızca bu pencerenin
satırları karşılaştırılır. `--state-window N` veri sonundan N bar önceyi elle seçer.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alamet.binance_data import load_universe_binance
from alamet.daily import (
    DailyUniverse,
    DayInputs,
    daily_from_bars,
    daily_pack,
    day_inputs,
    load_daily,
    map_to_bars,
    window_daily,
)
from alamet.panel import PanelParams, build_panel
from alamet.score import cross_percentile
from alamet.stages import SIGNAL_STAGES, STAGES, panel_params, stage_params, with_signal_tf
from alamet.system import run_system
from alamet.timeframe import SignalTf
from alamet.universe import SYMBOLS, Universe, load_universe
from alamet.verify import ColumnReport, compare_column, load_tv_csv

CALC_BARS = 4000
DAY_STAGE = "DAY"
DAILY_FROM = 8
SIGNAL_NEEDS_DAILY = {"SIGT", "SIGD", DAY_STAGE}


def needs_daily(stage: str) -> bool:
    """S8 ve sonrası (hedef/Devisso puanları) ile günlük paket gerektiren sinyal basamakları."""
    if stage in SIGNAL_NEEDS_DAILY:
        return True
    return stage.startswith("S") and stage[1:].isdigit() and int(stage[1:]) >= DAILY_FROM


@dataclass
class Context:
    universe: Universe
    df: pd.DataFrame
    rows: np.ndarray
    start: int
    confirm_last: bool
    day: DayInputs | None = None
    mapped: np.ndarray | None = None

    def pick(self, matrix: np.ndarray) -> np.ndarray:
        return matrix[self.rows - self.start]


def select_block(df: pd.DataFrame, block: int) -> pd.DataFrame:
    """Birden çok göstergenin tek dışa aktarımında tekrarlanan sütunlardan `block`. bloğu seçer.

    Yalnızca bir göstergede geçen sütunlar yalnızca 0. blokta bulunur (olmayan sütun atlanır).
    """
    base = ("time", "open", "high", "low", "close")
    names = [c for c in df.columns if c not in base and "." not in c]
    chosen = {}
    for name in names:
        source = name if block == 0 else f"{name}.{block}"
        if source in df.columns:
            chosen[source] = name
    return df[[*base, *chosen]].rename(columns=chosen)


def _rows(universe: Universe, df: pd.DataFrame) -> np.ndarray:
    rows = pd.Index(universe.time).get_indexer(df["time"].to_numpy(np.int64))
    if (rows < 0).any():
        raise SystemExit("Debug CSV'de veri dosyalarında olmayan zaman var")
    return rows


def _reference_column(stage: str, columns=()) -> str:
    if "ks0" not in columns and "stg" in columns:
        return "stg"
    if stage == DAY_STAGE:
        return "xs01"
    return "ntop" if stage in SIGNAL_STAGES else "ks0"


def _compare(ctx: Context, names: list[str], matrix: np.ndarray, **tolerance) -> list[ColumnReport]:
    picked = ctx.pick(matrix)
    return [
        compare_column(name, ctx.df[name.lower()], picked[:, column], **tolerance)
        for column, name in enumerate(names)
    ]


def _per_symbol(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{k + 1:02d}" for k in range(count)]


def _signal_reports(ctx: Context, stage: str) -> list[ColumnReport]:
    _, selector, extra = SIGNAL_STAGES[stage]
    params = with_signal_tf(panel_params(selector=selector, **extra), ctx.universe.bar_seconds)
    panel = build_panel(ctx.universe, params, ctx.confirm_last, ctx.start, day=ctx.day)
    div = panel.divergence
    n = ctx.universe.n
    reports = _compare(ctx, _per_symbol("TY", n), panel.types)
    reports += _compare(ctx, _per_symbol("SC", n), panel.scores)
    reports += _compare(ctx, _per_symbol("TR", n), div["t_rank"].astype(float))
    for name, series in (
        ("NTOP", panel.rank.top.sum(axis=1)),
        ("NEL", div["eligible"].sum(axis=1)),
        ("NOK", panel.own_ok.sum(axis=1)),
    ):
        reports.append(compare_column(name, ctx.df[name.lower()], ctx.pick(series).astype(float)))
    return reports


def _trade_reports(ctx: Context, stage: str) -> tuple[list[ColumnReport], object]:
    panel_p, trade_p = stage_params(stage)
    panel_p = with_signal_tf(panel_p, ctx.universe.bar_seconds)
    result = run_system(
        ctx.universe, panel_p, trade_p, ctx.confirm_last, start=ctx.start, day=ctx.day
    )
    n = ctx.universe.n
    tight = {"atol": 1e-6, "rtol": 1e-9}
    reports = _compare(ctx, _per_symbol("SD", n), result.sides, **tight)
    reports += _compare(ctx, _per_symbol("EN", n), result.entries, **tight)
    reports += _compare(ctx, _per_symbol("NO", n), result.notionals, **tight)
    for name, series in (
        ("KS0", result.realised),
        ("KS1", result.peak),
        ("KS2", result.max_dd),
        ("CL", result.closed_count.astype(float)),
    ):
        if name.lower() in ctx.df:
            reports.append(compare_column(name, ctx.df[name.lower()], ctx.pick(series), **tight))
    return reports, result


def _day_reports(ctx: Context) -> list[ColumnReport]:
    mapped, reports = ctx.mapped, []
    for k in range(3):
        names = [f"MD{k + 1:02d}_{c:02d}" for c in range(20)]
        picked = mapped[ctx.rows, k, :]
        reports += [
            compare_column(name, ctx.df[name.lower()], picked[:, c], atol=1e-9, rtol=1e-9)
            for c, name in enumerate(names)
        ]
    cross = cross_percentile(mapped[:, :, 18])
    for k in range(3):
        name = f"XS{k + 1:02d}"
        reports.append(compare_column(name, ctx.df[name.lower()], cross[ctx.rows, k]))
    return reports


def summarise(
    df: pd.DataFrame, reports: list[ColumnReport]
) -> bool:  # pylint: disable=too-many-locals
    groups: dict[str, list[ColumnReport]] = {}
    for report in reports:
        key = report.name.rstrip("0123456789") if report.name[-2:].isdigit() else report.name
        key = key.split("_")[0] if key.startswith("MD") else key
        groups.setdefault(key, []).append(report)
    ok = True
    for key, items in groups.items():
        bad = [r for r in items if not r.ok]
        total = sum(r.mismatches for r in items)
        status = "TAMAM" if not bad else "FARK VAR"
        print(f"  {key:<5} {status:<9} sütun={len(items)} uyuşmayan_toplam={total}")
        if bad:
            ok = False
            for r in bad[:4]:
                symbol = ""
                digits = "".join(ch for ch in r.name.split("_")[0] if ch.isdigit())
                if digits and 1 <= int(digits) <= len(SYMBOLS):
                    symbol = SYMBOLS[int(digits) - 1]
                first = r.first_bad[0] if r.first_bad else -1
                when = df["time"].iloc[first] if first >= 0 else "?"
                print(
                    f"    {r.name} {symbol}: {r.mismatches} satır, ilk satır {first} "
                    f"(time={when}), en büyük fark {r.max_abs_diff:.3e}"
                )
    return ok


def _prepare(args: argparse.Namespace) -> Context:
    df = select_block(load_tv_csv(args.debug_csv), args.block)
    csv_last = int(df["time"].iloc[-1])
    tf = SignalTf(int(np.median(np.diff(df["time"].to_numpy(np.int64)))))
    if args.binance:
        universe = load_universe_binance(Path(args.binance), SYMBOLS, None, csv_last * 1000, tf=tf)
    else:
        universe = load_universe(args.data1, args.data2)
    confirm_last = False
    if csv_last > int(universe.time[-1]):
        universe = universe.slice_to(universe.bars - 1)
        df = df[df["time"] <= universe.time[-1]].reset_index(drop=True)
        confirm_last = True
        print(
            f"  not: debug CSV veriden yeni; canlı veri satırı atıldı, son bar {universe.time[-1]}"
        )
    df = df.iloc[_first_valid(df, args.stage) :].reset_index(drop=True)
    rows = _rows(universe, df)
    if args.state_window:
        start = universe.bars - args.state_window
    else:
        start = int((csv_last - (CALC_BARS - 1) * tf.seconds - universe.time[0]) // tf.seconds)
    start = min(max(start, 0), int(rows[0]))
    ctx = Context(universe, df, rows, start, confirm_last)
    if needs_daily(args.stage):
        if not args.daily and not args.binance:
            raise SystemExit(f"{args.stage} için --daily datad1.csv datad2.csv gerekli")
        source = load_daily(*args.daily) if args.daily else daily_from_bars(universe, tf.seconds)
        daily: DailyUniverse = window_daily(
            source, universe.time[start], csv_last, CALC_BARS, tf.seconds
        )
        params = stage_params(args.stage)[0] if args.stage in STAGES else PanelParams()
        pack = daily_pack(daily, params.dr_rsi_len, params.dr_look)
        ctx.mapped = map_to_bars(pack, daily.time, universe.time, tf_seconds=tf.seconds)
        ctx.day = day_inputs(ctx.mapped)
    return ctx


def _first_valid(df: pd.DataFrame, stage: str) -> int:
    column = _reference_column(stage, df.columns)
    valid = df[column].notna().to_numpy()
    if not valid.any():
        raise SystemExit(f"{column} sütunu tamamen boş")
    return int(np.argmax(valid))


AUTO_STAGE = "AUTO"
STAGE_COLUMN = re.compile(r"stg(\.\d+)?")


def stage_blocks(csv_path: str) -> list[tuple[int, str]]:
    """DEBUG_ALL dışa aktarımında her bloğun basamak adı (`STG` sütunu basamak indeksidir)."""
    raw = load_tv_csv(csv_path)
    names = [c for c in raw.columns if STAGE_COLUMN.fullmatch(c)]
    if not names:
        raise SystemExit("CSV'de STG sütunu yok (DEBUG_ALL.pine gerekli)")
    order = list(STAGES)
    return [(b, order[int(raw[c].dropna().iloc[0])]) for b, c in enumerate(names)]


def _auto(args: argparse.Namespace) -> int:
    ok = True
    for block, stage in stage_blocks(args.debug_csv):
        sub = argparse.Namespace(**{**vars(args), "stage": stage, "block": block})
        ctx = _prepare(sub)
        reports, result = _trade_reports(ctx, stage)
        print(
            f"== {stage} (blok {block}) · {len(ctx.df)} satır, Python {len(result.trades)} işlem, "
            f"son kasa {result.realised[-1]:.2f}"
        )
        ok &= summarise(ctx.df, reports)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DEPV2 bütün-sistem doğrulaması")
    parser.add_argument(
        "stage", type=str.upper, choices=[*SIGNAL_STAGES, *STAGES, DAY_STAGE, AUTO_STAGE]
    )
    parser.add_argument("debug_csv")
    parser.add_argument("data1", nargs="?")
    parser.add_argument("data2", nargs="?")
    parser.add_argument(
        "--binance", metavar="KÖK", help="OHLCV'yi data1/data2 yerine Binance önbelleğinden al"
    )
    parser.add_argument("--daily", nargs=2, metavar=("DAILY1", "DAILY2"))
    parser.add_argument("--state-window", type=int, default=0)
    parser.add_argument("--block", type=int, default=0, help="çok göstergeli CSV'de blok no (0..)")
    args = parser.parse_args(argv)
    if not args.binance and not (args.data1 and args.data2):
        parser.error("data1 data2 ya da --binance KÖK gerekli")
    if args.stage == AUTO_STAGE:
        return _auto(args)
    ctx = _prepare(args)
    print(
        f"== {args.stage} · {len(ctx.df)} satır (hat başı bar {ctx.start}, CSV başı "
        f"{ctx.rows[0]}, veri {ctx.universe.bars} bar, {ctx.universe.n} sembol)"
    )
    if args.stage == DAY_STAGE:
        reports = _day_reports(ctx)
    elif args.stage in SIGNAL_STAGES:
        reports = _signal_reports(ctx, args.stage)
    else:
        reports, result = _trade_reports(ctx, args.stage)
        print(f"  Python: {len(result.trades)} işlem, son kasa {result.realised[-1]:.2f}")
    return 0 if summarise(ctx.df, reports) else 1


if __name__ == "__main__":
    sys.exit(main())
