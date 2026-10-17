"""
Pine doğrulama aracı — TradingView "Export chart data" CSV'sini Python modülüyle karşılaştırır.

Kullanım:
  python -m alamet.verify A1 /yol/a1.csv   (A1 … A8, B1, B1B, B2, C1A, C1B, C1C, C2, C3, C4, C5, C6, D1)

CSV'deki OHLC sütunları Python'a girdi olur, Pine'ın plot sütunları beklenen değerdir.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from alamet.devisso import devisso_daily
from alamet.divergence import DivergenceParams, run_divergence
from alamet.entry_quality import EntryQualityParams, entry_flags
from alamet.heikin import bucket_ids, ha_score
from alamet.levels import daily_open, period_ids, period_levels
from alamet.momentum import momentum_signal
from alamet.mtf import mtf_confirm
from alamet.own_divergence import MODE_PREV, MODE_SIGMA, own_gate_series
from alamet.primitives import pine_atr, pine_tr
from alamet.ranking import rank_all
from alamet.score import (
    atr_percent,
    base_score,
    combine_percentile,
    cross_percentile,
    ease_bonus,
    signal_pct,
    target_distance,
    target_distances,
)
from alamet.selector import SelectorParams, run_selector
from alamet.shadow import run_with_shadow
from alamet.symbol import (
    HA_MULTIPLIERS,
    METHOD_NAMES,
    SIGNAL_TF_SECONDS,
    symbol_signals,
)
from alamet.trend import trend_signal
from alamet.volatility import volatility_signal
from alamet.volume import volume_signal


@dataclass(frozen=True)
class ColumnReport:
    name: str
    rows: int
    compared: int
    mismatches: int
    max_abs_diff: float
    first_bad: list[int]

    @property
    def ok(self) -> bool:
        return self.mismatches == 0


def load_tv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def compare_column(
    name: str,
    expected: np.ndarray,
    actual: np.ndarray,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> ColumnReport:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    both_nan = np.isnan(expected) & np.isnan(actual)
    one_nan = np.isnan(expected) ^ np.isnan(actual)
    both = ~np.isnan(expected) & ~np.isnan(actual)
    close = np.zeros(len(expected), dtype=bool)
    close[both] = np.isclose(expected[both], actual[both], atol=atol, rtol=rtol)
    bad = one_nan | (both & ~close)
    diffs = np.abs(expected[both] - actual[both])
    return ColumnReport(
        name=name,
        rows=len(expected),
        compared=int(both.sum() + both_nan.sum()),
        mismatches=int(bad.sum()),
        max_abs_diff=float(diffs.max()) if len(diffs) else 0.0,
        first_bad=[int(i) for i in np.flatnonzero(bad)[:5]],
    )


def _ohlc(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        df["open"].to_numpy(float),
        df["high"].to_numpy(float),
        df["low"].to_numpy(float),
        df["close"].to_numpy(float),
    )


def _exact(name: str, df: pd.DataFrame, column: str, actual: np.ndarray) -> ColumnReport:
    return compare_column(name, df[column].to_numpy(float), actual, atol=0, rtol=0)


def verify_a1(df: pd.DataFrame) -> list[ColumnReport]:
    _, high, low, close = _ohlc(df)
    return [
        compare_column("TR", df["tr"].to_numpy(float), pine_tr(high, low, close)),
        compare_column("ATR", df["atr"].to_numpy(float), pine_atr(high, low, close, 10)),
    ]


def verify_a2(df: pd.DataFrame) -> list[ColumnReport]:
    _, high, low, close = _ohlc(df)
    res = trend_signal(high, low, close)
    return [
        compare_column("TD_Z", df["td_z"].to_numpy(float), res.z),
        _exact("TD_STATE", df, "td_state", res.state),
        _exact("TD_STR", df, "td_str", res.strength),
    ]


def verify_a3(df: pd.DataFrame) -> list[ColumnReport]:
    _, _, _, close = _ohlc(df)
    res = momentum_signal(close)
    return [
        compare_column("MO_Z", df["mo_z"].to_numpy(float), res.z),
        _exact("MO_STATE", df, "mo_state", res.state),
        _exact("MO_STR", df, "mo_str", res.strength),
    ]


def verify_a4(df: pd.DataFrame) -> list[ColumnReport]:
    open_, high, low, close = _ohlc(df)
    res = volatility_signal(open_, high, low, close)
    return [
        compare_column("VA_X", df["va_x"].to_numpy(float), res.x),
        compare_column("VA_RAT", df["va_rat"].to_numpy(float), res.ratio),
        _exact("VA_STATE", df, "va_state", res.state),
        _exact("VA_STR", df, "va_str", res.strength),
    ]


def verify_a5(df: pd.DataFrame) -> list[ColumnReport]:
    _, high, low, close = _ohlc(df)
    res = volume_signal(high, low, close, df["vol"].to_numpy(float))
    return [
        compare_column("HV_RELV", df["hv_relv"].to_numpy(float), res.rel_v),
        compare_column("HV_FLOW", df["hv_flow"].to_numpy(float), res.flow),
        compare_column("HV_SHARE", df["hv_share"].to_numpy(float), res.share),
        compare_column("HV_RAT", df["hv_rat"].to_numpy(float), res.ratio),
        _exact("HV_STATE", df, "hv_state", res.state),
        _exact("HV_STR", df, "hv_str", res.strength),
    ]


def _ha_scores(df: pd.DataFrame) -> list[np.ndarray]:
    open_, high, low, close = _ohlc(df)
    time_s = df["time"].to_numpy(np.int64)
    return [
        ha_score(open_, high, low, close, bucket_ids(time_s, SIGNAL_TF_SECONDS * mult))
        for mult in HA_MULTIPLIERS
    ]


def verify_a6(df: pd.DataFrame) -> list[ColumnReport]:
    scores = _ha_scores(df)
    return [
        compare_column(f"HS_{idx} ({mult}x)", df[f"hs_{idx}"].to_numpy(float), scores[idx - 1])
        for idx, mult in enumerate(HA_MULTIPLIERS, start=1)
    ]


def verify_a7(df: pd.DataFrame) -> list[ColumnReport]:
    res = mtf_confirm(_ha_scores(df))
    return [
        compare_column("FINAL_PCT", df["final_pct"].to_numpy(float), res.final_pct),
        _exact("N_BULL", df, "n_bull", res.n_bull),
        _exact("N_BEAR", df, "n_bear", res.n_bear),
        _exact("TFOK_L", df, "tfok_l", res.ok_long.astype(int)),
        _exact("TFOK_S", df, "tfok_s", res.ok_short.astype(int)),
    ]


GATE_PREFIXES = ("T", "M", "V", "H")


def verify_a8(df: pd.DataFrame) -> list[ColumnReport]:
    open_, high, low, close = _ohlc(df)
    result = symbol_signals(
        open_, high, low, close, df["vol"].to_numpy(float), df["time"].to_numpy(np.int64)
    )
    reports = []
    for prefix, name, gate in zip(GATE_PREFIXES, METHOD_NAMES, result.methods):
        columns = (
            ("TYPE", gate.last_type),
            ("START", gate.last_start),
            ("STR", gate.last_str),
            ("PEND", gate.pend_dir),
            ("SIG", gate.last_sig),
        )
        for suffix, actual in columns:
            column = f"{prefix}_{suffix}"
            reports.append(
                compare_column(
                    f"{column} ({name})",
                    df[column.lower()].to_numpy(float),
                    actual,
                    atol=0,
                    rtol=1e-12,
                )
            )
    return reports


LEVEL_KINDS = (
    ("W", ("WO", "WH", "WL", "WPH", "WPL", "WPC")),
    ("M", ("MO", "MH", "ML", "MPH", "MPL", "MPC")),
    ("Y", ("YO", "YH", "YL", "YPH", "YPL", "YPC")),
)


def verify_b1(df: pd.DataFrame) -> list[ColumnReport]:
    open_, high, low, close = _ohlc(df)
    time_s = df["time"].to_numpy(np.int64)
    reports = []
    for kind, names in LEVEL_KINDS:
        levels = period_levels(open_, high, low, close, period_ids(time_s, kind))
        actual = (
            levels.open,
            levels.high,
            levels.low,
            levels.prev_high,
            levels.prev_low,
            levels.prev_close,
        )
        for name, values in zip(names, actual):
            reports.append(
                compare_column(name, df[name.lower()].to_numpy(float), values, atol=0, rtol=1e-12)
            )
    return reports


def verify_b1b(df: pd.DataFrame) -> list[ColumnReport]:
    open_, _, _, _ = _ohlc(df)
    actual = daily_open(df["time"].to_numpy(np.int64), open_)
    return [compare_column("DOPEN", df["dopen"].to_numpy(float), actual, atol=0, rtol=1e-12)]


def verify_b2(df: pd.DataFrame) -> list[ColumnReport]:
    res = devisso_daily(df["close"].to_numpy(float))
    return [
        compare_column("DR_TOT", df["dr_tot"].to_numpy(float), res.total),
        compare_column("DR_RSI", df["dr_rsi"].to_numpy(float), res.rsi),
        compare_column("DR_K", df["dr_k"].to_numpy(float), res.ratio),
        compare_column("DR_B", df["dr_b"].to_numpy(float), res.prev_ratio),
        compare_column("DR_PR", df["dr_pr"].to_numpy(float), res.percentile),
    ]


def verify_c1a(df: pd.DataFrame) -> list[ColumnReport]:
    open_, high, low, close = _ohlc(df)
    result = symbol_signals(
        open_, high, low, close, df["vol"].to_numpy(float), df["time"].to_numpy(np.int64)
    )
    atr_p = atr_percent(pine_atr(high, low, close, 10), close)
    reports = [compare_column("ATRP", df["atrp"].to_numpy(float), atr_p)]
    for prefix, gate in zip(GATE_PREFIXES, result.methods):
        pct = signal_pct(close, gate.last_start)
        score = base_score(gate.last_type, pct, atr_p)
        reports.append(
            compare_column(f"PCT_{prefix}", df[f"pct_{prefix.lower()}"].to_numpy(float), pct)
        )
        reports.append(
            compare_column(f"SC_{prefix}", df[f"sc_{prefix.lower()}"].to_numpy(float), score)
        )
    return reports


def verify_c1b(df: pd.DataFrame) -> list[ColumnReport]:
    _, high, low, close = _ohlc(df)
    names = [name for _, group in LEVEL_KINDS for name in group]
    levels = np.column_stack([df[name.lower()].to_numpy(float) for name in names])
    atr = pine_atr(high, low, close, 10)
    reports = []
    for tag, up in (("UP", True), ("DN", False)):
        d_near, d_big = target_distances(levels, close, atr, up)
        combined = target_distance(levels, close, atr, up)
        for name, actual in ((f"DN_{tag}", d_near), (f"DB_{tag}", d_big), (f"TD_{tag}", combined)):
            reports.append(compare_column(name, df[name.lower()].to_numpy(float), actual))
    return reports


def verify_c1c(df: pd.DataFrame) -> list[ColumnReport]:
    count = 5
    ratios = np.column_stack([df[f"drb_{i}"].to_numpy(float) for i in range(1, count + 1)])
    owns = np.column_stack([df[f"drpo_{i}"].to_numpy(float) for i in range(1, count + 1)])
    xs = cross_percentile(ratios)
    combined = combine_percentile(owns, xs)
    bonus = ease_bonus(combined)
    reports = []
    for i in range(count):
        for name, actual in (("XS", xs), ("PR", combined), ("BON", bonus)):
            column = f"{name}_{i + 1}"
            reports.append(compare_column(column, df[column.lower()].to_numpy(float), actual[:, i]))
    return reports


def verify_c3(df: pd.DataFrame) -> list[ColumnReport]:
    open_, high, low, close = _ohlc(df)
    result = symbol_signals(
        open_, high, low, close, df["vol"].to_numpy(float), df["time"].to_numpy(np.int64)
    )
    types = np.column_stack([gate.last_type for gate in result.methods])
    pcts = np.column_stack([signal_pct(close, gate.last_start) for gate in result.methods])
    series = own_gate_series(types, pcts)
    ok_p, ref_p, count, cur = series[MODE_PREV]
    ok_s, ref_s, _, _ = series[MODE_SIGMA]
    reports = []
    for i, prefix in enumerate(GATE_PREFIXES):
        columns = (
            ("OKP", ok_p[:, i].astype(float)),
            ("REFP", ref_p[:, i]),
            ("OKS", ok_s[:, i].astype(float)),
            ("REFS", ref_s[:, i]),
            ("N", count[:, i].astype(float)),
            ("CUR", cur[:, i]),
        )
        for suffix, actual in columns:
            column = f"{prefix}_{suffix}"
            reports.append(compare_column(column, df[column.lower()].to_numpy(float), actual))
    return reports


def _rank_reports(df: pd.DataFrame, ties: str) -> list[ColumnReport]:
    count = 5

    def matrix(prefix: str) -> np.ndarray:
        return np.column_stack(
            [df[f"{prefix}_{i}".lower()].to_numpy(float) for i in range(1, count + 1)]
        )

    types, scores, own_ok = matrix("TY"), matrix("SC"), matrix("OK") > 0.5
    base = rank_all(scores, types, own_ok, n_top=2, gap_min=0.0, ties=ties)
    wide = rank_all(scores, types, own_ok, n_top=2, gap_min=0.5, ties=ties)
    reports = []
    for i in range(count):
        for name, actual in (
            (f"RL_{i + 1}", base.r_long[:, i]),
            (f"RS_{i + 1}", base.r_short[:, i]),
            (f"TOP_{i + 1}", base.top[:, i].astype(float)),
            (f"TOPB_{i + 1}", wide.top[:, i].astype(float)),
        ):
            reports.append(compare_column(name, df[name.lower()].to_numpy(float), actual))
    for r in range(count):
        for name, actual in ((f"IL_{r}", base.idx_long[:, r]), (f"IS_{r}", base.idx_short[:, r])):
            reports.append(
                compare_column(
                    f"{name} (eşitlik: {ties})",
                    df[name.lower()].to_numpy(float),
                    actual.astype(float),
                    atol=0,
                    rtol=0,
                )
            )
    reports.append(compare_column("L3", df["l3"].to_numpy(float), base.cut_long))
    reports.append(compare_column("S3", df["s3"].to_numpy(float), base.cut_short))
    return reports


def verify_c2(df: pd.DataFrame) -> list[ColumnReport]:
    candidates = [_rank_reports(df, ties) for ties in ("first", "last", "pine")]
    return min(candidates, key=lambda reports: sum(r.mismatches for r in reports))


def verify_c4(df: pd.DataFrame) -> list[ColumnReport]:
    count = 5

    def matrix(prefix: str, size: int = count) -> np.ndarray:
        return np.column_stack(
            [df[f"{prefix}_{i}".lower()].to_numpy(float) for i in range(1, size + 1)]
        )

    close, types, top = matrix("CL"), matrix("TY"), matrix("TOP") > 0.5
    idx_long = np.column_stack([df[f"il_{r}"].to_numpy(int) for r in range(4)])
    idx_short = np.column_stack([df[f"is_{r}"].to_numpy(int) for r in range(4)])
    base = run_divergence(close, types, top, idx_long, idx_short, track_last=False)
    loose = run_divergence(close, types, top, idx_long, idx_short, DivergenceParams(need_dbl=False))
    outputs = (
        ("CNT", base["count"]),
        ("ACT", base["active"]),
        ("MV", base["cur_move"]),
        ("DBL", base["last_dbl"]),
        ("LCK", base["lock"]),
        ("EL", base["eligible"]),
        ("TR", base["t_rank"]),
        ("ELB", loose["eligible"]),
    )
    reports = []
    for prefix, actual in outputs:
        for i in range(count):
            column = f"{prefix}_{i + 1}"
            reports.append(
                compare_column(
                    column, df[column.lower()].to_numpy(float), actual[:, i].astype(float)
                )
            )
    return reports


C5_CONFIGS = (
    SelectorParams(mode=4, conf_n=2),
    SelectorParams(mode=4, conf_n=3),
    SelectorParams(mode=5, seq_order=(0, 1, 3, 2), seq_n=3, seq_win=24),
    SelectorParams(mode=5, seq_order=(3, 2, 1, 0), seq_n=2, seq_win=6),
    SelectorParams(mode=0),
    SelectorParams(mode=3),
    SelectorParams(mode=4, enabled=(True, False, True, True), conf_n=2),
    SelectorParams(
        mode=5, enabled=(True, True, False, True), seq_order=(0, 1, 2, 3), seq_n=4, seq_win=48
    ),
)


def verify_c5(df: pd.DataFrame) -> list[ColumnReport]:
    open_, high, low, close = _ohlc(df)
    result = symbol_signals(
        open_, high, low, close, df["vol"].to_numpy(float), df["time"].to_numpy(np.int64)
    )

    def stack(field: str) -> np.ndarray:
        return np.stack([getattr(g, field) for g in result.methods], axis=1)[:, None, :]

    inputs = [stack(f) for f in ("last_type", "last_start", "last_str", "pend_dir", "last_sig")]
    reports = []
    for number, params in enumerate(C5_CONFIGS, start=1):
        out = run_selector(*inputs, params, track_last=False)
        for suffix, key in (
            ("TYPE", "sig_type"),
            ("START", "start"),
            ("STR", "strength"),
            ("PEND", "pending"),
            ("M", "method"),
        ):
            column = f"C{number}_{suffix}"
            reports.append(
                compare_column(
                    column, df[column.lower()].to_numpy(float), out[key][:, 0].astype(float)
                )
            )
    return reports


C6_SELECTORS = (
    SelectorParams(mode=6),
    SelectorParams(mode=6, auto_min_n=1, auto_min_net=-5.0, auto_block=False),
)


def verify_c6(df: pd.DataFrame) -> list[ColumnReport]:
    open_, high, low, close = _ohlc(df)
    result = symbol_signals(
        open_, high, low, close, df["vol"].to_numpy(float), df["time"].to_numpy(np.int64)
    )

    def stack(field: str) -> np.ndarray:
        return np.stack([getattr(g, field) for g in result.methods], axis=1)[:, None, :]

    gates = dict(
        zip(
            ("dirs", "starts", "strs", "pends", "sigs"),
            (stack(f) for f in ("last_type", "last_start", "last_str", "pend_dir", "last_sig")),
        )
    )
    atr = pine_atr(high, low, close, 10)
    out = run_with_shadow(gates, close[:, None], atr[:, None], C6_SELECTORS, track_last=False)
    reports = []
    for account in range(5):
        for suffix, key in (
            ("CNT", "count"),
            ("WIN", "wins"),
            ("NET", "net"),
            ("NETD", "net_d"),
            ("SUMR", "sum_r"),
            ("DIR", "side"),
        ):
            column = f"BK{account}_{suffix}"
            reports.append(
                compare_column(
                    column,
                    df[column.lower()].to_numpy(float),
                    out[f"book_{key}"][:, 0, account].astype(float),
                    atol=1e-9,
                    rtol=1e-9,
                )
            )
    for number in (1, 2):
        for suffix, key in (
            ("TYPE", "sig_type"),
            ("START", "start"),
            ("STR", "strength"),
            ("PEND", "pending"),
            ("M", "method"),
            ("AUTO", "auto"),
        ):
            column = f"S{number}_{suffix}"
            reports.append(
                compare_column(
                    column,
                    df[column.lower()].to_numpy(float),
                    out[f"s{number - 1}_{key}"][:, 0].astype(float),
                )
            )
    return reports


D1_SETS = (
    EntryQualityParams(),
    EntryQualityParams(
        min_power=20.0,
        max_ext=1.5,
        cool_bars=30,
        use_weekly_open=True,
        use_monthly_open=True,
        max_daily_open_dist=2.0,
    ),
)


def verify_d1(df: pd.DataFrame) -> list[ColumnReport]:
    open_, high, low, close = _ohlc(df)
    time_s = df["time"].to_numpy(np.int64)
    result = symbol_signals(open_, high, low, close, df["vol"].to_numpy(float), time_s)
    gates = {
        key: np.stack([getattr(g, field) for g in result.methods], axis=1)[:, None, :]
        for key, field in (
            ("dirs", "last_type"),
            ("starts", "last_start"),
            ("strs", "last_str"),
            ("pends", "pend_dir"),
            ("sigs", "last_sig"),
        )
    }
    atr = pine_atr(high, low, close, 10)
    book = run_with_shadow(
        gates, close[:, None], atr[:, None], (SelectorParams(mode=6),), track_last=False
    )
    since = np.arange(len(close)) - book["book_last_loss"][:, 0, 0]
    d_open = daily_open(time_s, open_)
    w_open = period_levels(open_, high, low, close, period_ids(time_s, "W")).open
    m_open = period_levels(open_, high, low, close, period_ids(time_s, "M")).open
    start = result.methods[0].last_start
    reports = []
    for number, params in enumerate(D1_SETS, start=1):
        for suffix, side in (("L", 1), ("S", -1)):
            actual = entry_flags(
                np.full(len(close), side),
                start,
                close,
                atr,
                result.mtf.final_pct,
                d_open,
                w_open,
                m_open,
                since,
                params,
            )
            column = f"Q{number}{suffix}"
            reports.append(
                compare_column(column, df[column.lower()].to_numpy(float), actual.astype(float))
            )
    return reports


MODULES: dict[str, Callable[[pd.DataFrame], list[ColumnReport]]] = {
    "A1": verify_a1,
    "A2": verify_a2,
    "A3": verify_a3,
    "A4": verify_a4,
    "A5": verify_a5,
    "A6": verify_a6,
    "A7": verify_a7,
    "A8": verify_a8,
    "B1": verify_b1,
    "B1B": verify_b1b,
    "B2": verify_b2,
    "C1A": verify_c1a,
    "C1B": verify_c1b,
    "C1C": verify_c1c,
    "C2": verify_c2,
    "C3": verify_c3,
    "C4": verify_c4,
    "C5": verify_c5,
    "C6": verify_c6,
    "D1": verify_d1,
}


def print_report(module: str, df: pd.DataFrame, reports: list[ColumnReport]) -> bool:
    print(f"== {module} · {len(df)} satır")
    all_ok = True
    for r in reports:
        status = "TAMAM" if r.ok else "FARK VAR"
        print(
            f"  {r.name:<10} {status:<9} karşılaştırılan={r.compared} "
            f"uyuşmayan={r.mismatches} en_büyük_fark={r.max_abs_diff:.3e}"
        )
        if not r.ok:
            all_ok = False
            for i in r.first_bad:
                print(f"    satır {i} (time={df['time'].iloc[i] if 'time' in df else '?'})")
    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DEPV2 Pine ↔ Python doğrulama")
    parser.add_argument("module", type=str.upper, choices=sorted(MODULES))
    parser.add_argument("csv_path")
    args = parser.parse_args(argv)
    df = load_tv_csv(args.csv_path)
    return 0 if print_report(args.module, df, MODULES[args.module](df)) else 1


if __name__ == "__main__":
    sys.exit(main())
