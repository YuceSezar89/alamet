"""
M3 — Zamanlama farkı: Piyasa Mumu alarmı (|Z| ≥ 2,5) ile DEPV2 olayları aynı hareketi ne zaman yakalıyor?

Piyasa Mumu tarafı `CONF_RADAR.html` `rdAlarmlar` ile aynı tanımla yeniden üretilir: 96 × 15 dk
pencere hareketi, coinlerin medyanı ve MAD'ı ile Z, en az 50 coin, yeni ayrışma = bir önceki mumda
aynı yönde ayrışmamış. DEPV2 tarafı kayıtlı veritabanından (ayrışma başlangıcı, çift ayrışma, işlem).
Tesadüfi eşleşmeyi görmek için alarmlar ±7 gün kaydırılarak aynı eşleştirme tekrarlanır.
Kâr ölçülmez; yalnızca zaman ve yön lehine fiyat farkı.

  python -m alamet.timing_compare --out docs/reports/m3_zamanlama_farki.md
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from alamet.binance_data import DEFAULT_ROOT

STEP_MS = 900_000
WINDOW = 96
Z_DIV = 2.5
MIN_COINS = 50
ERA_START_MS = int(pd.Timestamp("2026-04-25", tz="UTC").timestamp() * 1000)
BEFORE, AFTER = 96, 192  # eşleşme penceresi: alarmdan 24 sa önce … 48 sa sonra (bar)


def load_grid(root: Path, bases: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    opens, closes = {}, {}
    for base in bases:
        path = root / f"{base}USDT.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path).set_index("open_time")
        frame = frame[frame.index >= ERA_START_MS - WINDOW * STEP_MS]
        opens[base], closes[base] = frame["open"], frame["close"]
    index = pd.RangeIndex(0)
    opens_df, closes_df = pd.DataFrame(opens), pd.DataFrame(closes)
    full = np.arange(opens_df.index.min(), opens_df.index.max() + STEP_MS, STEP_MS)
    del index
    return opens_df.reindex(full), closes_df.reindex(full)


def market_alarms(opens: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """Günlük pencere (96 mum) için yeni ayrışma alarmları: (t, sembol, yön, Z, hareket %)."""
    move = (closes / opens.shift(WINDOW - 1) - 1.0) * 100.0
    med = move.median(axis=1)
    mad = 1.4826 * move.sub(med, axis=0).abs().median(axis=1)
    ok = (move.notna().sum(axis=1) >= MIN_COINS) & (mad > 1e-9)
    z = move.sub(med, axis=0).div(mad.where(ok), axis=0)
    div = pd.DataFrame(
        np.where(z >= Z_DIV, 1, np.where(z <= -Z_DIV, -1, 0)), index=z.index, columns=z.columns
    )
    new = (div != 0) & (div != div.shift(1, fill_value=0))
    rows = []
    for t in new.index[new.any(axis=1)]:
        if t < ERA_START_MS:
            continue
        for sym in new.columns[new.loc[t].to_numpy()]:
            rows.append(
                (int(t), sym, int(div.at[t, sym]), float(z.at[t, sym]), float(move.at[t, sym]))
            )
    return pd.DataFrame(rows, columns=["t", "sym", "dir", "z", "move"])


def depv2_events(db: Path) -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    ev = pd.read_sql_query(
        "SELECT bar_time, kind, symbol, payload FROM events "
        "WHERE kind IN ('DivergenceEvent','DoubleDivergenceEvent')",
        conn,
    )
    ev["p"] = ev.payload.map(json.loads)
    ev["n"] = ev.p.map(lambda p: p["number"])
    starts = ev[ev.kind == "DivergenceEvent"].copy()
    starts["dir"] = starts.p.map(lambda p: p["direction"])
    doubles = ev[ev.kind == "DoubleDivergenceEvent"][["bar_time", "symbol", "n"]].merge(
        starts[["symbol", "n", "dir"]], on=["symbol", "n"]
    )
    trades = pd.read_sql_query("SELECT symbol, side, open_time FROM trades", conn)
    conn.close()
    out = {
        "ayrışma başlangıcı (ilk 2'ye giriş)": starts[["bar_time", "symbol", "dir"]],
        "çift ayrışma": doubles[["bar_time", "symbol", "dir"]],
        "işlem açılışı": trades.rename(columns={"open_time": "bar_time", "side": "dir"})[
            ["bar_time", "symbol", "dir"]
        ],
    }
    for key, frame in out.items():
        frame = frame.rename(columns={"symbol": "sym"})
        frame["t"] = frame.bar_time.astype("int64") * 1000
        out[key] = frame[frame.t >= ERA_START_MS][["t", "sym", "dir"]].reset_index(drop=True)
    return out


def index_events(frame: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    return {key: np.sort(group.t.to_numpy()) for key, group in frame.groupby(["sym", "dir"])}


def first_alarms(alarms: pd.DataFrame, quiet_bars: int = 96) -> pd.DataFrame:
    """Aynı coin ve yönde `quiet_bars` içindeki tekrar alarmları atar (eşik çevresinde gidip gelme)."""
    keep, last = [], {}
    for row in alarms.sort_values("t").itertuples():
        key = (row.sym, row.dir)
        if key in last and row.t - last[key] < quiet_bars * STEP_MS:
            continue
        last[key] = row.t
        keep.append(row.Index)
    return alarms.loc[keep].reset_index(drop=True)


def first_in(times: np.ndarray | None, lo_ms: int, hi_ms: int) -> float:
    """[lo, hi] içindeki ilk olayın zamanı (ms) ya da NaN."""
    if times is None:
        return np.nan
    i = np.searchsorted(times, lo_ms, side="left")
    return float(times[i]) if i < len(times) and times[i] <= hi_ms else np.nan


def forward_stats(alarms: pd.DataFrame, events: dict, closes, shift_ms: int = 0) -> dict:
    """Alarmdan sonra (0…+48 sa) ve önce (−24 sa…−1 bar) aynı coin/yönde ilk olay."""
    post, pre, lags, moves = 0, 0, [], []
    for row in alarms.itertuples():
        times = events.get((row.sym, row.dir))
        centre = row.t + shift_ms
        after = first_in(times, centre, centre + AFTER * STEP_MS)
        before = first_in(times, centre - BEFORE * STEP_MS, centre - STEP_MS)
        pre += not np.isnan(before)
        if not np.isnan(after):
            post += 1
            lags.append((after - centre) / STEP_MS)
            if shift_ms == 0:
                try:
                    moves.append(
                        row.dir
                        * (closes.at[int(after), row.sym] / closes.at[row.t, row.sym] - 1)
                        * 100
                    )
                except KeyError:
                    pass
    n = max(len(alarms), 1)
    return {
        "post": post / n * 100,
        "pre": pre / n * 100,
        "lags": np.array(lags),
        "moves": np.array(moves),
    }


def summarize(name: str, alarms, events, closes) -> str:
    real = forward_stats(alarms, events, closes)
    shifts = [forward_stats(alarms, events, closes, s * 96 * STEP_MS) for s in (-7, 7)]
    base_post = np.mean([b["post"] for b in shifts])
    base_pre = np.mean([b["pre"] for b in shifts])
    lag = real["lags"] * 15 / 60
    mv = real["moves"]
    med_lag = f"{np.median(lag):.1f} sa" if len(lag) else "–"
    med_mv = f"{np.median(mv):+.2f}%" if len(mv) else "–"
    return (
        f"| {name} | {real['post']:.1f}% (tesadüf {base_post:.1f}%) | {med_lag} | {med_mv} | "
        f"{real['pre']:.1f}% (tesadüf {base_pre:.1f}%) |"
    )


def reverse_report(trades: pd.DataFrame, alarms: pd.DataFrame, closes) -> str:
    index = index_events(alarms[["t", "sym", "dir"]])

    def hits(shift_ms: int):
        found = []
        for row in trades.itertuples():
            times = index.get((row.sym, row.dir))
            centre = row.t + shift_ms
            if times is None:
                continue
            lo = np.searchsorted(times, centre - AFTER * STEP_MS, side="left")
            hi = np.searchsorted(times, centre, side="right")
            if hi > lo:
                found.append((row, times[hi - 1], centre))
        return found

    real = hits(0)
    base = np.mean([len(hits(s * 96 * STEP_MS)) for s in (-14, -7, 7, 14)])
    n = len(trades)
    if not real:
        return f"{n} işlemin hiçbirinden önceki 48 saatte aynı coin/yönde ilk-alarm yok."
    leads = np.array([(row.t - t_alarm) / STEP_MS for row, t_alarm, _ in real])
    moves = []
    for row, t_alarm, _ in real:
        try:
            moves.append(
                row.dir * (closes.at[row.t, row.sym] / closes.at[int(t_alarm), row.sym] - 1) * 100
            )
        except KeyError:
            pass
    return (
        f"- {n} işlemin **{len(real)}**'inden (%{len(real) / n * 100:.0f}) önceki 48 saatte aynı coin ve yönde bir ilk-alarm var "
        f"(tesadüf düzeyi: işlem zamanları ±7/14 gün kaydırılınca ortalama %{base / n * 100:.0f}).\n"
        f"- Alarmdan işlem girişine medyan süre: {np.median(leads) * 15 / 60:.1f} sa "
        f"(%25–%75: {np.percentile(leads, 25) * 15 / 60:.1f}–{np.percentile(leads, 75) * 15 / 60:.1f} sa).\n"
        f"- Alarm anından işlem girişine fiyatın yön lehine hareketi: medyan {np.median(moves):+.2f}% "
        f"(%25–%75: {np.percentile(moves, 25):+.2f}…{np.percentile(moves, 75):+.2f}%)."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zamanlama farkı ölçümü (M3)")
    parser.add_argument("--db", default="data/alamet.sqlite")
    parser.add_argument("--universe", default="data/universe.json")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out", default="docs/reports/m3_zamanlama_farki.md")
    args = parser.parse_args(argv)
    bases = [s["base"] for s in json.loads(Path(args.universe).read_text())["symbols"]]
    opens, closes = load_grid(Path(args.root), bases)
    raw_alarms = market_alarms(opens, closes)
    alarms = first_alarms(raw_alarms)
    events = depv2_events(Path(args.db))
    days = (closes.index.max() - ERA_START_MS) / STEP_MS / 96
    lines = [
        "# M3 — Zamanlama farkı: Piyasa Mumu alarmı ve DEPV2",
        "",
        f"168 coin, günlük pencere (96 × 15 dk), {days:.0f} gün (25 Nis 2026'dan beri, güvenilir veri dönemi). "
        f"Ham Piyasa Mumu alarmı {len(raw_alarms)} (eşik çevresinde gidip gelme dahil); aynı coin ve yönde 24 sa içindeki tekrarlar atılınca "
        f"**{len(alarms)}** ilk-alarm (günde {len(alarms) / days:.1f}). "
        "Aynı coin ve yön. Sonrası: alarmdan 0…48 sa içinde ilk DEPV2 olayı; önceki: alarmdan 24 sa öncesi…1 bar. "
        "**Tesadüf**: alarmlar ±7 gün kaydırılarak aynı eşleştirme. Kâr ölçülmedi.",
        "",
        "| DEPV2 olayı | Alarmdan sonra 48 sa içinde olay olan alarm | Medyan gecikme (alarm → olay) | Alarmdan olaya fiyat farkı (yön lehine, medyan) | Alarmdan ÖNCE 24 sa içinde olay olan alarm |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, frame in events.items():
        lines.append(summarize(name, alarms, index_events(frame), closes))
    lines += [
        "",
        "## Ters yön: DEPV2 işlemlerinin öncesinde Piyasa Mumu alarmı var mıydı?",
        "",
        reverse_report(events["işlem açılışı"], alarms, closes),
        "",
    ]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
