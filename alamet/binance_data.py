"""
M1a — Binance Futures tarihsel veri (sinyal TF'si ayarlanabilir, varsayılan 15 dk): indirme,
önbellek, bütünlük raporu, Universe üretimi.

Sembol başına `<kök>/<SEMBOL>USDT.parquet` dosyası tutulur (open_time ms, OHLC, hacim). İndirme
kaldığı yerden devam eder; kapanmamış mum yazılmaz. Veri yolu projenin kanonik
`BinanceClientManager.fetch_klines` fonksiyonudur (paylaşılan Redis oran sınırı + ban koruması).
Sayfa başına 1000 mum (weight 5) kullanılır ve sayfalar arasında bekleme konur; canlı servisler
aynı IP'yi paylaştığı için toplam weight düşük tutulur.

Kullanım:
  python -m alamet.binance_data download --start 2023-07-01 [--tf 5m]
  python -m alamet.binance_data report
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np
import pandas as pd

from alamet.exchange import default_client
from alamet.market import bases, load_snapshot
from alamet.timeframe import SignalTf, from_interval
from alamet.universe import SYMBOLS, Universe

DEFAULT_TF = SignalTf()
BAR_MS = DEFAULT_TF.millis
PAGE = 1000
PACING_SECONDS = 0.5
DEFAULT_ROOT = Path("data/binance_15m")


def default_root(tf: SignalTf) -> Path:
    return Path(f"data/binance_{tf.interval}")


COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "quote_volume"]

PageFetcher = Callable[[str, int], Awaitable[pd.DataFrame]]


def pair(base: str) -> str:
    return f"{base}USDT"


def to_ms(text: str) -> int:
    return int(pd.Timestamp(text).value // 10**6)


def cache_path(root: Path, base: str) -> Path:
    return root / f"{pair(base)}.parquet"


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={"quote_asset_volume": "quote_volume"})
    out = out[COLUMNS].copy()
    out["open_time"] = out["open_time"].astype("int64")
    for column in COLUMNS[1:]:
        out[column] = out[column].astype(float)
    return out


def load_cache(root: Path, base: str) -> pd.DataFrame:
    path = cache_path(root, base)
    if not path.exists():
        return pd.DataFrame({column: [] for column in COLUMNS}).astype({"open_time": "int64"})
    return pd.read_parquet(path)


def save_cache(root: Path, base: str, df: pd.DataFrame) -> None:
    root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path(root, base), index=False)


async def binance_page(symbol: str, start_ms: int, interval: str = "15m") -> pd.DataFrame:
    """Bir sayfa (1000) kapanmış mum; ban koruması ve istek bütçesi `exchange` içindedir."""
    return await default_client().klines(symbol, interval, PAGE, start_ms)


async def fetch_range(
    symbol: str,
    start_ms: int,
    fetch_page: PageFetcher | None = None,
    pacing: float = PACING_SECONDS,
    tf: SignalTf = DEFAULT_TF,
) -> pd.DataFrame:
    fetch_page = fetch_page or partial(binance_page, interval=tf.interval)
    frames: list[pd.DataFrame] = []
    cursor = start_ms
    while True:
        page = await fetch_page(symbol, cursor)
        if page is None or page.empty:
            break
        frames.append(_normalise(page))
        last = int(page["open_time"].iloc[-1])
        if last < cursor or len(page) < PAGE:
            break
        cursor = last + tf.millis
        await asyncio.sleep(pacing)
    if not frames:
        return pd.DataFrame({column: [] for column in COLUMNS}).astype({"open_time": "int64"})
    return pd.concat(frames, ignore_index=True)


async def update_symbol(
    root: Path,
    base: str,
    start_ms: int,
    fetch_page: PageFetcher | None = None,
    pacing: float = PACING_SECONDS,
    tf: SignalTf = DEFAULT_TF,
) -> tuple[int, int]:
    """Önbelleği tamamlar; (eklenen mum sayısı, toplam mum sayısı) döner.

    `start_ms` yalnızca önbellek boşken kullanılır; daha eski veri için dosyayı silip yeniden indir.
    """
    cached = load_cache(root, base)
    cursor = int(cached["open_time"].iloc[-1]) + tf.millis if len(cached) else start_ms
    fresh = await fetch_range(pair(base), cursor, fetch_page, pacing, tf)
    merged = pd.concat([cached, fresh], ignore_index=True)
    merged = merged.drop_duplicates("open_time", keep="last").sort_values("open_time")
    merged = merged.reset_index(drop=True)
    if len(fresh):
        save_cache(root, base, merged)
    return len(merged) - len(cached), len(merged)


@dataclass
class IntegrityReport:
    base: str
    bars: int
    first: int
    last: int
    duplicates: int
    gaps: list[tuple[int, int]]
    bad_ohlc: int

    @property
    def missing_bars(self) -> int:
        return sum(count for _, count in self.gaps)

    @property
    def ok(self) -> bool:
        return self.duplicates == 0 and not self.gaps and self.bad_ohlc == 0


def integrity(base: str, df: pd.DataFrame, tf: SignalTf = DEFAULT_TF) -> IntegrityReport:
    times = df["open_time"].to_numpy(np.int64)
    if len(times) == 0:
        return IntegrityReport(base, 0, 0, 0, 0, [], 0)
    diffs = np.diff(times)
    gap_idx = np.nonzero(diffs != tf.millis)[0]
    gaps = [
        (int(times[i]), int(diffs[i] // tf.millis - 1)) for i in gap_idx if diffs[i] > tf.millis
    ]
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    body = df[["open", "close"]].to_numpy()
    bad = int(
        (
            (high < low) | (high < body.max(axis=1)) | (low > body.min(axis=1)) | (df["volume"] < 0)
        ).sum()
    )
    return IntegrityReport(
        base, len(times), int(times[0]), int(times[-1]), int((diffs == 0).sum()), gaps, bad
    )


def infer_tf(root: Path, symbols: tuple[str, ...] = SYMBOLS) -> SignalTf:
    """Önbelleğin sinyal TF'si: ilk sembolün zaman adımlarının medyanı."""
    times = load_cache(root, symbols[0])["open_time"].to_numpy(np.int64)
    if len(times) < 2:
        raise ValueError(f"{root}: TF çıkarmak için yeterli veri yok")
    return SignalTf(int(np.median(np.diff(times))) // 1000)


def load_universe_binance(
    root: Path = DEFAULT_ROOT,
    symbols: tuple[str, ...] = SYMBOLS,
    start_ms: int | None = None,
    end_ms: int | None = None,
    allow_gaps: bool = False,
    tf: SignalTf = DEFAULT_TF,
    union: bool = False,
) -> Universe:
    """Önbellekten ortak zaman eksenli Universe üretir (zaman: UTC saniye).

    Varsayılan: yalnızca bütün sembollerin ortak zamanları (en yeni sembol başlangıcı belirler).
    `union=True`: bütün sembollerin zamanlarının birleşimi; bir sembolün işlem görmediği barlar
    (henüz listelenmemiş, durdurulmuş) `NaN` olur. Sistem `NaN` fiyatlı sembole işlem açmaz.
    """
    frames = {}
    for base in symbols:
        df = load_cache(root, base)
        if df.empty:
            raise ValueError(f"{base}: önbellek boş ({cache_path(root, base)})")
        frames[base] = df.set_index("open_time")
    common = frames[symbols[0]].index
    for base in symbols[1:]:
        common = (common.union if union else common.intersection)(frames[base].index)
    if start_ms is not None:
        common = common[common >= start_ms]
    if end_ms is not None:
        common = common[common <= end_ms]
    common = common.sort_values()
    if len(common) == 0:
        raise ValueError("Sembollerin ortak zaman aralığı boş")
    steps = np.diff(common.to_numpy(np.int64))
    if not allow_gaps and (steps != tf.millis).any():
        missing = int(((steps[steps > tf.millis] // tf.millis) - 1).sum())
        raise ValueError(f"Ortak eksende {missing} eksik bar var (allow_gaps=True ile geçilebilir)")
    stacked = {
        key: np.column_stack([frames[b][key].reindex(common).to_numpy(float) for b in symbols])
        for key in ("open", "high", "low", "close", "volume")
    }
    return Universe(symbols, (common.to_numpy(np.int64) // 1000), **stacked)


def _format(ms: int) -> str:
    return str(pd.Timestamp(ms, unit="ms"))[:16] if ms else "-"


async def _download(args: argparse.Namespace) -> int:
    tf = from_interval(args.tf)
    root = Path(args.root) if args.root else default_root(tf)
    start_ms = to_ms(args.start)
    page = partial(binance_page, interval=tf.interval)
    for base in args.symbols:
        added, total = await update_symbol(root, base, start_ms, page, PACING_SECONDS, tf)
        print(f"{base:<5} +{added:<7} toplam {total}", flush=True)
    return 0


def _report(args: argparse.Namespace) -> int:
    tf = from_interval(args.tf)
    root = Path(args.root) if args.root else default_root(tf)
    ok = True
    for base in args.symbols:
        rep = integrity(base, load_cache(root, base), tf)
        status = "TAMAM" if rep.ok else "SORUN"
        ok &= rep.ok
        print(
            f"{base:<5} {status:<6} bar={rep.bars:<7} {_format(rep.first)} → {_format(rep.last)} "
            f"yinelenen={rep.duplicates} eksik={rep.missing_bars} bozuk_ohlc={rep.bad_ohlc}"
        )
        for when, count in rep.gaps[:5]:
            print(f"      boşluk: {_format(when)} sonrası {count} bar")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Binance tarihsel veri (M1a)")
    parser.add_argument("command", choices=["download", "report"])
    parser.add_argument("--start", default="2023-07-01")
    parser.add_argument("--tf", default=DEFAULT_TF.interval, help="sinyal TF'si: 1m, 5m, 15m, ...")
    parser.add_argument("--root", default="", help="varsayılan: data/binance_<tf>")
    parser.add_argument("--symbols", nargs="+", default=list(SYMBOLS))
    parser.add_argument(
        "--universe", default="", help="alamet.market ile kaydedilmiş evren dosyası"
    )
    args = parser.parse_args(argv)
    if args.universe:
        args.symbols = list(bases(load_snapshot(Path(args.universe))))
    if args.command == "download":
        return asyncio.run(_download(args))
    return _report(args)


if __name__ == "__main__":
    sys.exit(main())
