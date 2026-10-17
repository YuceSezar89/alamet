"""
Toplu geçmiş veri: Binance'in `data.binance.vision` arşivinden aylık ve günlük mum dosyaları.

REST ile (1000 mum/sayfa, ağırlık sınırlı) yüzlerce sembolün yıllık geçmişi saatler sürer; arşiv
dosyaları ağırlık harcamaz ve çok daha hızlıdır. Her zip, yanındaki `.CHECKSUM` dosyasıyla (sha256)
doğrulanır. Aylık dosyalar tamamlanmış aylar, günlük dosyalar içinde bulunulan ay içindir; arşivin
yetişmediği son birkaç saat mevcut REST yoluyla (`binance_data.update_symbol`) tamamlanır.

Bir sembol için arşivde olmayan aylar (sözleşme henüz açılmamıştı) sessizce atlanır. Veri
`binance_data` önbelleğiyle aynı biçimde yazılır; aynı ay REST ile ve arşivle karşılaştırıldığında
fark sıfırdır (BTCUSDT 15 dk, Ocak 2024: 2.976 mum).

Kullanım:
  python -m alamet.bulk_data --universe data/universe.json --start 2023-07-01 --tf 15m
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import sys
import zipfile
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from alamet.binance_data import (
    COLUMNS,
    default_root,
    load_cache,
    save_cache,
    to_ms,
    update_symbol,
)
from alamet.market import bases, load_snapshot
from alamet.timeframe import SignalTf, from_interval
from alamet.universe import SYMBOLS

ARCHIVE = "https://data.binance.vision/data/futures/um"
MICROSECOND_THRESHOLD = 10**14
WORKERS = 8
RETRIES = 3

BytesFetcher = Callable[[str], Awaitable[bytes | None]]


def archive_url(symbol: str, interval: str, year: int, month: int, day: int | None = None) -> str:
    """Aylık (`day=None`) ya da günlük mum dosyasının adresi."""
    if day is None:
        return f"{ARCHIVE}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
    stem = f"{symbol}-{interval}-{year}-{month:02d}-{day:02d}"
    return f"{ARCHIVE}/daily/klines/{symbol}/{interval}/{stem}.zip"


def parse_archive(content: bytes, checksum: str | None = None) -> pd.DataFrame:
    """Zip içeriğini önbellek biçimine çevirir. `checksum` verilirse sha256 uyuşmalıdır."""
    if checksum is not None:
        digest = hashlib.sha256(content).hexdigest()
        if digest != checksum.split()[0].lower():
            raise ValueError("Arşiv sağlama toplamı uyuşmuyor")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        name = next(item for item in archive.namelist() if item.endswith(".csv"))
        with archive.open(name) as handle:
            raw = pd.read_csv(handle, header=None)
    if isinstance(raw.iloc[0, 0], str):
        raw = raw.iloc[1:]
    frame = pd.DataFrame(
        {
            "open_time": raw.iloc[:, 0].astype("int64"),
            "open": raw.iloc[:, 1].astype(float),
            "high": raw.iloc[:, 2].astype(float),
            "low": raw.iloc[:, 3].astype(float),
            "close": raw.iloc[:, 4].astype(float),
            "volume": raw.iloc[:, 5].astype(float),
            "quote_volume": raw.iloc[:, 7].astype(float),
        }
    ).reset_index(drop=True)
    micro = frame["open_time"] > MICROSECOND_THRESHOLD
    frame.loc[micro, "open_time"] = frame.loc[micro, "open_time"] // 1000
    return frame[COLUMNS]


def plan_files(
    start_ms: int, now_ms: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    """İndirilecek dosyalar: tamamlanmış aylar (yıl, ay) ve içinde bulunulan ayın dünden önceki
    günleri (yıl, ay, gün). Bugünün dosyası henüz yayımlanmamıştır; kuyruğu REST tamamlar."""
    now = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
    cursor = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    months: list[tuple[int, int]] = []
    year, month = cursor.year, cursor.month
    while (year, month) < (now.year, now.month):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    days: list[tuple[int, int, int]] = []
    first = max(cursor, datetime(now.year, now.month, 1, tzinfo=timezone.utc))
    day = datetime(first.year, first.month, first.day, tzinfo=timezone.utc)
    today = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    while day < today:
        days.append((day.year, day.month, day.day))
        day += timedelta(days=1)
    return months, days


async def fetch_archive(fetch: BytesFetcher, url: str) -> pd.DataFrame | None:
    """Tek dosyayı indirir ve doğrular; dosya yoksa (404) `None`."""
    content = await fetch(url)
    if content is None:
        return None
    checksum = await fetch(url + ".CHECKSUM")
    return parse_archive(content, checksum.decode().strip() if checksum else None)


async def bulk_symbol(
    root: Path, base: str, start_ms: int, now_ms: int, fetch: BytesFetcher, tf: SignalTf
) -> tuple[int, int]:
    """Sembolün eksik geçmişini arşivden tamamlar; (eklenen mum, toplam mum) döner."""
    symbol = f"{base}USDT"
    cached = load_cache(root, base)
    begin = int(cached["open_time"].iloc[-1]) + tf.millis if len(cached) else start_ms
    months, days = plan_files(begin, now_ms)
    urls = [archive_url(symbol, tf.interval, y, m) for y, m in months]
    urls += [archive_url(symbol, tf.interval, y, m, d) for y, m, d in days]
    frames = [await fetch_archive(fetch, url) for url in urls]
    found = [frame for frame in frames if frame is not None and len(frame)]
    if not found:
        return 0, len(cached)
    merged = pd.concat([cached, *found], ignore_index=True)
    merged = merged.drop_duplicates("open_time", keep="last").sort_values("open_time")
    merged = merged.reset_index(drop=True)
    save_cache(root, base, merged)
    return len(merged) - len(cached), len(merged)


async def http_fetch(url: str) -> bytes | None:
    import aiohttp  # pylint: disable=import-outside-toplevel

    for attempt in range(RETRIES):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status == 404:
                        return None
                    response.raise_for_status()
                    return await response.read()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt == RETRIES - 1:
                raise
            await asyncio.sleep(2 * (attempt + 1))
    return None


async def bulk_download(
    root: Path,
    symbols: tuple[str, ...],
    start_ms: int,
    tf: SignalTf,
    fetch: BytesFetcher = http_fetch,
    workers: int = WORKERS,
    now_ms: int | None = None,
    tail: bool = True,
) -> dict[str, tuple[int, int]]:
    """Sembolleri paralel doldurur, ardından REST ile kuyruğu tamamlar. {sembol: (eklenen, toplam)}."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000) if now_ms is None else now_ms
    gate = asyncio.Semaphore(workers)
    result: dict[str, tuple[int, int]] = {}

    async def one(base: str) -> None:
        async with gate:
            added, total = await bulk_symbol(root, base, start_ms, now_ms, fetch, tf)
            if tail:
                more, total = await update_symbol(root, base, start_ms, tf=tf)
                added += more
            result[base] = (added, total)
            print(f"{base:<8} +{added:<8} toplam {total}", flush=True)

    await asyncio.gather(*(one(base) for base in symbols))
    return {base: result[base] for base in symbols}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Binance toplu geçmiş veri (data.binance.vision)")
    parser.add_argument("--tf", default="15m")
    parser.add_argument("--start", default="2023-07-01")
    parser.add_argument("--root", default="")
    parser.add_argument("--symbols", nargs="+", default=list(SYMBOLS))
    parser.add_argument("--universe", default="", help="alamet.market ile kaydedilmiş evren dosyası")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--no-tail", action="store_true", help="REST ile kuyruk tamamlamayı atla")
    args = parser.parse_args(argv)
    tf = from_interval(args.tf)
    root = Path(args.root) if args.root else default_root(tf)
    symbols = bases(load_snapshot(Path(args.universe))) if args.universe else tuple(args.symbols)
    asyncio.run(
        bulk_download(
            root, symbols, to_ms(args.start), tf, workers=args.workers, tail=not args.no_tail
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
