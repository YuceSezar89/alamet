"""Toplu geçmiş veri: arşiv adresleri, ayrıştırma, sağlama toplamı, dosya planı, önbelleğe birleştirme."""

import asyncio
import hashlib
import io
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from alamet import binance_data as bd
from alamet.bulk_data import (
    archive_url,
    bulk_download,
    bulk_symbol,
    fetch_archive,
    parse_archive,
    plan_files,
)
from alamet.timeframe import SignalTf

HEADER = "open_time,open,high,low,close,volume,close_time,quote_volume,count,tb,tbq,ignore"
FIVE = SignalTf(300)


def ms(year, month, day=1, hour=0):
    return int(datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp() * 1000)


def rows(start_ms, count, step_ms, base=100.0):
    lines = []
    for i in range(count):
        open_time = start_ms + i * step_ms
        price = base + i
        lines.append(
            f"{open_time},{price},{price + 1},{price - 1},{price + 0.5},10,{open_time + step_ms - 1},"
            f"{price * 10},5,1,1,0"
        )
    return lines


def make_zip(lines, header=True):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.csv", "\n".join(([HEADER] if header else []) + lines))
    return buffer.getvalue()


def test_archive_urls_for_monthly_and_daily_files():
    assert archive_url("BTCUSDT", "15m", 2024, 1) == (
        "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2024-01.zip"
    )
    assert archive_url("ETHUSDT", "5m", 2026, 10, 3).endswith(
        "daily/klines/ETHUSDT/5m/ETHUSDT-5m-2026-10-03.zip"
    )


def test_parse_reads_with_or_without_header_and_converts_microseconds():
    lines = rows(ms(2024, 1), 4, 900_000)
    for header in (True, False):
        frame = parse_archive(make_zip(lines, header))
        assert list(frame.columns) == bd.COLUMNS and len(frame) == 4
        assert frame.open_time.iloc[0] == ms(2024, 1) and frame.quote_volume.iloc[1] == 1010.0
    micro = [
        line.replace(line.split(",")[0], str(int(line.split(",")[0]) * 1000), 1) for line in lines
    ]
    assert (
        parse_archive(make_zip(micro)).open_time.tolist()
        == parse_archive(make_zip(lines)).open_time.tolist()
    )


def test_checksum_must_match_the_downloaded_bytes():
    content = make_zip(rows(ms(2024, 1), 2, 900_000))
    good = hashlib.sha256(content).hexdigest() + "  BTCUSDT-15m-2024-01.zip"
    assert len(parse_archive(content, good)) == 2
    with pytest.raises(ValueError, match="sağlama"):
        parse_archive(content, "0" * 64 + "  x.zip")


def test_plan_lists_complete_months_and_the_days_before_today():
    months, days = plan_files(ms(2026, 8, 15), ms(2026, 10, 3, 10))
    assert months == [(2026, 8), (2026, 9)] and days == [(2026, 10, 1), (2026, 10, 2)]
    months, days = plan_files(ms(2025, 11, 20), ms(2026, 2, 1, 5))
    assert months == [(2025, 11), (2025, 12), (2026, 1)] and days == []
    assert plan_files(ms(2026, 10, 3), ms(2026, 10, 3, 9)) == ([], [])
    assert plan_files(ms(2026, 10, 1, 7), ms(2026, 10, 3, 9))[1] == [(2026, 10, 1), (2026, 10, 2)]


def archive_server(files):
    async def fetch(url):
        return files.get(url)

    return fetch


def month_files(symbol, interval, months, step_ms, checksums=True):
    files = {}
    for year, month, count in months:
        content = make_zip(rows(ms(year, month), count, step_ms, base=100.0 + month))
        url = archive_url(symbol, interval, year, month)
        files[url] = content
        if checksums:
            files[url + ".CHECKSUM"] = (hashlib.sha256(content).hexdigest() + "  f.zip").encode()
    return files


def test_fetch_archive_returns_none_for_missing_files_and_verifies_present_ones():
    files = month_files("BTCUSDT", "5m", [(2024, 1, 3)], 300_000)
    fetch = archive_server(files)
    assert asyncio.run(fetch_archive(fetch, archive_url("BTCUSDT", "5m", 2024, 2))) is None
    assert len(asyncio.run(fetch_archive(fetch, archive_url("BTCUSDT", "5m", 2024, 1)))) == 3
    url = archive_url("BTCUSDT", "5m", 2024, 1)
    files[url + ".CHECKSUM"] = b"deadbeef  x"
    with pytest.raises(ValueError):
        asyncio.run(fetch_archive(fetch, url))


def test_bulk_symbol_merges_months_skips_pre_listing_months_and_continues_the_cache(tmp_path):
    files = month_files("NEWUSDT", "5m", [(2024, 3, 5), (2024, 4, 4)], 300_000)
    fetch = archive_server(files)
    now = ms(2024, 5, 1, 12)
    added, total = asyncio.run(bulk_symbol(tmp_path, "NEW", ms(2024, 1), now, fetch, FIVE))
    assert (added, total) == (9, 9)
    times = bd.load_cache(tmp_path, "NEW")["open_time"].to_numpy()
    assert times[0] == ms(2024, 3) and (np.diff(times[:5]) == 300_000).all()
    again = asyncio.run(bulk_symbol(tmp_path, "NEW", ms(2024, 1), now, fetch, FIVE))
    assert again[1] == 9


def test_bulk_symbol_overwrites_cached_duplicates_without_double_counting(tmp_path):
    bd.save_cache(tmp_path, "AAA", parse_archive(make_zip(rows(ms(2024, 3), 3, 300_000))))
    files = month_files("AAAUSDT", "5m", [(2024, 3, 6)], 300_000)
    added, total = asyncio.run(
        bulk_symbol(tmp_path, "AAA", ms(2024, 1), ms(2024, 4, 2), archive_server(files), FIVE)
    )
    assert (added, total) == (3, 6)
    assert bd.integrity("AAA", bd.load_cache(tmp_path, "AAA"), FIVE).ok


def test_bulk_download_runs_every_symbol_and_reports(tmp_path):
    files = {}
    for base in ("AAA", "BBB", "CCC"):
        files.update(month_files(f"{base}USDT", "5m", [(2024, 3, 4)], 300_000))
    result = asyncio.run(
        bulk_download(
            tmp_path, ("AAA", "BBB", "CCC"), ms(2024, 3), FIVE, archive_server(files),
            workers=2, now_ms=ms(2024, 4, 2), tail=False,
        )
    )  # fmt: skip
    assert result == {"AAA": (4, 4), "BBB": (4, 4), "CCC": (4, 4)}
    assert not pd.read_parquet(tmp_path / "AAAUSDT.parquet").empty
