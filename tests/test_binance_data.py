import asyncio

import numpy as np
import pandas as pd
import pytest

from alamet import binance_data as bd

T0 = 1_700_000_000_000 - (1_700_000_000_000 % bd.BAR_MS)


def make_fetcher(total: int, skip: set[int] | None = None, calls: list | None = None):
    skip = skip or set()

    async def fetch(_symbol: str, start_ms: int) -> pd.DataFrame:
        if calls is not None:
            calls.append(start_ms)
        first = max(0, -(-(start_ms - T0) // bd.BAR_MS))
        index = [i for i in range(first, min(first + bd.PAGE, total)) if i not in skip]
        times = np.array([T0 + i * bd.BAR_MS for i in index], dtype=np.int64)
        base = 100.0 + np.array(index, dtype=float)
        return pd.DataFrame(
            {
                "open_time": times,
                "open": base,
                "high": base + 1,
                "low": base - 1,
                "close": base + 0.5,
                "volume": np.full(len(index), 10.0),
                "quote_asset_volume": base * 10,
            }
        )

    return fetch


def run(coro):
    return asyncio.run(coro)


def test_download_pages_and_resume(tmp_path):
    calls: list[int] = []
    added, total = run(bd.update_symbol(tmp_path, "BTC", T0, make_fetcher(2500, calls=calls), 0))
    assert (added, total) == (2500, 2500)
    assert len(calls) == 3
    calls.clear()
    added, total = run(bd.update_symbol(tmp_path, "BTC", T0, make_fetcher(2600, calls=calls), 0))
    assert (added, total) == (100, 2600)
    assert calls[0] == T0 + 2500 * bd.BAR_MS
    df = bd.load_cache(tmp_path, "BTC")
    assert bd.integrity("BTC", df).ok


def test_no_duplicates_when_rerun(tmp_path):
    run(bd.update_symbol(tmp_path, "ETH", T0, make_fetcher(1500), 0))
    added, total = run(bd.update_symbol(tmp_path, "ETH", T0, make_fetcher(1500), 0))
    assert (added, total) == (0, 1500)


def test_integrity_reports_gap(tmp_path):
    run(bd.update_symbol(tmp_path, "SOL", T0, make_fetcher(50, skip={10, 11, 12}), 0))
    rep = bd.integrity("SOL", bd.load_cache(tmp_path, "SOL"))
    assert not rep.ok
    assert rep.missing_bars == 3
    assert rep.gaps == [(T0 + 9 * bd.BAR_MS, 3)]


def test_integrity_flags_bad_ohlc():
    df = pd.DataFrame(
        {
            "open_time": [T0, T0 + bd.BAR_MS],
            "open": [1.0, 1.0],
            "high": [2.0, 0.5],
            "low": [0.5, 0.9],
            "close": [1.5, 1.0],
            "volume": [1.0, 1.0],
            "quote_volume": [1.0, 1.0],
        }
    )
    assert bd.integrity("X", df).bad_ohlc == 1


def test_universe_from_cache(tmp_path):
    symbols = ("BTC", "ETH")
    run(bd.update_symbol(tmp_path, "BTC", T0, make_fetcher(40), 0))
    run(bd.update_symbol(tmp_path, "ETH", T0, make_fetcher(30), 0))
    uni = bd.load_universe_binance(tmp_path, symbols)
    assert uni.bars == 30 and uni.n == 2
    assert uni.time[0] == T0 // 1000
    assert uni.close[0, 0] == pytest.approx(100.5)
    assert uni.volume.shape == (30, 2)


def test_universe_rejects_gaps_unless_allowed(tmp_path):
    run(bd.update_symbol(tmp_path, "BTC", T0, make_fetcher(40, skip={5}), 0))
    run(bd.update_symbol(tmp_path, "ETH", T0, make_fetcher(40), 0))
    with pytest.raises(ValueError, match="eksik bar"):
        bd.load_universe_binance(tmp_path, ("BTC", "ETH"))
    assert bd.load_universe_binance(tmp_path, ("BTC", "ETH"), allow_gaps=True).bars == 39


def test_universe_empty_cache_errors(tmp_path):
    with pytest.raises(ValueError, match="önbellek boş"):
        bd.load_universe_binance(tmp_path, ("BTC",))


def test_to_ms_is_utc():
    assert bd.to_ms("2026-03-01") == 1772323200000


def late_frame(start_index, total):
    base = 100.0 + np.arange(start_index, total, dtype=float)
    return pd.DataFrame(
        {
            "open_time": (T0 + np.arange(start_index, total) * bd.BAR_MS).astype("int64"),
            "open": base,
            "high": base + 1,
            "low": base - 1,
            "close": base + 0.5,
            "volume": np.full(total - start_index, 10.0),
            "quote_volume": base * 10,
        }
    )


def test_union_universe_keeps_late_listed_symbols_as_nan_instead_of_shrinking_the_axis(tmp_path):
    bd.save_cache(tmp_path, "BTC", late_frame(0, 40))
    bd.save_cache(tmp_path, "NEW", late_frame(10, 40))
    narrow = bd.load_universe_binance(tmp_path, ("BTC", "NEW"))
    wide = bd.load_universe_binance(tmp_path, ("BTC", "NEW"), union=True)
    assert narrow.bars == 30 and wide.bars == 40
    assert np.isnan(wide.close[:10, 1]).all() and not np.isnan(wide.close[10:, 1]).any()
    assert np.isnan(wide.volume[:10, 1]).all() and not np.isnan(wide.close[:, 0]).any()
    assert np.array_equal(wide.close[10:, 1], narrow.close[:, 1])


def test_union_universe_still_rejects_axis_gaps(tmp_path):
    bd.save_cache(tmp_path, "BTC", late_frame(0, 20).drop(index=5).reset_index(drop=True))
    bd.save_cache(tmp_path, "NEW", late_frame(0, 20).drop(index=5).reset_index(drop=True))
    with pytest.raises(ValueError, match="eksik bar"):
        bd.load_universe_binance(tmp_path, ("BTC", "NEW"), union=True)
