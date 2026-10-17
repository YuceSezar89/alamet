"""Canlı akış: hazırlık, birleştirme, REST onarımı, süresi dolan mumlar, uzlaştırma, akış başlatma."""

import asyncio
import json

import pandas as pd
import pytest

from alamet import binance_data as bd
from alamet.aggregator import Candle
from alamet.livefeed import BarSet, Correction, LiveFeed, candles_from_frame, same_candle
from alamet.timeframe import SignalTf

TF = SignalTf(300)
MIN = 60_000
B0 = 1_800_000_000_000 - (1_800_000_000_000 % 300_000)
SYMBOLS = ("AAA", "BBB")


def candle(open_time, price=100.0, volume=1.0):
    return Candle(open_time, price, price + 2, price - 1, price + 0.5, volume, volume * price)


def seed_cache(root, last_buckets=3):
    for number, base in enumerate(SYMBOLS):
        frame = pd.DataFrame(
            [candle(B0 - (last_buckets - i) * 300_000, 100.0 + number + i).__dict__
             for i in range(last_buckets)]
        )  # fmt: skip
        bd.save_cache(root, base, frame[bd.COLUMNS])


def ws_message(base, open_time, price, closed=True):
    k = {"t": open_time, "T": open_time + 59_999, "s": f"{base}USDT", "i": "1m", "o": str(price),
         "c": str(price + 0.5), "h": str(price + 2), "l": str(price - 1), "v": "1", "n": 1,
         "x": closed, "q": str(price), "V": "0", "Q": "0", "B": "0"}  # fmt: skip
    return json.dumps(
        {
            "stream": f"{base.lower()}usdt@kline_1m",
            "data": {"e": "kline", "s": f"{base}USDT", "k": k},
        }
    )


class Exchange:
    """Sahte REST: resmî mumlar ve istek sayacı."""

    def __init__(self):
        self.official: dict[tuple[str, int], Candle] = {}
        self.calls = 0

    async def fetch(self, symbol, interval, start_ms, limit):
        self.calls += 1
        base = symbol.removesuffix("USDT")
        return [
            self.official[(base, start_ms + i * 300_000)]
            for i in range(limit)
            if (base, start_ms + i * 300_000) in self.official
        ]


async def no_pages(_symbol, _start):
    return pd.DataFrame()


def make_feed(tmp_path, exchange=None, **kw):
    seed_cache(tmp_path)
    exchange = exchange or Exchange()
    bars, fixes = [], []
    feed = LiveFeed(
        tmp_path, SYMBOLS, TF, bars.append, fixes.append, fetch_candles=exchange.fetch,
        fetch_page=no_pages, repair_delay_s=0.0, **kw,
    )  # fmt: skip
    return feed, exchange, bars, fixes


def push_bucket(feed, start, skip=()):
    for base in SYMBOLS:
        for i in range(5):
            if (base, i) not in skip:
                feed.on_message(ws_message(base, start + i * MIN, 100.0 + i))


def test_prepare_starts_the_aggregator_right_after_the_newest_cached_bar(tmp_path):
    feed, *_ = make_feed(tmp_path)
    assert asyncio.run(feed.prepare()) == B0 and feed.aggregator.next_bucket == B0


def test_prepare_refuses_an_empty_cache(tmp_path):
    feed, *_ = make_feed(tmp_path)
    (tmp_path / "BBBUSDT.parquet").unlink()
    with pytest.raises(ValueError, match="bulk_data"):
        asyncio.run(feed.prepare())


def test_complete_bucket_is_built_stored_and_announced(tmp_path):
    feed, _, bars, _ = make_feed(tmp_path)
    asyncio.run(feed.prepare())
    push_bucket(feed, B0)
    (bar,) = asyncio.run(feed.process(B0 + 300_000))
    assert (
        isinstance(bar, BarSet) and bar.start_ms == B0 and bar.repaired == () and bar.missing == ()
    )
    assert list(bar.candles) == list(SYMBOLS) and bar.candles["AAA"].open == 100.0
    assert bar.candles["AAA"].high == 106.0 and bar.candles["AAA"].volume == 5.0
    assert bars == [bar]
    stored = bd.load_cache(tmp_path, "AAA")
    assert stored["open_time"].iloc[-1] == B0 and len(stored) == 4


def test_unclosed_and_non_kline_messages_are_ignored(tmp_path):
    feed, *_ = make_feed(tmp_path)
    asyncio.run(feed.prepare())
    feed.on_message(ws_message("AAA", B0, 100.0, closed=False))
    feed.on_message(json.dumps({"result": None, "id": 1}))
    assert feed.aggregator.stats.accepted == 0


def test_a_missing_minute_is_repaired_from_the_official_candle(tmp_path):
    feed, exchange, bars, _ = make_feed(tmp_path)
    exchange.official[("BBB", B0)] = candle(B0, 77.0)
    asyncio.run(feed.prepare())
    push_bucket(feed, B0, skip={("BBB", 2)})
    assert asyncio.run(feed.process(B0 + 300_000 + 1_000)) == []
    (bar,) = asyncio.run(feed.process(B0 + 300_000 + 15_000))
    assert bar.repaired == ("BBB",) and bar.missing == () and bar.candles["BBB"] == candle(B0, 77.0)
    assert bd.load_cache(tmp_path, "BBB")["close"].iloc[-1] == 77.5


def test_unrepairable_symbols_are_reported_after_the_attempts(tmp_path):
    feed, exchange, bars, _ = make_feed(tmp_path, repair_attempts=3)
    asyncio.run(feed.prepare())
    push_bucket(feed, B0, skip={("BBB", 0)})
    (bar,) = asyncio.run(feed.process(B0 + 300_000 + 20_000))
    assert bar.missing == ("BBB",) and list(bar.candles) == ["AAA"] and exchange.calls == 3
    assert bd.load_cache(tmp_path, "BBB")["open_time"].iloc[-1] == B0 - 300_000


def test_silent_bucket_is_announced_as_fully_missing_then_next_bucket_follows(tmp_path):
    feed, exchange, bars, _ = make_feed(tmp_path, repair_attempts=1)
    asyncio.run(feed.prepare())
    push_bucket(feed, B0 + 300_000)
    out = asyncio.run(feed.process(B0 + 600_000))
    assert [b.start_ms for b in out] == [B0, B0 + 300_000]
    assert out[0].missing == SYMBOLS and out[1].missing == ()


def test_reconcile_corrects_stale_candles_and_fills_absent_ones(tmp_path):
    feed, exchange, _, fixes = make_feed(tmp_path)
    asyncio.run(feed.prepare())
    last = B0 - 300_000
    exchange.official[("AAA", last)] = candle(last, 999.0)
    old_row = bd.load_cache(tmp_path, "BBB").iloc[-1]
    exchange.official[("BBB", last)] = Candle(last, *(float(old_row[c]) for c in bd.COLUMNS[1:]))
    exchange.official[("BBB", B0)] = candle(B0, 55.0)
    corrections = asyncio.run(feed.reconcile(buckets=2, now_ms=B0 + 300_000 + 5_000))
    assert [(c.base, c.start_ms, c.old is None) for c in corrections] == [
        ("AAA", last, False),
        ("BBB", B0, True),
    ]
    assert fixes == corrections and isinstance(corrections[0], Correction)
    assert bd.load_cache(tmp_path, "AAA")["close"].iloc[-1] == 999.5
    assert bd.load_cache(tmp_path, "BBB")["open_time"].iloc[-1] == B0
    assert asyncio.run(feed.reconcile(buckets=2, now_ms=B0 + 300_000 + 5_000)) == []


def test_start_subscribes_one_minute_streams_and_stop_closes_them(tmp_path):
    seen = {}

    async def starter(streams, on_message):
        seen["streams"] = streams
        seen["handler"] = on_message

        async def stop():
            seen["stopped"] = True

        return stop

    feed, *_ = make_feed(tmp_path)
    feed.start_stream = starter
    asyncio.run(feed.start())
    assert (
        seen["streams"] == ["aaausdt@kline_1m", "bbbusdt@kline_1m"]
        and seen["handler"] == feed.on_message
    )
    asyncio.run(feed.stop())
    assert seen["stopped"] and feed._stop is None  # pylint: disable=protected-access


def test_candle_helpers():
    frame = pd.DataFrame([candle(B0).__dict__])[bd.COLUMNS]
    assert candles_from_frame(frame) == [candle(B0)]
    assert same_candle(candle(B0), candle(B0, volume=1.0 + 1e-12))
    assert not same_candle(candle(B0), candle(B0, price=100.1))
    assert not same_candle(candle(B0), candle(B0, volume=2.0))


def test_update_symbol_uses_the_cache_interval_for_the_default_page_fetcher(tmp_path, monkeypatch):
    seen = []

    async def page(_symbol, _start, interval="15m"):
        seen.append(interval)
        return pd.DataFrame()

    monkeypatch.setattr(bd, "binance_page", page)
    seed_cache(tmp_path)
    asyncio.run(bd.update_symbol(tmp_path, "AAA", 0, tf=TF))
    assert seen == ["5m"]
