"""Canlı mum birleştirici: tam/eksik mumlar, sıra, bekleme süresi, tekrar ve geç veri, akış mesajı."""

import json

import pytest

from alamet.aggregator import (
    BarAggregator,
    Candle,
    merge_minutes,
    parse_kline_message,
)
from alamet.timeframe import SignalTf

T0 = 1_800_000_000_000 - (1_800_000_000_000 % 900_000)
MIN = 60_000
FIFTEEN = SignalTf(900)


def minute(index, base=100.0, volume=1.0):
    price = base + index
    return Candle(
        T0 + index * MIN, price, price + 2, price - 1, price + 0.5, volume, volume * price
    )


def feed(agg, symbol, indices, base=100.0):
    for index in indices:
        agg.add(symbol, minute(index, base))


def test_merge_takes_first_open_last_close_extremes_and_sums():
    mins = [minute(i) for i in range(3)]
    merged = merge_minutes(T0, mins)
    assert (merged.open_time, merged.open, merged.close) == (T0, 100.0, 102.5)
    assert merged.high == 104.0 and merged.low == 99.0
    assert merged.volume == 3.0 and merged.quote_volume == pytest.approx(100 + 101 + 102)


def test_bucket_closes_at_once_when_every_symbol_has_every_minute():
    agg = BarAggregator(("A", "B"), FIFTEEN)
    agg.start_at(T0)
    feed(agg, "A", range(15))
    feed(agg, "B", range(14), base=50.0)
    assert agg.ready(T0 + 15 * MIN) == []
    feed(agg, "B", [14], base=50.0)
    (closed,) = agg.ready(T0 + 15 * MIN)
    assert closed.start_ms == T0 and closed.missing == () and set(closed.candles) == {"A", "B"}
    assert closed.candles["B"].open == 50.0 and closed.candles["B"].close == 64.5
    assert agg.next_bucket == T0 + 900_000 and agg.ready(T0 + 15 * MIN + 1_000) == []


def test_missing_minute_waits_for_the_grace_period_then_reports_the_symbol():
    agg = BarAggregator(("A", "B"), FIFTEEN, grace_ms=10_000)
    agg.start_at(T0)
    feed(agg, "A", range(15))
    feed(agg, "B", [i for i in range(15) if i != 7])
    assert agg.ready(T0 + 900_000 + 9_999) == []
    (closed,) = agg.ready(T0 + 900_000 + 10_000)
    assert closed.missing == ("B",) and set(closed.candles) == {"A"}
    assert agg.stats.repaired_buckets == 1 and agg.stats.missing_candles == 1


def test_buckets_are_emitted_in_order_and_silent_buckets_come_out_as_all_missing():
    agg = BarAggregator(("A",), FIFTEEN, grace_ms=1_000)
    agg.start_at(T0)
    feed(agg, "A", range(30, 45))
    out = agg.ready(T0 + 45 * MIN)
    assert [b.start_ms for b in out] == [T0, T0 + 900_000, T0 + 1_800_000]
    assert [b.missing for b in out] == [("A",), ("A",), ()]


def test_later_complete_bucket_is_held_back_until_the_earlier_one_resolves():
    agg = BarAggregator(("A",), FIFTEEN, grace_ms=3_600_000)
    agg.start_at(T0)
    feed(agg, "A", range(0, 14))
    feed(agg, "A", range(15, 30))
    assert agg.ready(T0 + 30 * MIN) == []
    out = agg.ready(T0 + 15 * MIN + 3_600_000)
    assert [b.missing for b in out] == [("A",), ()]


def test_duplicates_overwrite_late_and_unknown_are_rejected():
    agg = BarAggregator(("A",), FIFTEEN)
    agg.start_at(T0 + 900_000)
    assert not agg.add("A", minute(3)) and agg.stats.late == 1
    assert not agg.add("Z", minute(20)) and agg.stats.unknown == 1
    agg.add("A", minute(20, base=1.0))
    agg.add("A", minute(20, base=200.0))
    assert agg.stats.duplicates == 1 and agg.stats.accepted == 1
    feed(agg, "A", [i for i in range(15, 30) if i != 20])
    (closed,) = agg.ready(T0 + 30 * MIN)
    assert closed.candles["A"].high == max(
        m.high for m in [minute(i) for i in range(15, 30) if i != 20] + [minute(20, base=200.0)]
    )


def test_first_data_sets_the_start_and_the_partial_first_bucket_needs_repair():
    agg = BarAggregator(("A",), FIFTEEN, grace_ms=0)
    feed(agg, "A", range(5, 15))
    assert agg.next_bucket == T0
    (closed,) = agg.ready(T0 + 900_000)
    assert closed.missing == ("A",)


def test_other_timeframes_and_alignment_rules():
    five = BarAggregator(("A",), SignalTf(300))
    five.start_at(T0)
    feed(five, "A", range(5))
    assert [b.missing for b in five.ready(T0 + 5 * MIN)] == [()]
    one = BarAggregator(("A",), SignalTf(60))
    one.start_at(T0)
    one.add("A", minute(0))
    assert one.ready(T0 + MIN)[0].candles["A"].open_time == T0
    with pytest.raises(ValueError):
        BarAggregator(("A",), FIFTEEN).start_at(T0 + MIN)
    assert BarAggregator(("A",), FIFTEEN).bucket_of(T0 + 14 * MIN + 59_000) == T0


def kline_message(closed=True, wrap=True):
    k = {"t": T0, "T": T0 + 59_999, "s": "BTCUSDT", "i": "1m", "o": "100", "c": "101", "h": "102",
         "l": "99", "v": "5", "n": 3, "x": closed, "q": "500", "V": "2", "Q": "200", "B": "0"}  # fmt: skip
    data = {"e": "kline", "E": T0, "s": "BTCUSDT", "k": k}
    return json.dumps({"stream": "btcusdt@kline_1m", "data": data} if wrap else data)


def test_parse_reads_combined_and_plain_kline_messages():
    for wrap in (True, False):
        symbol, candle, closed = parse_kline_message(kline_message(wrap=wrap))
        assert (symbol, closed) == ("BTCUSDT", True)
        assert candle == Candle(T0, 100.0, 102.0, 99.0, 101.0, 5.0, 500.0)
    assert parse_kline_message(kline_message(closed=False))[2] is False
    assert parse_kline_message(json.dumps({"result": None, "id": 1})) is None
