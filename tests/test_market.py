"""Evren seçimi: ayrıştırma, süzgeçler, dört seçim biçimi, kayıt ve belirlenimcilik."""

import asyncio
import random

import pytest

from alamet import market
from alamet.binance_data import pair
from alamet.market import (
    UniverseSpec,
    bases,
    eligible,
    fetch_market,
    load_snapshot,
    parse_exchange_info,
    parse_volumes,
    save_snapshot,
    select_universe,
)

NOW = 1_800_000_000_000
DAY = 86_400_000


def raw(symbol, *, status="TRADING", contract="PERPETUAL", quote="USDT", underlying="COIN",
        age_days=400, base=None):  # fmt: skip
    return {
        "symbol": symbol,
        "baseAsset": base or symbol.removesuffix(quote),
        "quoteAsset": quote,
        "status": status,
        "contractType": contract,
        "underlyingType": underlying,
        "onboardDate": NOW - age_days * DAY,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000"},
            {
                "filterType": "MARKET_LOT_SIZE",
                "stepSize": "0.01",
                "minQty": "0.01",
                "maxQty": "500",
            },
            {"filterType": "MIN_NOTIONAL", "notional": "5"},
        ],
    }


EXCHANGE = {
    "symbols": [
        raw("BTCUSDT"),
        raw("ETHUSDT"),
        raw("SOLUSDT"),
        raw("DOGEUSDT"),
        raw("NEWUSDT", age_days=3),
        raw("HALTUSDT", status="BREAK"),
        raw("BTCUSDT_250926", contract="CURRENT_QUARTER", base="BTC"),
        raw("BTCBUSD", quote="BUSD", base="BTC"),
        raw("NVDAUSDT", underlying="EQUITY"),
        raw("XAUUSDT", underlying="COMMODITY"),
        raw("PREUSDT", underlying="PREMARKET"),
        raw("LOWUSDT"),
    ]
}
TICKERS = [
    {"symbol": "BTCUSDT", "quoteVolume": "9000000000"},
    {"symbol": "ETHUSDT", "quoteVolume": "5000000000"},
    {"symbol": "SOLUSDT", "quoteVolume": "5000000000"},
    {"symbol": "DOGEUSDT", "quoteVolume": "800000000"},
    {"symbol": "NEWUSDT", "quoteVolume": "700000000"},
    {"symbol": "HALTUSDT", "quoteVolume": "600000000"},
    {"symbol": "BTCUSDT_250926", "quoteVolume": "100000000"},
    {"symbol": "BTCBUSD", "quoteVolume": "100000000"},
    {"symbol": "NVDAUSDT", "quoteVolume": "400000000"},
    {"symbol": "XAUUSDT", "quoteVolume": "300000000"},
    {"symbol": "PREUSDT", "quoteVolume": "200000000"},
    {"symbol": "LOWUSDT", "quoteVolume": "1000"},
]


@pytest.fixture(scope="module")
def infos():
    return parse_exchange_info(EXCHANGE, parse_volumes(TICKERS))


def test_parse_reads_contract_rules_and_volume(infos):
    btc = infos[0]
    assert (btc.base, btc.symbol, btc.status, btc.contract_type) == (
        "BTC",
        "BTCUSDT",
        "TRADING",
        "PERPETUAL",
    )
    assert (btc.tick_size, btc.step_size, btc.min_qty, btc.max_qty) == (0.1, 0.001, 0.001, 1000.0)
    assert (btc.market_step_size, btc.market_min_qty, btc.market_max_qty) == (0.01, 0.01, 500.0)
    assert btc.min_notional == 5.0 and btc.quote_volume == 9e9 and btc.onboard_ms == NOW - 400 * DAY


def test_parse_falls_back_when_market_lot_filter_is_missing():
    item = raw("ABCUSDT")
    item["filters"] = [f for f in item["filters"] if f["filterType"] != "MARKET_LOT_SIZE"]
    (info,) = parse_exchange_info({"symbols": [item]})
    assert info.market_step_size == info.step_size == 0.001 and info.quote_volume == 0.0


def test_filters_drop_halted_dated_foreign_quote_and_non_crypto(infos):
    names = {info.symbol for info in eligible(infos, UniverseSpec(), NOW)}
    assert names == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "NEWUSDT", "LOWUSDT"}


def test_volume_and_age_thresholds_and_explicit_exclusions(infos):
    spec = UniverseSpec(min_quote_volume=1e6, min_age_days=30, exclude=("DOGE",))
    assert {i.symbol for i in eligible(infos, spec, NOW)} == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    with_stocks = UniverseSpec(underlying=("COIN", "EQUITY"))
    assert "NVDAUSDT" in {i.symbol for i in eligible(infos, with_stocks, NOW)}


def test_top_n_orders_by_volume_then_name_and_is_capped(infos):
    top = select_universe(infos, UniverseSpec(n=3, min_quote_volume=1e6), NOW)
    assert bases(top) == ("BTC", "ETH", "SOL")
    assert bases(select_universe(infos, UniverseSpec(n=50, min_quote_volume=1e6), NOW)) == (
        "BTC", "ETH", "SOL", "DOGE", "NEW",
    )  # fmt: skip


def test_list_mode_keeps_the_given_order_and_fails_loudly_on_unknown_names(infos):
    spec = UniverseSpec(mode="list", symbols=("SOL", "BTC"))
    assert bases(select_universe(infos, spec, NOW)) == ("SOL", "BTC")
    with pytest.raises(ValueError, match="HALT"):
        select_universe(infos, UniverseSpec(mode="list", symbols=("BTC", "HALT", "NVDA")), NOW)


def test_all_mode_returns_every_eligible_symbol_by_volume(infos):
    chosen = select_universe(infos, UniverseSpec(mode="all", min_quote_volume=1e6), NOW)
    assert bases(chosen) == ("BTC", "ETH", "SOL", "DOGE", "NEW")
    assert len(select_universe(infos, UniverseSpec(mode="all"), NOW)) == 6


def test_selection_does_not_depend_on_input_order(infos):
    spec = UniverseSpec(mode="all")
    shuffled = list(infos)
    random.Random(4).shuffle(shuffled)
    assert select_universe(shuffled, spec, NOW) == select_universe(infos, spec, NOW)


def test_spec_validation():
    for bad in (
        dict(mode="top"),
        dict(mode="top_n", n=0),
        dict(mode="list"),
        dict(mode="list", symbols=("BTC", "BTC")),
    ):
        with pytest.raises(ValueError):
            UniverseSpec(**bad)


def test_snapshot_round_trip_keeps_order_and_order_rules(infos, tmp_path):
    spec = UniverseSpec(n=3)
    chosen = select_universe(infos, spec, NOW)
    path = tmp_path / "evren" / "universe.json"
    save_snapshot(path, chosen, spec)
    assert load_snapshot(path) == chosen
    assert [pair(base) for base in bases(load_snapshot(path))] == [i.symbol for i in chosen]


def test_fetch_market_merges_exchange_rules_with_volume():
    async def fake(url):
        return TICKERS if url.endswith("ticker/24hr") else EXCHANGE

    result = asyncio.run(fetch_market(fake))
    by_symbol = {info.symbol: info for info in result}
    assert by_symbol["ETHUSDT"].quote_volume == 5e9 and by_symbol["ETHUSDT"].min_notional == 5.0
    assert len(result) == len(EXCHANGE["symbols"]) and market.QUOTE == "USDT"
