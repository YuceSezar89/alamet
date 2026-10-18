"""Emir niyeti: yuvarlama, borsa kuralları, ölçek, tavanlar, kapanış, kopya koruması."""

from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest

from alamet.events import CloseEvent, OpenEvent
from alamet.market import SymbolInfo
from alamet.orders import (
    BUY,
    SELL,
    OrderIntent,
    OrderRules,
    Rejection,
    RiskLimits,
    client_id,
    order_side,
    round_down,
    size_close,
    size_open,
    stop_client_id,
    stop_price,
)
from alamet.stages import stage_params
from alamet.stepper import StepSystem
from tests.test_stepper import first_day_of, random_universe

BTC_RULES = OrderRules(step=0.001, min_qty=0.001, max_qty=120.0, min_notional=50.0)
LIMITS = RiskLimits(scale=10.0, max_position_usdt=5_000.0, max_total_usdt=20_000.0)


def opened(notional=100.0, entry=80_000.0, side=1, trade_id=7):
    return SimpleNamespace(trade_id=trade_id, entry=entry, notional=notional, side=side)


def test_rounding_is_exact_decimal_and_always_downward():
    assert round_down(0.3, 0.1) == 0.3 and round_down(0.30000000000000004, 0.1) == 0.3
    assert round_down(0.0019999, 0.001) == 0.001 and round_down(1.9999999, 0.001) == 1.999
    assert round_down(5.0, 0.5) == 5.0 and round_down(123.456, 1.0) == 123.0
    assert round_down(0.0, 0.001) == 0.0
    assert np.isnan(round_down(float("nan"), 0.001)) and round_down(2.5, 0.0) == 2.5
    value = round_down(0.1 + 0.2, 0.1)
    assert Decimal(str(value)) % Decimal("0.1") == 0


def test_rules_prefer_market_lot_filters():
    info = SymbolInfo(
        "BTC", "BTCUSDT", "TRADING", "PERPETUAL", "USDT", "COIN", 0, 1.0, 0.1, 0.001, 0.001,
        1000.0, 0.01, 0.01, 200.0, 100.0,
    )  # fmt: skip
    rules = OrderRules.from_info(info)
    assert (rules.step, rules.min_qty, rules.max_qty, rules.min_notional) == (
        0.01,
        0.01,
        200.0,
        100.0,
    )
    fallback = SymbolInfo(
        "X", "XUSDT", "TRADING", "PERPETUAL", "USDT", "COIN", 0, 1.0, 0.1, 0.001, 0.001,
        1000.0, 0.0, 0.0, 0.0, 5.0,
    )  # fmt: skip
    assert (
        OrderRules.from_info(fallback).step == 0.001
        and OrderRules.from_info(fallback).max_qty == 1000.0
    )


def test_open_scales_the_virtual_notional_and_rounds_the_quantity_down():
    intent = size_open(opened(notional=100.0), "BTC", BTC_RULES, LIMITS, 0.0, 1_800_000_000)
    assert isinstance(intent, OrderIntent)
    assert intent.quantity == 0.012 and intent.side == BUY and not intent.reduce_only
    assert intent.symbol == "BTCUSDT" and intent.ref_price == 80_000.0
    assert intent.notional == pytest.approx(960.0)
    assert intent.client_id == "al7o" and intent.bar_time == 1_800_000_000
    short = size_open(opened(side=-1), "BTC", BTC_RULES, LIMITS, 0.0, 0)
    assert short.side == SELL


def test_exchange_rules_reject_with_the_reason():
    def reason(result):
        assert isinstance(result, Rejection)
        return result.reason

    tiny = size_open(opened(notional=0.5), "BTC", BTC_RULES, LIMITS, 0.0, 0)
    assert reason(tiny) == "min_miktar"
    assert (
        reason(
            size_open(
                opened(notional=20.0, entry=80_000.0),
                "BTC",
                BTC_RULES,
                RiskLimits(1.0, 1e5, 1e6),
                0.0,
                0,
            )
        )
        == "min_miktar"
    )
    near = OrderRules(0.001, 0.001, 120.0, 100.0)
    assert (
        reason(
            size_open(
                opened(notional=99.0, entry=100.0), "X", near, RiskLimits(1.0, 1e5, 1e6), 0.0, 0
            )
        )
        == "min_tutar"
    )
    big = OrderRules(0.001, 0.001, 0.05, 5.0)
    assert (
        reason(
            size_open(
                opened(notional=1000.0, entry=100.0), "X", big, RiskLimits(1.0, 1e5, 1e6), 0.0, 0
            )
        )
        == "azami_miktar"
    )


def test_notional_exactly_at_the_minimum_passes_after_rounding():
    rules = OrderRules(0.001, 0.001, 120.0, 50.0)
    limits = RiskLimits(1.0, 1e5, 1e6)
    ok = size_open(opened(notional=50.0, entry=50.0), "X", rules, limits, 0.0, 0)
    assert isinstance(ok, OrderIntent) and ok.quantity == 1.0
    just_under = size_open(opened(notional=49.99, entry=50.0), "X", rules, limits, 0.0, 0)
    assert isinstance(just_under, Rejection) and just_under.reason == "min_tutar"


def test_hard_caps_reject_instead_of_clipping():
    position = size_open(opened(notional=1000.0), "BTC", BTC_RULES, LIMITS, 0.0, 0)
    assert isinstance(position, Rejection) and position.reason == "pozisyon_tavanı"
    total = size_open(opened(notional=100.0), "BTC", BTC_RULES, LIMITS, 19_500.0, 0)
    assert isinstance(total, Rejection) and total.reason == "toplam_tavan"
    assert isinstance(
        size_open(opened(notional=100.0), "BTC", BTC_RULES, LIMITS, 19_000.0, 0), OrderIntent
    )


def test_allow_list_and_invalid_inputs():
    limits = RiskLimits(10.0, 5_000.0, 20_000.0, allowed=frozenset({"ETH"}))
    assert size_open(opened(), "BTC", BTC_RULES, limits, 0.0, 0).reason == "izinli_değil"
    for bad in (
        opened(entry=float("nan")),
        opened(entry=0.0),
        opened(notional=float("nan")),
        opened(notional=-5.0),
    ):
        assert isinstance(size_open(bad, "BTC", BTC_RULES, LIMITS, 0.0, 0), Rejection)


def test_close_uses_the_held_quantity_is_reduce_only_and_flips_the_side():
    event = SimpleNamespace(trade_id=7, side=1, exit=81_000.0, status=3, reason=2)
    intent = size_close(event, "BTC", 0.012, BTC_RULES, 5)
    assert isinstance(intent, OrderIntent) and intent.reduce_only
    assert (intent.side, intent.quantity, intent.client_id) == (SELL, 0.012, "al7c")
    assert intent.reason == "durum=3 sebep=2"
    short = size_close(
        SimpleNamespace(trade_id=8, side=-1, exit=1.0, status=1, reason=0),
        "BTC",
        -0.5,
        BTC_RULES,
        5,
    )
    assert short.side == BUY and short.quantity == 0.5
    assert size_close(event, "BTC", 0.0, BTC_RULES, 5).reason == "kapatılacak_miktar_yok"


def test_client_ids_are_deterministic_distinct_and_short():
    assert client_id(123, "open") == client_id(123, "open") != client_id(123, "close")
    assert client_id(123, "open") != client_id(124, "open")
    assert all(len(client_id(10**9, kind)) <= 36 for kind in ("open", "close"))
    assert all(ch.isalnum() for ch in client_id(5, "open"))


def test_order_side_table():
    assert [order_side(s, k) for s in (1, -1) for k in ("open", "close")] == [BUY, SELL, SELL, BUY]


def test_limits_validate_their_arguments():
    for bad in ((0.0, 1.0, 2.0), (1.0, 0.0, 2.0), (1.0, 3.0, 2.0)):
        with pytest.raises(ValueError):
            RiskLimits(*bad)


def test_every_event_of_a_real_run_becomes_a_valid_intent_or_a_recorded_rejection():
    universe = random_universe(10, 96 * 12, 6)
    panel, trade = stage_params("S17")
    system = StepSystem(universe.n, panel, trade, "history", first_day_of(universe), False, True)
    for i in range(universe.bars):
        system.step(
            universe.time[i],
            universe.open[i],
            universe.high[i],
            universe.low[i],
            universe.close[i],
            universe.volume[i],
        )
    events = system.events.ordered()
    opens = [e for e in events if isinstance(e, OpenEvent)]
    closes = {e.trade_id: e for e in events if isinstance(e, CloseEvent)}
    assert len(opens) > 5
    rules = OrderRules(0.01, 0.01, 1e6, 5.0)
    limits = RiskLimits(scale=40.0, max_position_usdt=1e6, max_total_usdt=1e7)
    accepted, rejected = {}, []
    for event in opens:
        result = size_open(event, universe.symbols[event.symbol], rules, limits, 0.0, 0)
        if isinstance(result, OrderIntent):
            accepted[event.trade_id] = result
        else:
            rejected.append(result)
    assert len(accepted) + len(rejected) == len(opens)
    for trade_id, intent in accepted.items():
        assert Decimal(str(intent.quantity)) % Decimal("0.01") == 0 and intent.notional >= 5.0
        if trade_id in closes:
            closing = size_close(closes[trade_id], intent.base, intent.quantity, rules, 1)
            assert closing.quantity == intent.quantity and closing.reduce_only
            assert {intent.side, closing.side} == {BUY, SELL}
    assert all(r.reason for r in rejected)


def test_stop_price_rounds_away_from_the_position_to_the_tick():
    rules = OrderRules(0.001, 0.001, 1000.0, 5.0, tick=0.5)
    limits = RiskLimits(1.0, 1000.0, 5000.0)

    def event(side):
        return SimpleNamespace(side=side, entry=100.0, dist=0.0123, trade_id=1)

    assert stop_price(event(1), rules, limits) == 95.0
    assert stop_price(event(-1), rules, limits) == 105.0
    assert stop_price(SimpleNamespace(side=1, entry=100.0, dist=float("nan")), rules, limits) == 0.0
    assert stop_price(event(1), rules, RiskLimits(1.0, 1000.0, 5000.0, stop_r=0)) == 0.0
    assert stop_client_id(7) == "al7s"
