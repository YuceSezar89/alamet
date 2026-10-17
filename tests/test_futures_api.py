"""İmzalı Binance istemcisi: imza, adres, güvenlik anahtarı, hata eşleme, belirsiz emir."""

import asyncio
from urllib.parse import parse_qs, urlparse

import pytest

from alamet.futures_api import (
    ApiError,
    Credentials,
    FuturesClient,
    LiveNotAllowed,
    MissingCredentials,
    OrderUncertain,
    decimal_text,
    load_credentials,
    parse_order,
    read_env_file,
    sign,
)
from alamet.orders import OrderIntent

CREDS = Credentials("KEYVALUE123", "SECRETVALUE456")
DOC_SECRET = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
DOC_QUERY = (
    "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1"
    "&recvWindow=5000&timestamp=1499827319559"
)


class Wire:
    """Sahte ağ: gönderilen istekleri kaydeder, sıradaki yanıtı (ya da hatayı) verir."""

    def __init__(self, *responses):
        self.responses, self.sent = list(responses), []

    async def __call__(self, method, url, headers):
        self.sent.append((method, url, headers))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(*responses, clock=lambda: 1_800_000_000.0):
    wire = Wire(*responses)
    return FuturesClient("demo", CREDS, wire, clock), wire


def intent(reduce_only=False, quantity=0.012, client_id="al7o"):
    return OrderIntent(
        "open", "BTC", "BTCUSDT", "BUY", quantity, 80_000.0, 7, 0, client_id, reduce_only
    )


def run(coro):
    return asyncio.run(coro)


def test_signature_matches_the_example_in_the_binance_documentation():
    assert sign(DOC_QUERY, DOC_SECRET) == (
        "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"
    )


def test_signed_urls_append_window_timestamp_and_a_valid_signature_last():
    api, _ = client()
    url = api.build_url("/fapi/v1/order", {"symbol": "BTCUSDT", "side": "BUY", "skip": None}, True)
    parts = urlparse(url)
    assert parts.path == "/fapi/v1/order" and url.startswith("https://demo-fapi.binance.com")
    query = parts.query
    assert query.endswith(f"&signature={sign(query.rsplit('&signature=', 1)[0], 'SECRETVALUE456')}")
    fields = parse_qs(query)
    assert fields["recvWindow"] == ["5000"] and fields["timestamp"] == ["1800000000000"]
    assert "skip" not in fields
    assert query.index("timestamp") < query.index("signature")
    public = api.build_url("/fapi/v1/time", None, False)
    assert public == "https://demo-fapi.binance.com/fapi/v1/time"


def test_credentials_never_show_in_repr_or_str():
    assert "KEYVALUE123" not in repr(CREDS) and "SECRETVALUE456" not in repr(CREDS)
    assert "SECRETVALUE456" not in str(CREDS) and "SECRETVALUE456" not in repr(
        client()[0].credentials
    )


def test_env_file_parsing_and_missing_keys(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# yorum\nDEPV2_TESTNET_API_KEY=abc\nDEPV2_TESTNET_API_SECRET=def\nDEPV2_LIVE_API_KEY=\n"
    )
    assert read_env_file(path)["DEPV2_TESTNET_API_KEY"] == "abc"
    assert load_credentials("demo", path).key == "abc"
    with pytest.raises(MissingCredentials):
        load_credentials("live", path)


def test_live_environment_needs_the_explicit_flag(monkeypatch):
    monkeypatch.delenv("ALAMET_ALLOW_LIVE", raising=False)
    with pytest.raises(LiveNotAllowed):
        FuturesClient("live", CREDS)
    monkeypatch.setenv("ALAMET_ALLOW_LIVE", "evet")
    with pytest.raises(LiveNotAllowed):
        FuturesClient("live", CREDS)
    monkeypatch.setenv("ALAMET_ALLOW_LIVE", "EVET")
    assert FuturesClient("live", CREDS).base == "https://fapi.binance.com"
    with pytest.raises(ValueError):
        FuturesClient("başka", CREDS)


def test_api_errors_and_http_failures_are_raised_with_the_code():
    api, _ = client((400, {"code": -2022, "msg": "ReduceOnly Order is rejected."}))
    with pytest.raises(ApiError) as info:
        run(api.call("POST", "/fapi/v1/order", {}))
    assert info.value.code == -2022
    api, _ = client((503, None))
    with pytest.raises(ApiError):
        run(api.call("GET", "/x", signed=False))


def test_time_sync_shifts_the_signed_timestamp():
    api, wire = client((200, {"serverTime": 1_800_000_002_000}), (200, []))
    assert run(api.sync_time()) == 2000
    run(api.call("GET", "/fapi/v2/balance"))
    assert "timestamp=1800000002000" in wire.sent[1][1]
    assert wire.sent[1][2]["X-MBX-APIKEY"] == "KEYVALUE123"


def test_account_helpers_parse_the_answers():
    api, _ = client(
        (200, [{"asset": "BNB", "balance": "1", "availableBalance": "1"},
               {"asset": "USDT", "balance": "5000.5", "availableBalance": "4900", "crossUnPnl": "1.5"}]),
        (200, {"dualSidePosition": False}),
        (200, {"multiAssetsMargin": True}),
        (200, [{"symbol": "BTCUSDT", "positionAmt": "-0.012"}, {"symbol": "ETHUSDT", "positionAmt": "0.000"}]),
        (200, {"symbol": "BTCUSDT", "markPrice": "80123.4"}),
    )  # fmt: skip
    assert run(api.balance()) == {"balance": 5000.5, "available": 4900.0, "cross_unrealized": 1.5}
    assert run(api.position_mode_is_one_way()) is True
    assert run(api.multi_assets_mode()) is True
    assert run(api.positions()) == {"BTCUSDT": -0.012}
    assert run(api.mark_price("BTCUSDT")) == 80123.4


def test_margin_type_already_set_is_fine_but_other_errors_are_not():
    api, _ = client((400, {"code": -4046, "msg": "No need to change margin type."}))
    run(api.set_margin_type("BTCUSDT"))
    api, _ = client((400, {"code": -4047, "msg": "Margin type cannot be changed."}))
    with pytest.raises(ApiError):
        run(api.set_margin_type("BTCUSDT"))


def test_quantity_text_has_no_scientific_notation_or_trailing_zeros():
    assert decimal_text(0.00001) == "0.00001" and decimal_text(1e-05) == "0.00001"
    assert (
        decimal_text(5.0) == "5"
        and decimal_text(0.012) == "0.012"
        and decimal_text(1234.5) == "1234.5"
    )


def test_order_parameters_are_market_idempotent_and_reduce_only_when_closing():
    api, wire = client((200, {"orderId": 1, "clientOrderId": "al7o", "symbol": "BTCUSDT", "side": "BUY",
                              "status": "FILLED", "executedQty": "0.012", "avgPrice": "80010.5", "cumQuote": "960.126"}),
                       (200, {"orderId": 2, "clientOrderId": "al7c", "status": "FILLED", "reduceOnly": True}))  # fmt: skip
    result = run(api.place_order(intent()))
    assert result.filled and result.executed_qty == 0.012 and result.avg_price == 80010.5
    query = parse_qs(urlparse(wire.sent[0][1]).query)
    assert query["type"] == ["MARKET"] and query["newClientOrderId"] == ["al7o"]
    assert query["newOrderRespType"] == ["RESULT"] and "reduceOnly" not in query
    run(api.place_order(intent(reduce_only=True, client_id="al7c")))
    assert parse_qs(urlparse(wire.sent[1][1]).query)["reduceOnly"] == ["true"]


def test_rejected_orders_raise_and_are_not_retried():
    api, wire = client((400, {"code": -2022, "msg": "ReduceOnly Order is rejected."}))
    with pytest.raises(ApiError):
        run(api.place_order(intent(reduce_only=True)))
    assert len(wire.sent) == 1


def test_an_unanswered_order_is_looked_up_by_client_id_and_never_sent_twice(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", lambda _s: _noop())
    order = {"orderId": 9, "clientOrderId": "al7o", "symbol": "BTCUSDT", "side": "BUY", "status": "FILLED", "executedQty": "0.012", "avgPrice": "1", "cumQuote": "1"}  # fmt: skip
    api, wire = client(asyncio.TimeoutError(), (200, order))
    result = run(api.place_order(intent()))
    assert result.order_id == 9 and result.filled
    assert [m for m, _, _ in wire.sent] == ["POST", "GET"]


def test_an_order_that_cannot_be_confirmed_is_reported_as_uncertain(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", lambda _s: _noop())
    missing = (400, {"code": -2013, "msg": "Order does not exist."})
    api, wire = client(OSError("kopuk"), missing, missing, missing)
    with pytest.raises(OrderUncertain):
        run(api.place_order(intent()))
    assert [m for m, _, _ in wire.sent].count("POST") == 1


async def _noop():
    return None


def test_fills_sum_commission_and_realized_pnl():
    api, _ = client(
        (
            200,
            [
                {"commission": "0.2", "realizedPnl": "1.5", "price": "1", "qty": "1"},
                {"commission": "0.1", "realizedPnl": "-0.5", "price": "1", "qty": "1"},
            ],
        )
    )
    summary = run(api.fills("BTCUSDT", 9))
    assert summary["commission"] == pytest.approx(0.3)
    assert summary["realized_pnl"] == pytest.approx(1.0) and summary["trades"] == 2.0


def test_parse_order_tolerates_missing_fields():
    result = parse_order({"orderId": 5, "status": "NEW"})
    assert (result.order_id, result.status, result.executed_qty, result.filled) == (
        5,
        "NEW",
        0.0,
        False,
    )


def test_a_market_order_answer_with_zero_average_price_falls_back_to_cumulative_quote():
    result = parse_order(
        {
            "orderId": 1,
            "status": "FILLED",
            "executedQty": "0.002",
            "avgPrice": "0.00000",
            "cumQuote": "160.5",
        }
    )
    assert result.avg_price == pytest.approx(80250.0)
    unknown = parse_order(
        {"orderId": 1, "status": "FILLED", "executedQty": "0.002", "avgPrice": "0", "cumQuote": "0"}
    )
    assert unknown.avg_price == 0.0


def test_fills_give_the_real_average_price_from_the_trades():
    api, _ = client((200, [{"commission": "0.1", "realizedPnl": "0", "price": "100", "qty": "1"},
                           {"commission": "0.1", "realizedPnl": "0", "price": "110", "qty": "3"}]))  # fmt: skip
    summary = run(api.fills("BTCUSDT", 5))
    assert summary["quantity"] == 4.0 and summary["avg_price"] == pytest.approx(107.5)
