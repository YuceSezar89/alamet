"""REST istemcisi: ban süresi, ban sırasında istek atmama, bütçe, yeniden deneme, kapanmış mum."""

import asyncio

import pytest

from alamet.exchange import BinanceError, BinanceRest, ban_until, kline_weight

NOW = 1_800_000_000.0
KLINE = [1, "10", "12", "9", "11", "5", 2, "50", 3, "1", "1", "0"]


class Clock:
    def __init__(self, now=NOW):
        self.now = now
        self.slept = []

    def time(self):
        return self.now

    async def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def client(responses, clock=None, **kw):
    clock = clock or Clock()
    log = []
    queue = list(responses)

    async def request(url, params):
        log.append((clock.now, url, dict(params)))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    rest = BinanceRest(
        request=request, clock=clock.time, sleep=clock.sleep, retry_delay_s=1.0, **kw
    )
    return rest, clock, log


def kline(open_time, close_time):
    return [open_time, "10", "12", "9", "11", "5", close_time, "50", 3, "1", "1", "0"]


def test_kline_weight_follows_binance_tiers():
    assert [kline_weight(n) for n in (1, 99, 100, 499, 500, 1000, 1001)] == [1, 1, 2, 2, 5, 5, 10]


def test_ban_time_comes_from_message_then_retry_after_then_defaults():
    until = int((NOW + 300) * 1000)
    assert ban_until(418, {}, {"msg": f"IP banned until {until}."}, NOW) == NOW + 300
    assert ban_until(429, {"Retry-After": "45"}, {}, NOW) == NOW + 45
    assert ban_until(429, {"retry-after": "x"}, None, NOW) == NOW + 60
    assert ban_until(418, {}, None, NOW) == NOW + 120


def test_only_closed_candles_are_returned_with_cache_columns():
    rest, clock, _ = client(
        [(200, {}, [kline(1000, int(NOW * 1000) - 1), kline(2000, int(NOW * 1000) + 5)])]
    )
    frame = asyncio.run(rest.klines("BTCUSDT", "15m", 2, start_ms=500))
    assert list(frame.columns) == [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ]
    assert frame.open_time.tolist() == [1000] and frame.close.iloc[0] == 11.0
    assert frame.quote_volume.iloc[0] == 50.0


def test_empty_answer_gives_an_empty_frame():
    rest, *_ = client([(200, {}, [])])
    assert asyncio.run(rest.klines("BTCUSDT", "15m")).empty


def test_ban_blocks_every_request_until_it_expires():
    until = int((NOW + 100) * 1000)
    rest, clock, log = client(
        [(418, {}, {"msg": f"banned until {until}"}), (200, {}, [kline(1, 2)])]
    )
    frame = asyncio.run(rest.klines("BTCUSDT", "15m"))
    assert len(frame) == 1 and len(log) == 2
    assert log[1][0] >= NOW + 100 and rest.banned is False
    assert clock.slept[0] == pytest.approx(100.0)


def test_api_error_code_raises_with_the_code():
    rest, *_ = client([(400, {}, {"code": -1121, "msg": "Invalid symbol."})])
    with pytest.raises(BinanceError) as info:
        asyncio.run(rest.klines("XXX", "15m"))
    assert info.value.code == -1121


def test_network_and_server_errors_are_retried_then_given_up():
    rest, clock, log = client([OSError("x"), (503, {}, None), (200, {}, [kline(1, 2)])])
    assert len(asyncio.run(rest.klines("BTCUSDT", "15m"))) == 1 and len(log) == 3
    failing, *_ = client([OSError("a")] * 4)
    with pytest.raises(BinanceError, match="başarısız"):
        asyncio.run(failing.klines("BTCUSDT", "15m"))


def test_budget_makes_requests_wait_for_the_window_to_roll():
    rest, clock, log = client([(200, {}, [])] * 4, budget=3)

    async def burst():
        for _ in range(4):
            await rest.klines("BTCUSDT", "15m", 10)

    asyncio.run(burst())
    assert [round(t - NOW) for t, _, _ in log] == [0, 0, 0, 60]
