"""
Binance Futures REST istemcisi (herkese açık uçlar): ban koruması, istek bütçesi, yeniden deneme.

Eski sistemden alınan dersler: 418/429 yanıtında ban süresi `banned until <ms>` mesajından, yoksa
`Retry-After` başlığından, o da yoksa sabit bekleme (418: 120 sn, 429: 60 sn) olarak okunur ve
ban sürerken ağa HİÇ istek çıkmaz (ban sırasında istek atmak süreyi uzatır). Endpoint ağırlığı
sliding-window bütçesiyle sınırlanır (gerçek limit 2400/dk; varsayılan bütçe yarısı). Kapanmamış mum
döndürülmez.

İstek işlevi (`request`) ve saat dışarıdan değiştirilebilir; böylece mantık ağsız test edilir.
Bu istemci başka bir süreçle bütçe paylaşmaz: aynı IP'den birden çok süreç çalıştırırken toplam
bütçeyi bölüştürmek çağıranın işidir.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

import pandas as pd

BASE_URL = "https://fapi.binance.com/fapi/v1"
WINDOW_S = 60.0
DEFAULT_BUDGET = 1200
MAX_ATTEMPTS = 4
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
    "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]  # fmt: skip
CACHE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "quote_volume"]

Request = Callable[[str, dict[str, Any]], Awaitable[tuple[int, dict[str, str], Any]]]


class BinanceError(RuntimeError):
    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


def kline_weight(limit: int) -> int:
    """`/klines` isteğinin ağırlığı (Binance belgesi)."""
    if limit < 100:
        return 1
    if limit < 500:
        return 2
    return 5 if limit <= 1000 else 10


def ban_until(status: int, headers: dict[str, str], body: Any, now: float) -> float:
    """418/429 yanıtından banın biteceği an (epoch sn)."""
    message = body.get("msg", "") if isinstance(body, dict) else ""
    match = re.search(r"banned until (\d+)", message)
    if match:
        return int(match.group(1)) / 1000
    retry = {key.lower(): value for key, value in headers.items()}.get("retry-after")
    if retry:
        try:
            return now + float(retry)
        except ValueError:
            pass
    return now + (120 if status == 418 else 60)


async def aiohttp_request(url: str, params: dict[str, Any]) -> tuple[int, dict[str, str], Any]:
    import aiohttp  # pylint: disable=import-outside-toplevel

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as r:
            try:
                body = await r.json(content_type=None)
            except ValueError:
                body = None
            return r.status, dict(r.headers), body


class BinanceRest:  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        base_url: str = BASE_URL,
        request: Request = aiohttp_request,
        budget: int = DEFAULT_BUDGET,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        retry_delay_s: float = 2.0,
    ) -> None:
        self.base_url, self.request, self.budget = base_url, request, budget
        self.clock, self.sleep, self.retry_delay_s = clock, sleep, retry_delay_s
        self.banned_until = 0.0
        self.log: Callable[[str], None] = lambda _text: None
        self._spent: deque[tuple[float, int]] = deque()

    @property
    def banned(self) -> bool:
        return self.clock() < self.banned_until

    async def _throttle(self, weight: int) -> None:
        """Ban bitene ve pencere bütçesi yer açana kadar bekler."""
        while True:
            now = self.clock()
            if now < self.banned_until:
                self.log(f"REST: IP bekleme süresi, {self.banned_until - now:.0f} sn bekleniyor")
                await self.sleep(self.banned_until - now)
                continue
            while self._spent and self._spent[0][0] <= now - WINDOW_S:
                self._spent.popleft()
            used = sum(w for _, w in self._spent)
            if used + weight <= self.budget or not self._spent:
                self._spent.append((now, weight))
                return
            wait = max(self._spent[0][0] + WINDOW_S - now, 0.05)
            if wait > 2:
                self.log(f"REST: istek bütçesi dolu, {wait:.0f} sn bekleniyor")
            await self.sleep(wait)

    async def get_json(self, path: str, params: dict[str, Any], weight: int = 1) -> Any:
        url = f"{self.base_url}/{path}"
        error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            await self._throttle(weight)
            try:
                status, headers, body = await self.request(url, params)
            except (OSError, asyncio.TimeoutError) as exc:
                error = exc
                self.log(f"REST: {path} deneme {attempt + 1} başarısız ({type(exc).__name__})")
                await self.sleep(self.retry_delay_s * (attempt + 1))
                continue
            if status in (418, 429):
                self.log(f"REST: HTTP {status}")
                self.banned_until = max(
                    self.banned_until, ban_until(status, headers, body, self.clock())
                )
                error = BinanceError(f"HTTP {status}: IP bekleme süresi", status)
                continue
            if isinstance(body, dict) and body.get("code"):
                raise BinanceError(f"Binance {body['code']}: {body.get('msg', '')}", body["code"])
            if status >= 500:
                error = BinanceError(f"HTTP {status}", status)
                await self.sleep(self.retry_delay_s * (attempt + 1))
                continue
            if status >= 400:
                raise BinanceError(f"HTTP {status}", status)
            return body
        raise BinanceError(f"{path}: istek başarısız ({error})")

    async def klines(
        self, symbol: str, interval: str, limit: int = 500, start_ms: int | None = None
    ) -> pd.DataFrame:
        """Kapanmış mumlar, önbellek sütunlarıyla (`open_time` ms, OHLC, hacimler)."""
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        data = await self.get_json("klines", params, kline_weight(limit))
        if not data:
            return pd.DataFrame({column: [] for column in CACHE_COLUMNS}).astype(
                {"open_time": "int64"}
            )
        frame = pd.DataFrame(data, columns=KLINE_COLUMNS)
        frame = frame[frame["close_time"].astype("int64") < int(self.clock() * 1000)]
        out = frame[CACHE_COLUMNS].copy()
        out["open_time"] = out["open_time"].astype("int64")
        for column in CACHE_COLUMNS[1:]:
            out[column] = out[column].astype(float)
        return out.reset_index(drop=True)


_DEFAULT: BinanceRest | None = None


def default_client() -> BinanceRest:
    """Süreç genelinde tek istemci (ban durumu ve bütçe paylaşılır)."""
    global _DEFAULT  # pylint: disable=global-statement
    if _DEFAULT is None:
        _DEFAULT = BinanceRest()
    return _DEFAULT
