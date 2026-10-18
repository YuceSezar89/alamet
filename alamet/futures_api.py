"""
Binance USDT-M Futures imzalı istemcisi (emir, hesap, pozisyon).

Güvenlik:
  - Varsayılan ortam DEMO. Gerçek ortam (`live`) yalnızca süreç ortamında `ALAMET_ALLOW_LIVE=EVET`
    varsa açılır; aksi halde `LiveNotAllowed`.
  - Anahtarlar `.env` dosyasından okunur ve hiçbir zaman yazdırılmaz (`Credentials.__repr__` gizler).
  - Emir idempotenttir: `newClientOrderId` işlemden türetilir. Yanıt belirsiz kalırsa (zaman aşımı)
    emir kimliğiyle sorgulanır; ikinci kez gönderilmez.

İstek işlevi ve saat dışarıdan değiştirilebilir (ağsız test).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

URLS = {
    "demo": "https://demo-fapi.binance.com",
    "testnet": "https://testnet.binancefuture.com",
    "live": "https://fapi.binance.com",
}
KEY_NAMES = {
    "demo": ("DEPV2_TESTNET_API_KEY", "DEPV2_TESTNET_API_SECRET"),
    "testnet": ("DEPV2_TESTNET_API_KEY", "DEPV2_TESTNET_API_SECRET"),
    "live": ("DEPV2_LIVE_API_KEY", "DEPV2_LIVE_API_SECRET"),
}
RECV_WINDOW = 5000
LIVE_FLAG = "ALAMET_ALLOW_LIVE"
LIVE_ANSWER = "EVET"
FILLED = "FILLED"
TIMESTAMP_ERROR = -1021
STOP_DONE = ("TRIGGERED", "FINISHED")
UNKNOWN_ALGO = (-2011, -2013)
DUPLICATE_ID = -4116

Transport = Callable[[str, str, dict[str, str]], Awaitable[tuple[int, Any]]]


class LiveNotAllowed(RuntimeError):
    """Gerçek ortam açık onay olmadan istendi."""


class MissingCredentials(RuntimeError):
    pass


class ApiError(RuntimeError):
    def __init__(self, code: int, message: str, status: int = 0) -> None:
        super().__init__(f"Binance {code}: {message}")
        self.code, self.status = code, status


class OrderUncertain(RuntimeError):
    """Emrin borsada oluşup oluşmadığı doğrulanamadı: elle bakılmalı, kör yeniden gönderme yok."""


@dataclass(frozen=True)
class Credentials:
    key: str = field(repr=False)
    secret: str = field(repr=False)

    def __repr__(self) -> str:
        return "Credentials(<gizli>)"


@dataclass(frozen=True)
class OrderResult:
    client_id: str
    order_id: int
    symbol: str
    side: str
    status: str
    executed_qty: float
    avg_price: float
    quote_qty: float
    reduce_only: bool
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def filled(self) -> bool:
        return self.status == FILLED


def read_env_file(path: Path | str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip()
    return values


def load_credentials(environment: str, path: Path | str = ".env") -> Credentials:
    key_name, secret_name = KEY_NAMES[environment]
    values = read_env_file(path)
    key, secret = values.get(key_name, ""), values.get(secret_name, "")
    if not key or not secret:
        raise MissingCredentials(f"{environment} anahtarları {path} içinde tanımlı değil")
    return Credentials(key, secret)


def sign(query: str, secret: str) -> str:
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


def decimal_text(value: float) -> str:
    """Bilimsel gösterimsiz, gereksiz sıfırsız sayı metni (miktar/fiyat parametresi)."""
    return format(Decimal(str(value)).normalize(), "f")


async def aiohttp_transport(method: str, url: str, headers: dict[str, str]) -> tuple[int, Any]:
    import aiohttp  # pylint: disable=import-outside-toplevel
    from yarl import URL  # pylint: disable=import-outside-toplevel

    async with aiohttp.ClientSession() as session:
        async with session.request(
            method, URL(url, encoded=True), headers=headers, timeout=aiohttp.ClientTimeout(total=15)
        ) as response:
            try:
                body = await response.json(content_type=None)
            except ValueError:
                body = None
            return response.status, body


class FuturesClient:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        environment: str = "demo",
        credentials: Credentials | None = None,
        transport: Transport = aiohttp_transport,
        clock: Callable[[], float] = time.time,
        env_path: Path | str = ".env",
    ) -> None:
        if environment not in URLS:
            raise ValueError(f"bilinmeyen ortam: {environment}")
        if environment == "live" and os.environ.get(LIVE_FLAG) != LIVE_ANSWER:
            raise LiveNotAllowed(f"Gerçek ortam için {LIVE_FLAG}={LIVE_ANSWER} gerekir")
        self.environment, self.base = environment, URLS[environment]
        self.credentials = credentials or load_credentials(environment, env_path)
        self.transport, self.clock = transport, clock
        self.offset_ms = 0

    # ---------------------------------------------------------------- çekirdek
    def _timestamp(self) -> int:
        return int(self.clock() * 1000) + self.offset_ms

    def build_url(self, path: str, params: dict[str, Any] | None, signed: bool) -> str:
        query = {k: v for k, v in (params or {}).items() if v is not None}
        if signed:
            query["recvWindow"] = RECV_WINDOW
            query["timestamp"] = self._timestamp()
        text = urlencode(query)
        if signed:
            text += f"&signature={sign(text, self.credentials.secret)}"
        return f"{self.base}{path}" + (f"?{text}" if text else "")

    async def call(
        self, method: str, path: str, params: dict[str, Any] | None = None, signed: bool = True
    ) -> Any:
        """İstek; saat penceresi hatasında (−1021) saat farkı yenilenip BİR kez tekrar denenir."""
        try:
            return await self._call_once(method, path, params, signed)
        except ApiError as exc:
            if exc.code != TIMESTAMP_ERROR or not signed:
                raise
            await self.sync_time()
            return await self._call_once(method, path, params, signed)

    async def _call_once(
        self, method: str, path: str, params: dict[str, Any] | None, signed: bool
    ) -> Any:
        url = self.build_url(path, params, signed)
        headers = {"X-MBX-APIKEY": self.credentials.key}
        status, body = await self.transport(method, url, headers)
        if isinstance(body, dict) and "code" in body and isinstance(body["code"], int):
            if body["code"] < 0:
                raise ApiError(body["code"], str(body.get("msg", "")), status)
        if status >= 400:
            raise ApiError(-status, f"HTTP {status}", status)
        return body

    async def sync_time(self) -> int:
        """Yerel saat ile borsa saati farkını ölçer (imzalı isteklerin zaman penceresi için)."""
        body = await self.call("GET", "/fapi/v1/time", signed=False)
        self.offset_ms = int(body["serverTime"]) - int(self.clock() * 1000)
        return self.offset_ms

    # ----------------------------------------------------------------- hesap
    async def balance(self, asset: str = "USDT") -> dict[str, float]:
        rows = await self.call("GET", "/fapi/v2/balance")
        row = next((r for r in rows if r["asset"] == asset), None)
        if row is None:
            return {"balance": 0.0, "available": 0.0, "cross_unrealized": 0.0}
        return {
            "balance": float(row["balance"]),
            "available": float(row["availableBalance"]),
            "cross_unrealized": float(row.get("crossUnPnl", 0.0)),
        }

    async def position_mode_is_one_way(self) -> bool:
        body = await self.call("GET", "/fapi/v1/positionSide/dual")
        return not bool(body["dualSidePosition"])

    async def multi_assets_mode(self) -> bool:
        body = await self.call("GET", "/fapi/v1/multiAssetsMargin")
        return bool(body["multiAssetsMargin"])

    async def positions(self) -> dict[str, float]:
        """Açık pozisyonlar: {sembol: işaretli miktar} (uzun +, kısa −)."""
        rows = await self.call("GET", "/fapi/v2/positionRisk")
        return {
            r["symbol"]: float(r["positionAmt"]) for r in rows if float(r["positionAmt"]) != 0.0
        }

    async def mark_price(self, symbol: str) -> float:
        body = await self.call("GET", "/fapi/v1/premiumIndex", {"symbol": symbol}, signed=False)
        return float(body["markPrice"])

    async def set_leverage(self, symbol: str, leverage: int) -> int:
        body = await self.call(
            "POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}
        )
        return int(body["leverage"])

    async def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> None:
        """Marj türünü ayarlar; zaten o türdeyse (−4046) sorun değil."""
        try:
            await self.call(
                "POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": margin_type}
            )
        except ApiError as exc:
            if exc.code != -4046:
                raise

    # ------------------------------------------------------------------ emir
    def _order_params(self, intent) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": intent.symbol,
            "side": intent.side,
            "type": "MARKET",
            "quantity": decimal_text(intent.quantity),
            "newClientOrderId": intent.client_id,
            "newOrderRespType": "RESULT",
        }
        if intent.reduce_only:
            params["reduceOnly"] = "true"
        return params

    async def test_order(self, intent) -> None:
        """Emri göndermeden borsada doğrulatır (kural/yuvarlama hataları için)."""
        await self.call("POST", "/fapi/v1/order/test", self._order_params(intent))

    async def place_order(self, intent, retries: int = 3, pause_s: float = 0.5) -> OrderResult:
        """Piyasa emri. Ağ yanıtı belirsizse emri kimliğiyle sorgular, ikinci kez GÖNDERMEZ."""
        try:
            body = await self.call("POST", "/fapi/v1/order", self._order_params(intent))
            return parse_order(body)
        except (OSError, asyncio.TimeoutError) as exc:
            for _ in range(retries):
                await asyncio.sleep(pause_s)
                try:
                    return await self.query_order(intent.symbol, intent.client_id)
                except ApiError as lookup:
                    if lookup.code != -2013:
                        raise
                except (OSError, asyncio.TimeoutError):
                    continue
            raise OrderUncertain(
                f"{intent.client_id}: gönderim yanıtı alınamadı ve sorgulanamadı ({exc!r})"
            ) from exc

    # ------------------------------------------------------- felaket stopu
    async def place_stop(
        self, symbol: str, side: str, trigger_price: float, client_algo_id: str
    ) -> dict:
        """Pozisyonu tamamen kapatan koşullu piyasa stopu (`closePosition`, işaret fiyatı).
        Aynı kimlik zaten varsa (−4116) yeniden gönderim sayılır, hata değil."""
        params = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "triggerPrice": decimal_text(trigger_price),
            "closePosition": "true",
            "workingType": "MARK_PRICE",
            "clientAlgoId": client_algo_id,
        }
        try:
            return await self.call("POST", "/fapi/v1/algoOrder", params)
        except ApiError as exc:
            if exc.code != DUPLICATE_ID:
                raise
            return {"clientAlgoId": client_algo_id, "algoStatus": "DUPLICATE"}

    async def cancel_stop(self, client_algo_id: str) -> bool:
        """Stopu iptal eder; zaten yoksa/bitmişse False."""
        try:
            await self.call("DELETE", "/fapi/v1/algoOrder", {"clientAlgoId": client_algo_id})
        except ApiError as exc:
            if exc.code in UNKNOWN_ALGO:
                return False
            raise
        return True

    async def stop_status(self, client_algo_id: str) -> str | None:
        """Stopun durumu (NEW, CANCELED, TRIGGERED, FINISHED ...); borsa tanımıyorsa None."""
        try:
            body = await self.call("GET", "/fapi/v1/algoOrder", {"clientAlgoId": client_algo_id})
        except ApiError as exc:
            if exc.code in UNKNOWN_ALGO:
                return None
            raise
        return str(body.get("algoStatus", "")) or None

    async def open_stops(self) -> set[str]:
        """Borsada bekleyen stopların `clientAlgoId` kümesi."""
        rows = await self.call("GET", "/fapi/v1/openAlgoOrders")
        return {str(r["clientAlgoId"]) for r in rows}

    async def query_order(self, symbol: str, client_id: str) -> OrderResult:
        body = await self.call(
            "GET", "/fapi/v1/order", {"symbol": symbol, "origClientOrderId": client_id}
        )
        return parse_order(body)

    async def fills(self, symbol: str, order_id: int) -> dict[str, float]:
        """Emrin dolumları: toplam komisyon (USDT cinsinden varsayılır) ve gerçekleşen K/Z."""
        rows = await self.call(
            "GET", "/fapi/v1/userTrades", {"symbol": symbol, "orderId": order_id}
        )
        quantity = sum(float(r["qty"]) for r in rows)
        quote = sum(float(r["price"]) * float(r["qty"]) for r in rows)
        return {
            "commission": sum(float(r["commission"]) for r in rows),
            "realized_pnl": sum(float(r.get("realizedPnl", 0.0)) for r in rows),
            "trades": float(len(rows)),
            "quantity": quantity,
            "avg_price": quote / quantity if quantity > 0 else 0.0,
        }


def parse_order(body: dict) -> OrderResult:
    """Emir yanıtı. Piyasa emrinde `avgPrice` yanıt anında 0 gelebilir: o zaman `cumQuote` /
    `executedQty` kullanılır; ikisi de yoksa fiyat 0 kalır ve dolumlardan (`fills`) alınmalıdır."""
    executed = float(body.get("executedQty", 0.0))
    quote = float(body.get("cumQuote", 0.0))
    average = float(body.get("avgPrice", 0.0))
    if average <= 0 and executed > 0 and quote > 0:
        average = quote / executed
    return OrderResult(
        client_id=str(body.get("clientOrderId", "")),
        order_id=int(body.get("orderId", 0)),
        symbol=str(body.get("symbol", "")),
        side=str(body.get("side", "")),
        status=str(body.get("status", "")),
        executed_qty=executed,
        avg_price=average,
        quote_qty=quote,
        reduce_only=bool(body.get("reduceOnly", False)),
        raw=body,
    )


async def health_check(api: FuturesClient) -> dict[str, Any]:
    """Salt-okunur sağlık kontrolü: saat farkı, bakiye, pozisyon modu, açık pozisyonlar."""
    offset = await api.sync_time()
    return {
        "ortam": api.environment,
        "saat_farkı_ms": offset,
        "bakiye": await api.balance(),
        "tek_yönlü_pozisyon": await api.position_mode_is_one_way(),
        "çoklu_varlık_modu": await api.multi_assets_mode(),
        "açık_pozisyon": await api.positions(),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse  # pylint: disable=import-outside-toplevel
    import json  # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser(description="Binance Futures bağlantı kontrolü (salt okunur)")
    parser.add_argument("command", choices=["check"])
    parser.add_argument("--env", choices=sorted(URLS), default="demo")
    args = parser.parse_args(argv)
    api = FuturesClient(args.env)
    print(json.dumps(asyncio.run(health_check(api)), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
