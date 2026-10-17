"""
Evren seçimi: hangi vadeli işlem sembolleri bota girer. Tek ayar nesnesi (`UniverseSpec`) ve dört
kullanım: hacme göre ilk N, elle seçilen liste, tüm market, kayıtlı liste dosyası.

Seçim mantığı ağdan bağımsız ve belirlenimcidir (aynı girdi → aynı sıra). Sıra önemlidir: sıralama
eşitliklerinde Pine ilk indeksi tutar, bu yüzden hacme göre azalan, eşitlikte sembol adına göre.

Binance verisi (`/fapi/v1/exchangeInfo` + `/fapi/v1/ticker/24hr`) yalnızca `fetch_market` içinde
okunur; ayrıştırma saf fonksiyonlardır. Seçilen evren `save_snapshot` ile dosyaya yazılırsa bot
yeniden başlarken aynı listeyi (ve lot/tick/minimum tutar kurallarını) kullanır.

Uyarı (yanlılık): bugünün 24 saatlik hacmiyle seçilen evrenle geçmiş test, hayatta kalan sembolleri
seçmiş olur (survivorship). Araştırmada bunun etkisini ayrıca ölçmek gerekir.
"""

# pylint: disable=too-many-instance-attributes
from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

MODES = ("top_n", "list", "all")
QUOTE = "USDT"
PERPETUAL = "PERPETUAL"
TRADING = "TRADING"
DEFAULT_UNDERLYING = ("COIN",)
MS_PER_DAY = 86_400_000

JsonFetcher = Callable[[str], Awaitable[object]]


@dataclass(frozen=True)
class SymbolInfo:
    """Bir vadeli işlem sözleşmesi: kimlik, hacim ve emir kuralları (lot, tick, minimum tutar)."""

    base: str
    symbol: str
    status: str
    contract_type: str
    quote: str
    underlying: str
    onboard_ms: int
    quote_volume: float
    tick_size: float
    step_size: float
    min_qty: float
    max_qty: float
    market_step_size: float
    market_min_qty: float
    market_max_qty: float
    min_notional: float


@dataclass(frozen=True)
class UniverseSpec:
    """Evren ayarı. `top_n`: hacme göre ilk `n`; `list`: `symbols` (taban adlar, sıra korunur);
    `all`: süzgeçlerden geçen her sembol."""

    mode: str = "top_n"
    n: int = 20
    symbols: tuple[str, ...] = ()
    min_quote_volume: float = 0.0
    min_age_days: int = 0
    exclude: tuple[str, ...] = ()
    underlying: tuple[str, ...] = DEFAULT_UNDERLYING

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"Evren modu {MODES} içinden olmalı: {self.mode}")
        if self.mode == "top_n" and self.n < 1:
            raise ValueError("top_n için n en az 1 olmalı")
        if self.mode == "list" and not self.symbols:
            raise ValueError("list modunda sembol listesi boş olamaz")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Sembol listesinde yinelenen sembol var")


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _filters(raw: Sequence[dict]) -> dict[str, dict]:
    return {item.get("filterType", ""): item for item in raw}


def parse_exchange_info(data: dict, volumes: dict[str, float] | None = None) -> list[SymbolInfo]:
    """`/fapi/v1/exchangeInfo` yanıtından sembol listesi. `volumes`: {sembol: 24s USDT hacmi}."""
    volumes = volumes or {}
    out = []
    for item in data.get("symbols", []):
        symbol = item["symbol"]
        flt = _filters(item.get("filters", []))
        price, lot = flt.get("PRICE_FILTER", {}), flt.get("LOT_SIZE", {})
        market = flt.get("MARKET_LOT_SIZE", lot)
        notional = flt.get("MIN_NOTIONAL", {})
        out.append(
            SymbolInfo(
                base=item.get("baseAsset") or symbol.removesuffix(QUOTE),
                symbol=symbol,
                status=item.get("status", ""),
                contract_type=item.get("contractType", ""),
                quote=item.get("quoteAsset", ""),
                underlying=item.get("underlyingType", ""),
                onboard_ms=int(item.get("onboardDate", 0) or 0),
                quote_volume=volumes.get(symbol, 0.0),
                tick_size=_float(price.get("tickSize")),
                step_size=_float(lot.get("stepSize")),
                min_qty=_float(lot.get("minQty")),
                max_qty=_float(lot.get("maxQty")),
                market_step_size=_float(market.get("stepSize")),
                market_min_qty=_float(market.get("minQty")),
                market_max_qty=_float(market.get("maxQty")),
                min_notional=_float(notional.get("notional")),
            )
        )
    return out


def parse_volumes(tickers: Sequence[dict]) -> dict[str, float]:
    """`/fapi/v1/ticker/24hr` yanıtından {sembol: quoteVolume}."""
    return {item["symbol"]: _float(item.get("quoteVolume")) for item in tickers}


def eligible(infos: Sequence[SymbolInfo], spec: UniverseSpec, now_ms: int) -> list[SymbolInfo]:
    """Süzgeçler: işlem görüyor, sürekli (perpetual), USDT, kripto (`COIN`), hariç tutulmamış, hacim
    ve yaş eşiği. Hisse, FX, emtia, endeks ve ön-piyasa sözleşmeleri varsayılan olarak dışarıda."""
    min_age = spec.min_age_days * MS_PER_DAY
    return [
        info
        for info in infos
        if info.status == TRADING
        and info.contract_type == PERPETUAL
        and info.quote == QUOTE
        and info.underlying in spec.underlying
        and info.base not in spec.exclude
        and info.quote_volume >= spec.min_quote_volume
        and now_ms - info.onboard_ms >= min_age
    ]


def select_universe(
    infos: Sequence[SymbolInfo], spec: UniverseSpec, now_ms: int | None = None
) -> list[SymbolInfo]:
    """Seçilen evren, bota verilecek sırayla. `list` modunda bulunamayan sembol hata verir."""
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    pool = eligible(infos, spec, now_ms)
    if spec.mode == "list":
        by_base = {info.base: info for info in pool}
        missing = [name for name in spec.symbols if name not in by_base]
        if missing:
            raise ValueError(
                f"Seçilemeyen semboller (işlemde değil ya da süzgeçte elendi): {missing}"
            )
        return [by_base[name] for name in spec.symbols]
    ranked = sorted(pool, key=lambda info: (-info.quote_volume, info.symbol))
    return ranked[: spec.n] if spec.mode == "top_n" else ranked


def bases(selected: Sequence[SymbolInfo]) -> tuple[str, ...]:
    """`Universe` ve veri önbelleği için taban adlar (ör. "BTC")."""
    return tuple(info.base for info in selected)


def save_snapshot(path: Path, selected: Sequence[SymbolInfo], spec: UniverseSpec) -> None:
    """Seçimi, ayarı ve emir kurallarını dosyaya yazar (tekrarlanabilirlik)."""
    payload = {
        "taken_at_ms": int(time.time() * 1000),
        "spec": asdict(spec),
        "symbols": [asdict(info) for info in selected],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def load_snapshot(path: Path) -> list[SymbolInfo]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [SymbolInfo(**item) for item in payload["symbols"]]


async def fetch_market(
    fetch_json: JsonFetcher | None = None,
) -> list[SymbolInfo]:
    """Binance'ten sözleşme kuralları ve 24 saatlik hacim (`fetch_json` test için değiştirilebilir)."""
    if fetch_json is None:
        fetch_json = _http_get_json
    base = "https://fapi.binance.com/fapi/v1"
    exchange = await fetch_json(f"{base}/exchangeInfo")
    tickers = await fetch_json(f"{base}/ticker/24hr")
    return parse_exchange_info(exchange, parse_volumes(tickers))  # type: ignore[arg-type]


async def _http_get_json(url: str) -> object:
    import aiohttp  # pylint: disable=import-outside-toplevel

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            response.raise_for_status()
            return await response.json()


def _describe(selected: Sequence[SymbolInfo]) -> str:
    lines = [f"{'#':>3} {'sembol':<14}{'24s hacim (M$)':>16}{'tick':>12}{'adım':>10}{'min $':>8}"]
    for number, info in enumerate(selected, 1):
        lines.append(
            f"{number:>3} {info.symbol:<14}{info.quote_volume / 1e6:>16,.1f}{info.tick_size:>12g}"
            f"{info.step_size:>10g}{info.min_notional:>8g}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse  # pylint: disable=import-outside-toplevel
    import asyncio  # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser(description="Evren seçimi (Binance USDT-M vadeli)")
    parser.add_argument("--mode", choices=MODES, default="top_n")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--symbols", nargs="*", default=[], help="list modu: taban adlar")
    parser.add_argument("--min-volume", type=float, default=0.0, help="24s USDT hacim eşiği")
    parser.add_argument("--min-age-days", type=int, default=0)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--out", default="", help="seçimi bu JSON dosyasına yaz")
    args = parser.parse_args(argv)
    spec = UniverseSpec(
        args.mode, args.n, tuple(args.symbols), args.min_volume, args.min_age_days,
        tuple(args.exclude),
    )  # fmt: skip
    selected = select_universe(asyncio.run(fetch_market()), spec)
    print(_describe(selected))
    print(f"\nseçilen: {len(selected)}")
    if args.out:
        save_snapshot(Path(args.out), selected, spec)
        print(f"yazıldı: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
