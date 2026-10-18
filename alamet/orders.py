"""
Emir niyeti: sistemin AÇ/KAPAT olaylarını Binance kurallarına uygun, güvenlik sınırlı emre çevirir.

Saf koddur (ağ yok). Karar çekirdeğine dokunmaz: olaydaki sanal büyüklük sabit bir ÖLÇEK oranıyla
gerçek büyüklüğe çevrilir, lot adımına aşağı yuvarlanır, borsa kuralları (minimum miktar/tutar,
azami miktar) ve katı USDT tavanları denetlenir. Uygun değilse `Rejection` döner (neden kayıtlı);
sessizce kırpılmaz, çünkü kırpılmış büyüklük sistemin varsaydığı oranı bozar.

Kapanış, açılışta gerçekleşen MİKTARI kapatır (yeniden hesaplanmaz) ve her zaman `reduce_only`dir.
`client_id` işlemden türetilir: aynı olay iki kez işlense de borsa ikincisini reddeder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal

from alamet.market import SymbolInfo

BUY, SELL = "BUY", "SELL"


@dataclass(frozen=True)
class OrderRules:
    """Bir sözleşmenin piyasa emri kuralları (`SymbolInfo`'dan)."""

    step: float
    min_qty: float
    max_qty: float
    min_notional: float
    tick: float = 0.0

    @classmethod
    def from_info(cls, info: SymbolInfo) -> "OrderRules":
        """Piyasa emri için `MARKET_LOT_SIZE` kuralları (yoksa `LOT_SIZE`)."""
        return cls(
            info.market_step_size or info.step_size,
            info.market_min_qty or info.min_qty,
            info.market_max_qty or info.max_qty,
            info.min_notional,
            info.tick_size,
        )


@dataclass(frozen=True)
class RiskLimits:
    """Boyutlandırma ölçeği ve katı USDT tavanları (sistemden bağımsız emniyet)."""

    scale: float
    max_position_usdt: float
    max_total_usdt: float
    slippage_warn_pct: float = 0.3
    allowed: frozenset[str] | None = None
    stop_r: float = 4.0
    leverage: int = 5

    def __post_init__(self) -> None:
        if not (self.scale > 0 and self.max_position_usdt > 0 and self.max_total_usdt > 0):
            raise ValueError("ölçek ve tavanlar pozitif olmalı")
        if self.max_position_usdt > self.max_total_usdt:
            raise ValueError("tek pozisyon tavanı toplam tavanı aşamaz")


@dataclass(frozen=True)
class OrderIntent:
    kind: str
    base: str
    symbol: str
    side: str
    quantity: float
    ref_price: float
    trade_id: int
    bar_time: int
    client_id: str
    reduce_only: bool
    reason: str = ""
    stop_price: float = 0.0

    @property
    def notional(self) -> float:
        return self.quantity * self.ref_price


@dataclass(frozen=True)
class Rejection:
    base: str
    trade_id: int
    reason: str
    detail: str = ""


def round_down(value: float, step: float) -> float:
    """`step` katına AŞAĞI yuvarlar (ondalık aritmetik: 0.1+0.2 gibi hatalar olmaz)."""
    if step <= 0 or not math.isfinite(value):
        return value
    units = (Decimal(str(value)) / Decimal(str(step))).to_integral_value(rounding=ROUND_DOWN)
    return float(units * Decimal(str(step)))


def round_up(value: float, step: float) -> float:
    if step <= 0 or not math.isfinite(value):
        return value
    units = (Decimal(str(value)) / Decimal(str(step))).to_integral_value(rounding=ROUND_CEILING)
    return float(units * Decimal(str(step)))


def stop_client_id(trade_id: int) -> str:
    return f"al{int(trade_id)}s"


def stop_price(event, rules: OrderRules, limits: RiskLimits) -> float:
    """Felaket stopu: giriş ∓ stop_r × R (R = motorun risk mesafesi). Borsa tick'ine pozisyondan
    UZAĞA yuvarlanır; hesaplanamazsa ya da kapalıysa 0 (stop yok)."""
    dist = float(getattr(event, "dist", math.nan))
    if not limits.stop_r or not (math.isfinite(dist) and dist > 0):
        return 0.0
    side = int(event.side)
    raw = float(event.entry) * (1.0 - side * limits.stop_r * dist)
    if not (math.isfinite(raw) and raw > 0):
        return 0.0
    return round_down(raw, rules.tick) if side == 1 else round_up(raw, rules.tick)


def client_id(trade_id: int, kind: str) -> str:
    """Borsa istemci kimliği: yalnızca işlem ve türünden türetilir (kopyaya karşı)."""
    return f"al{int(trade_id)}{'o' if kind == 'open' else 'c'}"


def order_side(position_side: int, kind: str) -> str:
    long_open = (position_side == 1) == (kind == "open")
    return BUY if long_open else SELL


def size_open(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    event,
    base: str,
    rules: OrderRules,
    limits: RiskLimits,
    open_notional: float,
    bar_time: int,
) -> OrderIntent | Rejection:
    """AÇ olayı → emir. `open_notional`: şu an açık gerçek pozisyonların toplam USDT değeri."""
    trade_id = int(event.trade_id)
    if limits.allowed is not None and base not in limits.allowed:
        return Rejection(base, trade_id, "izinli_değil")
    price = float(event.entry)
    if not (math.isfinite(price) and price > 0):
        return Rejection(base, trade_id, "geçersiz_fiyat", str(price))
    wanted = float(event.notional) * limits.scale
    if not (math.isfinite(wanted) and wanted > 0):
        return Rejection(base, trade_id, "geçersiz_büyüklük", str(wanted))
    qty = round_down(wanted / price, rules.step)
    if qty < rules.min_qty or qty <= 0:
        return Rejection(base, trade_id, "min_miktar", f"{qty} < {rules.min_qty}")
    if qty > rules.max_qty:
        return Rejection(base, trade_id, "azami_miktar", f"{qty} > {rules.max_qty}")
    notional = qty * price
    if notional < rules.min_notional:
        return Rejection(base, trade_id, "min_tutar", f"{notional:.2f} < {rules.min_notional}")
    if notional > limits.max_position_usdt:
        return Rejection(
            base, trade_id, "pozisyon_tavanı", f"{notional:.2f} > {limits.max_position_usdt}"
        )
    if open_notional + notional > limits.max_total_usdt:
        return Rejection(
            base, trade_id, "toplam_tavan", f"{open_notional + notional:.2f} > {limits.max_total_usdt}"
        )  # fmt: skip
    return OrderIntent(
        "open", base, f"{base}USDT", order_side(int(event.side), "open"), qty, price, trade_id,
        bar_time, client_id(trade_id, "open"), False,
        stop_price=stop_price(event, rules, limits),
    )  # fmt: skip


def size_close(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    event, base: str, held_qty: float, rules: OrderRules, bar_time: int
) -> OrderIntent | Rejection:
    """KAPAT olayı → pozisyonu kapatan `reduce_only` emri (açılışta dolan miktar)."""
    trade_id = int(event.trade_id)
    qty = round_down(abs(float(held_qty)), rules.step)
    if qty <= 0:
        return Rejection(base, trade_id, "kapatılacak_miktar_yok", str(held_qty))
    return OrderIntent(
        "close", base, f"{base}USDT", order_side(int(event.side), "close"), qty,
        float(event.exit), trade_id, bar_time, client_id(trade_id, "close"), True,
        reason=f"durum={int(event.status)} sebep={int(event.reason)}",
    )  # fmt: skip
