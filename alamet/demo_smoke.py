"""
Demo ortamında tek emir uçtan uca denemesi (duman testi): aç → doğrula → kapat.

Gerçek (`live`) ortamda ÇALIŞMAZ. Ölçülenler: gecikme, kayma (mark fiyatına göre), komisyon,
gerçekleşen K/Z. Aynı yolu (`orders` → `futures_api`) üretim emirleri de kullanır.
"""

from __future__ import annotations

import asyncio
import time

from alamet.futures_api import FuturesClient, health_check
from alamet.market import fetch_market
from alamet.orders import BUY, OrderIntent, OrderRules, client_id, order_side, round_down


async def run_smoke(
    api: FuturesClient, base: str = "BTC", notional: float = 100.0, leverage: int = 5
):
    if api.environment == "live":
        raise RuntimeError("Duman testi gerçek ortamda çalıştırılamaz")
    symbol = f"{base}USDT"
    infos = {i.symbol: i for i in await fetch_market()}
    rules = OrderRules.from_info(infos[symbol])
    state = await health_check(api)
    if not state["tek_yönlü_pozisyon"] or state["açık_pozisyon"].get(symbol):
        raise RuntimeError(f"Başlangıç koşulu uygun değil: {state}")
    await api.set_margin_type(symbol, "ISOLATED")
    await api.set_leverage(symbol, leverage)
    mark = await api.mark_price(symbol)
    quantity = round_down(notional / mark, rules.step)
    stamp = int(time.time())
    opening = OrderIntent(
        "open", base, symbol, BUY, quantity, mark, stamp, stamp, client_id(stamp, "open"), False
    )
    await api.test_order(opening)
    started = time.perf_counter()
    filled = await api.place_order(opening)
    open_ms = (time.perf_counter() - started) * 1000
    open_fills = await api.fills(symbol, filled.order_id)
    open_price = filled.avg_price or open_fills["avg_price"]
    held = (await api.positions()).get(symbol, 0.0)
    mark_close = await api.mark_price(symbol)
    closing = OrderIntent(
        "close", base, symbol, order_side(1, "close"), abs(held), mark_close, stamp, stamp,
        client_id(stamp, "close"), True,
    )  # fmt: skip
    started = time.perf_counter()
    closed = await api.place_order(closing)
    close_ms = (time.perf_counter() - started) * 1000
    close_fills = await api.fills(symbol, closed.order_id)
    close_price = closed.avg_price or close_fills["avg_price"]
    after = await api.positions()
    return {
        "sembol": symbol,
        "miktar": quantity,
        "açılış": {
            "durum": filled.status, "doldu": filled.executed_qty, "ort_fiyat": open_price,
            "mark_önce": mark, "kayma_%": (open_price / mark - 1) * 100,
            "gecikme_ms": round(open_ms), "komisyon": open_fills["commission"],
        },
        "pozisyon_açıkken": held,
        "kapanış": {
            "durum": closed.status, "doldu": closed.executed_qty, "ort_fiyat": close_price,
            "mark_önce": mark_close, "kayma_%": (1 - close_price / mark_close) * 100,
            "gecikme_ms": round(close_ms), "komisyon": close_fills["commission"],
            "gerçekleşen_kz": close_fills["realized_pnl"],
        },
        "pozisyon_kapandıktan_sonra": after.get(symbol, 0.0),
    }  # fmt: skip


def main() -> int:
    import json  # pylint: disable=import-outside-toplevel
    import sys  # pylint: disable=import-outside-toplevel

    environment = sys.argv[1] if len(sys.argv) > 1 else "demo"
    api = FuturesClient(environment)
    print(json.dumps(asyncio.run(run_smoke(api)), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
