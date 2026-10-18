"""
Yürütücü: sistemin AÇ/KAPAT olaylarını emre çevirir, borsaya gönderir, sonucu kaydeder.

Akış (bar başına):
  1. `plan`   : olaylar → emir niyeti (`orders.size_*`) → `outbox` satırları. Satırlar durumla AYNI
                işlemde yazılır (`Store.commit(outbox=...)`): çökme olursa niyet kaybolmaz.
  2. `execute_pending`: bekleyen satırlar sırayla gönderilir. Gönderimden önce `attempts` yazılır;
                yeniden denemede emir önce kimliğiyle sorgulanır (aynı emir iki kez gitmez).
  3. `reconcile`: sistemin pozisyonları ile borsadaki pozisyonlar karşılaştırılır; fark varsa DURUR.

Felaket stopu: açılış dolunca borsaya `closePosition` koşullu stop konur (yalnızca bot yokken
korur; motor çıkışı yine kendi yapar). Kapanışta iptal edilir; eksik kalan stop `reconcile`
sırasında yeniden konur. Stop tetiklenmişse sistem pozisyonu kapalı yazılır, bot durmaz.

Güvenlik anahtarları yalnızca YENİ AÇILIŞLARI engeller; kapanışlar (riski azaltan) her zaman geçer:
  - acil durdurma (`HALT` dosyası ya da kayıtlı durdurma), - günlük kayıp tavanı,
  - günlük emir sayısı sınırı, - kuru çalışma (borsaya hiçbir şey gitmez, kayıt tutulur).
"""

# pylint: disable=too-many-instance-attributes
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from alamet.events import CloseEvent, OpenEvent
from alamet.futures_api import STOP_DONE, ApiError, FuturesClient, OrderUncertain
from alamet.orders import (
    OrderIntent,
    OrderRules,
    Rejection,
    RiskLimits,
    size_close,
    size_open,
    stop_client_id,
)

HALT_KEY = "halt"
DAY_MS = 86_400_000
UNKNOWN_ORDER = -2013


@dataclass(frozen=True)
class Safety:
    """Güvenlik anahtarları. `None` olan sınır uygulanmaz."""

    daily_loss_usdt: float | None = None
    max_orders_per_day: int | None = None
    dry_run: bool = True
    halt_file: Path | None = None


class Executor:
    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        api: FuturesClient | None,
        store,
        symbols: Sequence[str],
        rules: Mapping[str, OrderRules],
        limits: RiskLimits,
        safety: Safety = Safety(),
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if api is None and not safety.dry_run:
            raise ValueError("Gerçek emir için borsa istemcisi gerekir")
        self.api, self.store, self.symbols = api, store, tuple(symbols)
        self._prepared: set[str] = set()
        self.rules, self.limits, self.safety, self.log, self.clock = (
            rules,
            limits,
            safety,
            log,
            clock,
        )

    # ------------------------------------------------------------ durdurma
    def halted(self) -> str | None:
        if self.safety.halt_file is not None and Path(self.safety.halt_file).exists():
            return "HALT dosyası"
        return self.store.get_meta(HALT_KEY)

    def halt(self, reason: str) -> None:
        if self.store.get_meta(HALT_KEY) is None:
            with self.store.conn:
                self.store.set_meta(HALT_KEY, reason)
            self.log(f"DURDURULDU: {reason}")

    def resume(self) -> None:
        with self.store.conn:
            self.store.conn.execute("DELETE FROM meta WHERE key=?", (HALT_KEY,))

    # --------------------------------------------------------------- plan
    def plan(self, events: Sequence, bar_time: int) -> list[dict]:
        """Olaylardan `outbox` satırları (henüz yazılmadı). Reddedilenler `note` ile kaydedilir."""
        held = self.store.positions()
        quantity = {int(r.trade_id): float(r.quantity) for r in held.itertuples()}
        open_notional = float((held.quantity * held.avg_price).sum()) if len(held) else 0.0
        rows: list[dict] = []
        for event in events:
            if not isinstance(event, (OpenEvent, CloseEvent)):
                continue
            base = self.symbols[int(event.symbol)]
            rules = self.rules.get(base)
            if rules is None:
                self._rejected(Rejection(base, int(event.trade_id), "kural_yok"), bar_time)
                continue
            if isinstance(event, OpenEvent):
                result = size_open(event, base, rules, self.limits, open_notional, bar_time)
                if isinstance(result, OrderIntent):
                    open_notional += result.notional
                    quantity[result.trade_id] = result.quantity
            else:
                result = size_close(
                    event, base, quantity.get(int(event.trade_id), 0.0), rules, bar_time
                )
            if isinstance(result, Rejection):
                self._rejected(result, bar_time)
            else:
                rows.append(self._row(result))
        return rows

    @staticmethod
    def _row(intent: OrderIntent) -> dict:
        return {
            "trade_id": intent.trade_id, "kind": intent.kind, "base": intent.base,
            "symbol": intent.symbol, "side": intent.side, "quantity": intent.quantity,
            "ref_price": intent.ref_price, "client_id": intent.client_id,
            "reduce_only": int(intent.reduce_only), "reason": intent.reason or None,
            "bar_time": intent.bar_time, "stop_price": intent.stop_price,
        }  # fmt: skip

    def _rejected(self, rejection: Rejection, bar_time: int) -> None:
        self.log(
            f"EMİR REDDİ {rejection.base} işlem {rejection.trade_id}: {rejection.reason} {rejection.detail}"
        )
        with self.store.conn:
            self.store.note("EmirReddi", bar_time, rejection.base, rejection.__dict__)

    # ----------------------------------------------------------- yürütme
    def _today(self) -> tuple[float, int]:
        """Bugünkü (UTC) net K/Z (komisyon dahil) ve gönderilen emir sayısı."""
        start = int(self.clock() * 1000) // DAY_MS * DAY_MS
        row = self.store.conn.execute(
            "SELECT COALESCE(SUM(COALESCE(realized_pnl,0) - COALESCE(commission,0)),0), COUNT(*) "
            "FROM orders WHERE sent_at >= ? AND status != 'DRY'",
            (start,),
        ).fetchone()
        return float(row[0]), int(row[1])

    def _blocked(self, row: dict) -> str | None:
        """Yeni açılışı engelleyen neden (kapanışlar asla engellenmez)."""
        if row["kind"] != "open":
            return None
        if (reason := self.halted()) is not None:
            return f"durdurma: {reason}"
        pnl, count = self._today()
        cap = self.safety.daily_loss_usdt
        if cap is not None and pnl <= -abs(cap):
            return f"günlük_kayıp_tavanı: {pnl:.2f} <= -{abs(cap)}"
        limit = self.safety.max_orders_per_day
        if limit is not None and count >= limit:
            return f"günlük_emir_sınırı: {count} >= {limit}"
        return None

    async def execute_pending(self) -> int:
        """Bekleyen niyetleri sırayla yürütür; işlenen satır sayısını döner."""
        done = 0
        for row in self.store.pending_outbox():
            if (reason := self._blocked(row)) is not None:
                self._finish(row, "blocked", reason)
                self.log(f"ENGELLENDİ {row['base']} {row['client_id']}: {reason}")
            elif row["kind"] == "close" and not self._has_open_position(row["trade_id"]):
                self._finish(row, "skipped", "açık_pozisyon_yok")
                self.log(f"ATLANDI {row['base']} {row['client_id']}: bu işlemin açık pozisyonu yok")
            elif self.safety.dry_run:
                self._dry(row)
            else:
                await self._send(row)
            done += 1
        return done

    def _has_open_position(self, trade_id: int) -> bool:
        return trade_id in set(self.store.positions().trade_id)

    def _finish(self, row: dict, state: str, error: str | None = None) -> None:
        with self.store.conn:
            self.store.mark_outbox(row["id"], state, error)

    def _dry(self, row: dict) -> None:
        now = int(self.clock() * 1000)
        with self.store.conn:
            self.store.record_order(
                {"client_id": row["client_id"], "trade_id": row["trade_id"], "kind": row["kind"],
                 "symbol": row["symbol"], "side": row["side"], "status": "DRY",
                 "quantity": row["quantity"], "executed_qty": row["quantity"],
                 "avg_price": row["ref_price"], "ref_price": row["ref_price"],
                 "slippage_pct": 0.0, "commission": 0.0, "realized_pnl": 0.0,
                 "latency_ms": 0.0, "sent_at": now, "bar_time": row["bar_time"]}
            )  # fmt: skip
            self._apply_position(row, row["quantity"], row["ref_price"], now)
            self.store.mark_outbox(row["id"], "dry")
        self.log(f"KURU {row['side']} {row['quantity']} {row['symbol']} @ {row['ref_price']}")

    def _apply_position(self, row: dict, qty: float, price: float, now: int) -> None:
        if row["kind"] == "open":
            side = 1 if row["side"] == "BUY" else -1
            self.store.open_position(row["trade_id"], row["symbol"], side, qty, price, now)
        else:
            self.store.close_position(row["trade_id"], now)

    async def _send(self, row: dict) -> None:
        api = self.api
        intent = OrderIntent(
            row["kind"], row["base"], row["symbol"], row["side"], row["quantity"],
            row["ref_price"], row["trade_id"], row["bar_time"], row["client_id"],
            bool(row["reduce_only"]), row["reason"] or "",
        )  # fmt: skip
        resend = row["attempts"] > 0
        if row["kind"] == "open":
            await self._prepare(row["symbol"])
        with self.store.conn:
            self.store.mark_outbox(row["id"], "pending", None, 1)
        started = time.perf_counter()
        try:
            result = None
            if resend:
                try:
                    result = await api.query_order(row["symbol"], row["client_id"])
                except ApiError as lookup:
                    if lookup.code != UNKNOWN_ORDER:
                        raise
            if result is None:
                result = await api.place_order(intent)
        except OrderUncertain as exc:
            self._finish(row, "uncertain", str(exc))
            self.halt(f"belirsiz emir {row['client_id']}")
            return
        except ApiError as exc:
            self._finish(row, "failed", f"{exc.code}: {exc}")
            self.log(f"EMİR HATASI {row['client_id']}: {exc.code} {exc}")
            return
        latency = (time.perf_counter() - started) * 1000.0
        await self._record(row, result, latency)

    async def _record(self, row: dict, result, latency_ms: float) -> None:
        extra = {"commission": None, "realized_pnl": None}
        avg = result.avg_price
        try:
            fills = await self.api.fills(row["symbol"], result.order_id)
            extra = {"commission": fills["commission"], "realized_pnl": fills["realized_pnl"]}
            avg = avg or fills["avg_price"]
        except (ApiError, OSError) as exc:
            self.log(f"UYARI dolum bilgisi alınamadı {row['client_id']}: {exc!r}")
        ref = row["ref_price"]
        sign = 1.0 if row["side"] == "BUY" else -1.0
        slippage = (avg - ref) / ref * 100.0 * sign if avg and ref else None
        now = int(self.clock() * 1000)
        ok = result.executed_qty > 0
        with self.store.conn:
            self.store.record_order(
                {"client_id": row["client_id"], "trade_id": row["trade_id"], "kind": row["kind"],
                 "symbol": row["symbol"], "side": row["side"], "order_id": result.order_id,
                 "status": result.status, "quantity": row["quantity"],
                 "executed_qty": result.executed_qty, "avg_price": avg, "ref_price": ref,
                 "slippage_pct": slippage, "latency_ms": latency_ms, "sent_at": now,
                 "bar_time": row["bar_time"], **extra}
            )  # fmt: skip
            if ok:
                self._apply_position(row, result.executed_qty, avg, now)
            self.store.mark_outbox(
                row["id"], "done" if ok else "failed", None if ok else result.status
            )
        if ok:
            await self._after_fill(row)
        warn = self.limits.slippage_warn_pct
        if slippage is not None and slippage > warn:
            self.log(f"UYARI kayma {row['symbol']} {row['kind']}: %{slippage:.3f} > %{warn}")
        self.log(
            f"EMİR {row['kind']} {row['side']} {result.executed_qty} {row['symbol']} @ {avg} "
            f"kayma {slippage if slippage is None else round(slippage, 4)} gecikme {latency_ms:.0f} ms"
        )

    # ------------------------------------------------------ felaket stopu
    async def _prepare(self, symbol: str) -> None:
        """Sembolü izole marjin ve ayarlı kaldıraca getirir (süreç başına bir kez)."""
        if symbol in self._prepared:
            return
        await self.api.set_margin_type(symbol, "ISOLATED")
        await self.api.set_leverage(symbol, self.limits.leverage)
        self._prepared.add(symbol)

    async def _after_fill(self, row: dict) -> None:
        """Açılış dolunca stop koy, kapanış dolunca stopu iptal et. Hata botu durdurmaz."""
        try:
            if row["kind"] == "open":
                await self._place_stop(row)
            else:
                await self.api.cancel_stop(stop_client_id(row["trade_id"]))
        except (ApiError, OSError) as exc:
            self.log(f"UYARI stop işlemi başarısız {row['client_id']}: {exc!r}")
            with self.store.conn:
                self.store.note("StopHatası", row["bar_time"], row["base"], {"hata": repr(exc)})

    async def _place_stop(self, row: dict) -> None:
        price = float(row.get("stop_price") or 0.0)
        if price <= 0:
            return
        side = "SELL" if row["side"] == "BUY" else "BUY"
        await self.api.place_stop(row["symbol"], side, price, stop_client_id(row["trade_id"]))
        self.log(f"STOP {side} {row['symbol']} tetik {price} ({stop_client_id(row['trade_id'])})")

    def _open_intents(self) -> dict[int, dict]:
        """Açık pozisyonların açılış niyetleri (stop fiyatı ve yönü için)."""
        ids = [int(i) for i in self.store.positions().trade_id]
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        cursor = self.store.conn.execute(
            f"SELECT * FROM outbox WHERE kind='open' AND trade_id IN ({marks})", ids
        )
        names = [c[0] for c in cursor.description]
        return {int(r[names.index("trade_id")]): dict(zip(names, r)) for r in cursor.fetchall()}

    async def ensure_stops(self, theirs: Mapping[str, float]) -> None:
        """Borsada pozisyonu olup stopu olmayanlara stop koyar; tetiklenmiş stopu işler."""
        intents = self._open_intents()
        if not intents:
            return
        resting = await self.api.open_stops()
        positions = self.store.positions()
        for r in positions.itertuples():
            intent = intents.get(int(r.trade_id))
            if intent is None or float(intent.get("stop_price") or 0.0) <= 0:
                continue
            stop_id = stop_client_id(r.trade_id)
            if theirs.get(r.symbol, 0.0) == 0.0:
                if await self.api.stop_status(stop_id) in STOP_DONE:
                    with self.store.conn:
                        self.store.close_position(int(r.trade_id), int(self.clock() * 1000))
                        self.store.note(
                            "StopTetiklendi", int(intent["bar_time"]), intent["base"],
                            {"trade_id": int(r.trade_id), "tetik": intent["stop_price"]},
                        )  # fmt: skip
                    self.log(
                        f"STOP TETİKLENDİ {r.symbol} işlem {int(r.trade_id)}: sistem kapatıldı"
                    )
            elif stop_id not in resting:
                try:
                    await self._place_stop(intent)
                except ApiError as exc:
                    self.log(f"UYARI stop yeniden konamadı {r.symbol}: {exc!r}")

    # ---------------------------------------------------------- uzlaştırma
    async def reconcile(self) -> list[str]:
        """Sistem pozisyonları ile borsa pozisyonları. Fark varsa durdurur ve farkları döner."""
        if self.safety.dry_run or self.api is None:
            return []
        theirs = await self.api.positions()
        try:
            await self.ensure_stops(theirs)
        except (ApiError, OSError) as exc:
            self.log(f"UYARI stop denetimi yapılamadı: {exc!r}")
        mine: dict[str, float] = {}
        for r in self.store.positions().itertuples():
            mine[r.symbol] = mine.get(r.symbol, 0.0) + r.side * float(r.quantity)
        universe = {f"{b}USDT" for b in self.symbols}
        problems = []
        for symbol in sorted(set(mine) | (set(theirs) & universe)):
            a, b = mine.get(symbol, 0.0), theirs.get(symbol, 0.0)
            step = (
                self.rules[symbol.removesuffix("USDT")].step
                if symbol.removesuffix("USDT") in self.rules
                else 0.0
            )
            if abs(a - b) > max(step, 1e-12) / 2:
                problems.append(f"{symbol}: sistem {a} borsa {b}")
        if problems:
            self.halt("pozisyon farkı: " + "; ".join(problems))
        return problems
