"""Yürütücü: işlem kutusu, kuru çalışma, güvenlik anahtarları, belirsiz emir, uzlaştırma."""

import asyncio
from types import SimpleNamespace

import pytest

from alamet.events import CloseEvent, OpenEvent
from alamet.executor import Executor, Safety
from alamet.futures_api import ApiError, OrderResult, OrderUncertain
from alamet.orders import OrderRules, RiskLimits
from alamet.store import Store

RULES = {
    "BTC": OrderRules(0.001, 0.001, 1000.0, 100.0),
    "ETH": OrderRules(0.01, 0.01, 1000.0, 20.0),
}
LIMITS = RiskLimits(scale=1.0, max_position_usdt=50_000.0, max_total_usdt=100_000.0)
NOW = 1_800_000_000.0


class FakeApi:
    def __init__(self, price_shift=0.0):
        self.sent, self.queried, self.shift = [], [], price_shift
        self.positions_value, self.fail, self.known = {}, None, {}
        self.prepared, self.stops, self.stop_states = [], {}, {}

    async def place_order(self, intent):
        if self.fail is not None:
            raise self.fail
        self.sent.append(intent)
        avg = intent.ref_price * (1 + self.shift)
        result = OrderResult(intent.client_id, len(self.sent), intent.symbol, intent.side,
                             "FILLED", intent.quantity, avg, avg * intent.quantity,
                             intent.reduce_only)  # fmt: skip
        self.known[intent.client_id] = result
        return result

    async def query_order(self, symbol, client_id):
        self.queried.append(client_id)
        if client_id not in self.known:
            raise ApiError(-2013, "Order does not exist")
        return self.known[client_id]

    async def fills(self, symbol, order_id):
        return {"commission": 0.04, "realized_pnl": -1.0, "trades": 1.0, "quantity": 1.0,
                "avg_price": 0.0}  # fmt: skip

    async def positions(self):
        return dict(self.positions_value)

    async def set_margin_type(self, symbol, margin_type="ISOLATED"):
        self.prepared.append((symbol, margin_type))

    async def set_leverage(self, symbol, leverage):
        self.prepared.append((symbol, leverage))
        return leverage

    async def place_stop(self, symbol, side, trigger_price, client_algo_id):
        self.stops[client_algo_id] = (symbol, side, trigger_price)
        return {"algoStatus": "NEW"}

    async def cancel_stop(self, client_algo_id):
        return self.stops.pop(client_algo_id, None) is not None

    async def stop_status(self, client_algo_id):
        return self.stop_states.get(client_algo_id)

    async def open_stops(self):
        return set(self.stops)


def open_event(trade_id=1, symbol=0, side=1, entry=80_000.0, notional=800.0, dist=float("nan")):
    return OpenEvent(bar=0, phase=0, symbol=symbol, side=side, trade_id=trade_id, method=0,
                     strength=0, rank=1, div_n=0, entry=entry, take=0.0, notional=notional,
                     risk_pct=0.0, multiplier=1.0, dyn_mult=1.0, kelly_mult=1.0, dr_mult=1.0,
                     rank_mult=1.0, dyn_on=False, dyn_score=0.0, dd_factor=1.0, streak_factor=1.0,
                     below_start_cap=False, kelly_on=False, kelly_mean=0.0, kelly_effective_n=0.0,
                     kelly_exploring=False, dr_on=False, dr_ratio=0.0, dr_percentile=0.0,
                     dist=dist)  # fmt: skip


def close_event(trade_id=1, symbol=0, side=1, exit_=81_000.0):
    return CloseEvent(bar=1, phase=0, symbol=symbol, side=side, trade_id=trade_id, status=2,
                      reason=1, entry=80_000.0, exit=exit_, r=1.0, pnl_pct=1.0, usd=1.0,
                      hold=1)  # fmt: skip


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "t.sqlite") as s:
        yield s


def make(store, api=None, **safety):
    safety.setdefault("dry_run", False)
    return Executor(api, store, ("BTC", "ETH"), RULES, LIMITS, Safety(**safety), log=lambda m: None,
                    clock=lambda: NOW)  # fmt: skip


def queue(store, executor, events, bar_time=100):
    rows = executor.plan(events, bar_time)
    with store.conn:
        store._write_outbox(rows)  # pylint: disable=protected-access
    return rows


def run(coro):
    return asyncio.run(coro)


def test_open_then_close_round_trip(store):
    api = FakeApi(price_shift=0.0001)
    ex = make(store, api)
    queue(store, ex, [open_event()])
    assert run(ex.execute_pending()) == 1
    pos = store.positions()
    assert len(pos) == 1 and pos.quantity[0] == 0.01 and pos.side[0] == 1
    queue(store, ex, [close_event()])
    run(ex.execute_pending())
    assert store.positions().empty
    assert [i.reduce_only for i in api.sent] == [False, True]
    orders = store.orders()
    assert list(orders.status) == ["FILLED", "FILLED"]
    by_id = orders.set_index("client_id")
    assert by_id.slippage_pct["al1o"] == pytest.approx(0.01)
    assert by_id.slippage_pct["al1c"] == pytest.approx(-0.01)
    assert by_id.commission["al1o"] == 0.04


def test_same_event_queued_twice_sends_once(store):
    api = FakeApi()
    ex = make(store, api)
    queue(store, ex, [open_event()])
    queue(store, ex, [open_event()])
    run(ex.execute_pending())
    assert len(api.sent) == 1 and len(store.outbox()) == 1


def test_retry_after_crash_queries_before_sending(store):
    api = FakeApi()
    ex = make(store, api)
    queue(store, ex, [open_event()])
    run(ex.execute_pending())
    with store.conn:
        store.conn.execute("UPDATE outbox SET state='pending', attempts=1")
    run(ex.execute_pending())
    assert len(api.sent) == 1 and api.queried == ["al1o"]


def test_retry_sends_when_exchange_never_saw_the_order(store):
    api = FakeApi()
    ex = make(store, api)
    queue(store, ex, [open_event()])
    with store.conn:
        store.conn.execute("UPDATE outbox SET attempts=1")
    run(ex.execute_pending())
    assert api.queried == ["al1o"] and len(api.sent) == 1


def test_dry_run_sends_nothing_but_tracks_positions(store):
    ex = make(store, None, dry_run=True)
    queue(store, ex, [open_event()])
    run(ex.execute_pending())
    assert len(store.positions()) == 1
    queue(store, ex, [close_event()])
    run(ex.execute_pending())
    assert store.positions().empty and set(store.orders().status) == {"DRY"}


def test_real_orders_require_a_client():
    with pytest.raises(ValueError):
        Executor(None, SimpleNamespace(), (), {}, LIMITS, Safety(dry_run=False))


def test_halt_blocks_opens_but_not_closes(store):
    api = FakeApi()
    ex = make(store, api)
    queue(store, ex, [open_event()])
    run(ex.execute_pending())
    ex.halt("deneme")
    queue(store, ex, [open_event(trade_id=2, symbol=1, entry=3000.0, notional=300.0)])
    queue(store, ex, [close_event()])
    run(ex.execute_pending())
    states = dict(zip(store.outbox().client_id, store.outbox().state))
    assert states == {"al1o": "done", "al2o": "blocked", "al1c": "done"}
    ex.resume()
    assert ex.halted() is None


def test_halt_file(store, tmp_path):
    flag = tmp_path / "HALT"
    ex = make(store, FakeApi(), halt_file=flag)
    queue(store, ex, [open_event()])
    flag.write_text("x")
    run(ex.execute_pending())
    assert store.outbox().state[0] == "blocked"


def test_daily_loss_cap_blocks_new_opens(store):
    api = FakeApi()
    ex = make(store, api, daily_loss_usdt=1.0)
    queue(store, ex, [open_event()])
    run(ex.execute_pending())
    queue(store, ex, [close_event()])
    run(ex.execute_pending())
    queue(store, ex, [open_event(trade_id=2)])
    run(ex.execute_pending())
    assert store.outbox().state.tolist() == ["done", "done", "blocked"]
    assert "günlük_kayıp" in store.outbox().error[2]


def test_daily_order_limit_blocks_opens(store):
    ex = make(store, FakeApi(), max_orders_per_day=1)
    queue(store, ex, [open_event(), open_event(trade_id=2, symbol=1, entry=3000.0, notional=300.0)])
    run(ex.execute_pending())
    assert store.outbox().state.tolist() == ["done", "blocked"]


def test_uncertain_order_halts(store):
    api = FakeApi()
    api.fail = OrderUncertain("belirsiz")
    ex = make(store, api)
    queue(store, ex, [open_event()])
    run(ex.execute_pending())
    assert store.outbox().state[0] == "uncertain" and ex.halted()


def test_api_error_marks_failed_without_halt(store):
    api = FakeApi()
    api.fail = ApiError(-2019, "Margin is insufficient.")
    ex = make(store, api)
    queue(store, ex, [open_event()])
    run(ex.execute_pending())
    assert store.outbox().state[0] == "failed" and ex.halted() is None
    assert store.positions().empty


def test_rejections_are_noted_and_close_without_position_is_rejected(store):
    ex = make(store, FakeApi())
    assert queue(store, ex, [open_event(notional=1.0)]) == []
    assert queue(store, ex, [close_event()]) == []
    kinds = store.conn.execute("SELECT kind, payload FROM events").fetchall()
    assert [k for k, _ in kinds] == ["EmirReddi", "EmirReddi"]
    assert "min_miktar" in kinds[0][1] and "miktar_yok" in kinds[1][1]


def test_reconcile_detects_difference_and_halts(store):
    api = FakeApi()
    ex = make(store, api)
    queue(store, ex, [open_event()])
    run(ex.execute_pending())
    api.positions_value = {"BTCUSDT": 0.01}
    assert run(ex.reconcile()) == [] and ex.halted() is None
    api.positions_value = {"BTCUSDT": 0.0}
    assert len(run(ex.reconcile())) == 1 and ex.halted()


def test_reconcile_ignores_symbols_outside_universe(store):
    api = FakeApi()
    api.positions_value = {"DOGEUSDT": 5.0}
    ex = make(store, api)
    assert run(ex.reconcile()) == [] and ex.halted() is None


def test_short_positions_reconcile_with_sign(store):
    api = FakeApi()
    ex = make(store, api)
    queue(store, ex, [open_event(side=-1)])
    run(ex.execute_pending())
    assert store.orders().side[0] == "SELL"
    api.positions_value = {"BTCUSDT": -0.01}
    assert run(ex.reconcile()) == []


def test_open_and_close_in_the_same_bar_run_in_order(store):
    api = FakeApi()
    ex = make(store, api)
    queue(store, ex, [open_event(), close_event()])
    run(ex.execute_pending())
    assert [i.client_id for i in api.sent] == ["al1o", "al1c"]
    assert store.positions().empty and store.outbox().state.tolist() == ["done", "done"]


def test_close_is_skipped_when_its_open_was_blocked(store):
    api = FakeApi()
    ex = make(store, api)
    ex.halt("deneme")
    queue(store, ex, [open_event(), close_event()])
    run(ex.execute_pending())
    assert api.sent == []
    assert store.outbox().state.tolist() == ["blocked", "skipped"]


def test_close_is_skipped_when_its_open_failed(store):
    api = FakeApi()
    api.fail = ApiError(-2019, "Margin is insufficient.")
    ex = make(store, api)
    queue(store, ex, [open_event(), close_event()])
    run(ex.execute_pending())
    assert store.outbox().state.tolist() == ["failed", "skipped"]


def test_unknown_exchange_position_in_universe_halts(store):
    api = FakeApi()
    api.positions_value = {"ETHUSDT": 1.0}
    ex = make(store, api)
    assert len(run(ex.reconcile())) == 1 and ex.halted()


def test_hard_kill_between_commit_and_send_resends_once(tmp_path):
    import subprocess
    import sys
    import textwrap

    db = tmp_path / "k.sqlite"
    child = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, "tests")
        from test_executor import *
        s = Store({str(db)!r})
        ex = make(s, FakeApi())
        queue(s, ex, [open_event()])
        os._exit(9)
    """)
    done = subprocess.run([sys.executable, "-c", child], check=False, capture_output=True)
    assert done.returncode == 9, done.stderr
    api = FakeApi()
    with Store(db) as reopened:
        ex = make(reopened, api)
        assert len(reopened.pending_outbox()) == 1
        run(ex.execute_pending())
        run(ex.execute_pending())
        assert len(api.sent) == 1 and len(reopened.positions()) == 1


def test_hard_kill_after_send_before_record_does_not_double_send(tmp_path):
    db = tmp_path / "k2.sqlite"
    api = FakeApi()
    with Store(db) as first:
        ex = make(first, api)
        queue(first, ex, [open_event()])
        with first.conn:
            first.mark_outbox(first.pending_outbox()[0]["id"], "pending", None, 1)
        run(api.place_order(OrderIntentStub()))
    with Store(db) as second:
        ex = make(second, api)
        run(ex.execute_pending())
        assert len(api.sent) == 1 and len(second.positions()) == 1


class OrderIntentStub:
    client_id, symbol, side, quantity, ref_price, reduce_only = (
        "al1o", "BTCUSDT", "BUY", 0.01, 80_000.0, False,
    )  # fmt: skip


def test_open_prepares_symbol_and_places_stop(store):
    api = FakeApi()
    ex = make(store, api)
    queue(store, ex, [open_event(dist=0.01)])
    run(ex.execute_pending())
    assert api.prepared == [("BTCUSDT", "ISOLATED"), ("BTCUSDT", 5)]
    assert api.stops == {"al1s": ("BTCUSDT", "SELL", pytest.approx(76_800.0))}


def test_short_open_places_buy_stop_above_entry(store):
    api = FakeApi()
    ex = make(store, api)
    queue(store, ex, [open_event(side=-1, dist=0.01)])
    run(ex.execute_pending())
    assert api.stops == {"al1s": ("BTCUSDT", "BUY", pytest.approx(83_200.0))}


def test_no_stop_without_risk_distance(store):
    api = FakeApi()
    ex = make(store, api)
    queue(store, ex, [open_event()])
    run(ex.execute_pending())
    assert api.stops == {} and len(store.positions()) == 1


def test_close_cancels_stop(store):
    api = FakeApi()
    ex = make(store, api)
    queue(store, ex, [open_event(dist=0.01)])
    run(ex.execute_pending())
    queue(store, ex, [close_event()])
    run(ex.execute_pending())
    assert api.stops == {}


def test_stop_failure_does_not_halt_and_reconcile_replaces_it(store):
    api = FakeApi()
    ex = make(store, api)

    async def broken(*_):
        raise ApiError(-1000, "boom")

    place = api.place_stop
    api.place_stop = broken
    queue(store, ex, [open_event(dist=0.01)])
    run(ex.execute_pending())
    assert len(store.positions()) == 1 and ex.halted() is None
    assert any(r.kind == "StopHatası" for r in store.events().itertuples())
    api.place_stop = place
    api.positions_value = {"BTCUSDT": 0.01}
    assert run(ex.reconcile()) == [] and "al1s" in api.stops


def test_fired_stop_closes_system_position_without_halt(store):
    api = FakeApi()
    ex = make(store, api)
    queue(store, ex, [open_event(dist=0.01)])
    run(ex.execute_pending())
    api.stops.clear()
    api.stop_states["al1s"] = "TRIGGERED"
    api.positions_value = {}
    assert run(ex.reconcile()) == []
    assert store.positions().empty and ex.halted() is None
    assert any(r.kind == "StopTetiklendi" for r in store.events().itertuples())


def test_flat_exchange_without_fired_stop_still_halts(store):
    api = FakeApi()
    ex = make(store, api)
    queue(store, ex, [open_event(dist=0.01)])
    run(ex.execute_pending())
    api.positions_value = {}
    assert run(ex.reconcile()) != [] and ex.halted() is not None
