"""İzleme arayüzü: salt okunur API, boş veritabanı, çıkış nedeni metinleri, mod çözümleme."""

import asyncio
import json

from aiohttp.test_utils import TestClient, TestServer

from alamet.store import Store
from alamet.webui import connect, make_app, mode_of


def universe(tmp_path):
    path = tmp_path / "u.json"
    path.write_text(json.dumps({"symbols": [{"base": "BTC"}, {"base": "ETH"}]}), encoding="utf-8")
    return path


def fetch(db, uni, tmp_path):
    async def go():
        app = make_app(db, uni, tmp_path / "HALT")
        async with TestClient(TestServer(app)) as client:
            page = await client.get("/")
            api = await client.get("/api/all")
            return page.status, await api.json()

    return asyncio.run(go())


def test_empty_database_gives_empty_but_valid_payload(tmp_path):
    db = tmp_path / "t.sqlite"
    Store(db).close()
    status, data = fetch(db, universe(tmp_path), tmp_path)
    assert status == 200
    assert data["candidates"] == [] and data["open"] == [] and data["trades"] == []
    assert data["status"]["last_bar"] is None and data["status"]["halt"] is None


def test_trades_orders_and_halt_are_shown(tmp_path):
    db = tmp_path / "t.sqlite"
    with Store(db) as store, store.conn:
        store.conn.execute(
            "INSERT INTO trades(trade_id, symbol, side, open_time, close_time, entry, exit, "
            "notional, usd, r, pnl_pct, reason, rank, method, strength, div_n, mfe, hold, "
            "signal_pct) VALUES (1,'BTC',1,100,200,10,11,5,0.5,1,10,1,1,2,2,5,1.5,4,3)"
        )
        store.set_meta("halt", "deneme")
    (tmp_path / "HALT").write_text("x")
    _, data = fetch(db, universe(tmp_path), tmp_path)
    assert data["trades"][0]["reason_text"] == "Kâr geri verme kilidi"
    assert data["trades"][0]["method_text"] == "Volatilite"
    assert data["status"]["halt"] == "deneme"


def test_database_is_opened_read_only(tmp_path):
    db = tmp_path / "t.sqlite"
    Store(db).close()
    conn = connect(db)
    try:
        conn.execute("INSERT INTO meta(key, value) VALUES ('x', '1')")
        raise AssertionError("yazma kabul edildi")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        assert "readonly" in str(exc).lower()
    finally:
        conn.close()


def test_mode_of_reads_flags():
    mode = mode_of("python -m alamet.runner --trade demo --only BTC --max-total 5000")
    assert mode["trade"] == "demo" and mode["only"] == "BTC" and mode["max_total"] == "5000"
    assert mode_of("python -m alamet.runner")["trade"] == "off"
