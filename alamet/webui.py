"""
Yerel izleme arayüzü (salt okunur): `data/alamet.sqlite` dosyasını açar, çalışan bota dokunmaz.

  python -m alamet.webui --db data/alamet.sqlite --universe data/universe.json

Tarayıcıda http://127.0.0.1:8765 . Sunucu yalnızca bu bilgisayardan erişilir.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle  # nosec B403
import sqlite3
import subprocess  # nosec B404
import time
import zlib
from pathlib import Path

from aiohttp import web

STALE_SECONDS = 20 * 60
CLOSE_REASONS = {
    0: "Kâr al (TP)",
    1: "Kâr geri verme kilidi",
    2: "Ters sinyal",
    3: "Sıralama değişimi (swap)",
    4: "Felaket stop",
    5: "Zaman stopu",
    6: "Sert DD acil çıkış",
    7: "Günlük kayıp limiti",
    8: "Kasa kâr kilidi",
}
LOCK_TEXT = {0: "yön kilidi: geçmiş yok", 2: "yön kilidi: belirsiz"}
METHODS = {0: "Trend", 1: "Momentum", 2: "Volatilite", 3: "Hacim"}
EVENT_NAMES = {
    "OpenEvent": "İşlem açıldı",
    "CloseEvent": "İşlem kapandı",
    "DivergenceEvent": "Ayrışma",
    "DoubleDivergenceEvent": "Çift ayrışma ★",
    "HardDrawdownEvent": "Sert DD",
    "DayLossEvent": "Günlük kayıp limiti",
    "EquityLockEvent": "Kasa kâr kilidi",
    "EmirReddi": "Emir reddi",
    "VeriDüzeltmesi": "Veri düzeltmesi",
}


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def rows(conn: sqlite3.Connection, query: str, args: tuple = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(query, args).fetchall()]


def runner_processes() -> list[str]:
    try:
        out = subprocess.run(  # noqa: S603  # nosec B603
            ["/bin/ps", "-eo", "args"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line for line in out.splitlines() if "alamet.runner" in line and "ps -eo" not in line]


def mode_of(command: str) -> dict:
    parts = command.split()
    value = lambda flag: parts[parts.index(flag) + 1] if flag in parts else None  # noqa: E731
    return {
        "trade": value("--trade") or "off",
        "only": value("--only"),
        "max_position": value("--max-position"),
        "max_total": value("--max-total"),
        "daily_loss": value("--daily-loss"),
        "max_orders": value("--max-orders"),
    }


class Snapshot:
    """Son anlık görüntüyü yalnızca yeni bar geldiğinde yükler (pahalı işlem)."""

    def __init__(self) -> None:
        self.bar_time = None
        self.system = None

    def load(self, conn: sqlite3.Connection):
        row = conn.execute(
            "SELECT bar_time, state FROM snapshots ORDER BY bar_time DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None, None
        if row["bar_time"] != self.bar_time:
            self.system = pickle.loads(zlib.decompress(row["state"]))  # noqa: S301  # nosec B301
            self.bar_time = row["bar_time"]
        return self.bar_time, self.system


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def candidates(system, names: list[str]) -> list[dict]:
    tracker = system.tracker
    p = tracker.p
    out = []
    for k in range(tracker.n):
        if not tracker.active[k]:
            continue
        n, md = int(tracker.count[k]), p.max_div
        sign = -1 if tracker.dir[k] == -1 else 1
        current = tracker.res_mv[k, (n - 1) % md] * sign
        previous = [
            tracker.res_mv[k, (j - 1) % md] * sign
            for j in range(max(1, n - md + 1), n)
            if not math.isnan(tracker.res_mv[k, (j - 1) % md])
        ]
        threshold = None
        if len(previous) >= p.min_prev:
            mean = sum(previous) / len(previous)
            var = sum(v * v for v in previous) / len(previous) - mean * mean
            std = math.sqrt(max(var, 0.0)) if len(previous) > 1 else 0.0
            threshold = mean + p.z_min * std
        lock = tracker.lock(k)
        double, side = tracker.last_double(k), int(tracker.dir[k])
        blockers = []
        if not double:
            blockers.append("çift ayrışma yok")
        if lock != side:
            blockers.append(LOCK_TEXT.get(lock, "yön kilidi ters yönde"))
        out.append(
            {
                "symbol": names[k],
                "side": side,
                "number": n,
                "current": clean(float(current)),
                "threshold": clean(threshold),
                "double": double,
                "lock": int(lock),
                "ready": not blockers,
                "blockers": blockers,
            }
        )
    out.sort(key=lambda x: (not x["ready"], x["symbol"]))
    return out


def open_trades(system, names: list[str]) -> list[dict]:
    state = system.engine.s
    return [
        {
            "symbol": names[k],
            "side": int(state.side[k]),
            "entry": clean(float(state.entry[k])),
            "notional": clean(float(state.notional[k])),
            "bars": int(system.index - state.open_bar[k]),
        }
        for k in range(len(names))
        if state.active[k]
    ]


def snapshot_payload(conn, cache: Snapshot, names, halt_file: Path) -> dict:
    bar_time, system = cache.load(conn)
    now = time.time()
    procs = runner_processes()
    meta = {r["key"]: json.loads(r["value"]) for r in conn.execute("SELECT key, value FROM meta")}
    today = int(now) // 86400 * 86400 * 1000
    counts = {
        r["state"]: r["n"]
        for r in conn.execute("SELECT state, COUNT(*) n FROM outbox GROUP BY state")
    }
    status = {
        "now": int(now),
        "last_bar": bar_time,
        "age_s": None if bar_time is None else int(now - bar_time - 900),
        "alive": bool(procs),
        "mode": mode_of(procs[0]) if procs else None,
        "halt": meta.get("halt") or ("HALT dosyası" if halt_file.exists() else None),
        "symbols": len(names),
        "outbox": counts,
        "orders_today": conn.execute(
            "SELECT COUNT(*) FROM orders WHERE sent_at >= ?", (today,)
        ).fetchone()[0],
        "trade_count": conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0],
    }
    trades = rows(conn, "SELECT * FROM trades ORDER BY trade_id DESC LIMIT 30")
    for t in trades:
        t["reason_text"] = CLOSE_REASONS.get(t["reason"], str(t["reason"]))
        t["method_text"] = METHODS.get(t["method"], str(t["method"]))
    events = rows(
        conn, "SELECT bar_time, kind, symbol, payload FROM events " "ORDER BY id DESC LIMIT 40"
    )
    for e in events:
        e["kind_text"] = EVENT_NAMES.get(e["kind"], e["kind"])
        e["payload"] = e["payload"][:160]
    return {
        "status": status,
        "candidates": candidates(system, names) if system else [],
        "open": open_trades(system, names) if system else [],
        "positions": rows(conn, "SELECT * FROM positions WHERE closed_at IS NULL"),
        "orders": rows(conn, "SELECT * FROM orders ORDER BY sent_at DESC LIMIT 30"),
        "outbox_problems": rows(
            conn,
            "SELECT base, kind, state, error, bar_time FROM outbox "
            "WHERE state NOT IN ('done','pending','dry') "
            "ORDER BY id DESC LIMIT 20",
        ),
        "trades": trades,
        "events": events,
    }


def make_app(db_path: Path, universe_path: Path, halt_file: Path) -> web.Application:
    names = [s["base"] for s in json.loads(universe_path.read_text(encoding="utf-8"))["symbols"]]
    cache = Snapshot()

    async def api(_request: web.Request) -> web.Response:
        conn = connect(db_path)
        try:
            payload = snapshot_payload(conn, cache, names, halt_file)
        finally:
            conn.close()
        return web.Response(text=json.dumps(payload, default=str), content_type="application/json")

    async def index(_request: web.Request) -> web.Response:
        page = (Path(__file__).parent / "webui.html").read_text(encoding="utf-8")
        return web.Response(text=page, content_type="text/html")

    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/api/all", api)])
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alamet yerel izleme arayüzü (salt okunur)")
    parser.add_argument("--db", default="data/alamet.sqlite")
    parser.add_argument("--universe", default="data/universe.json")
    parser.add_argument("--halt-file", default="data/HALT")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    app = make_app(Path(args.db), Path(args.universe), Path(args.halt_file))
    web.run_app(app, host="127.0.0.1", port=args.port, print=lambda *_: None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
