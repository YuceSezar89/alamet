"""
Kalıcı depo (SQLite): sistemin durumu, kapanan işlemler ve olaylar.

Bot kapanıp açılınca kaldığı yerden devam eder. Her kayıt tek bir işlemde (transaction) şunları
birlikte yazar: kapanan işlemler + olaylar + sistemin sıkıştırılmış anlık görüntüsü. Çökme olsa bile
kayıtlı durum ile işlem/olay geçmişi aynı bardadır; yeniden başlayınca kaçırılan barlar oynatılır ve
çift kayıt oluşmaz.

Açılışta sürüm ve ayar parmak izi denetlenir: uyuşmazsa `StateMismatch` ile reddedilir (sessizce
yeniden kurulmaz); kütüphane sürümü farkı yalnızca uyarıdır. Aynı depoyu iki süreç açamaz.

Anlık görüntü `pickle` ile yazılır; depo yalnızca bu süreç tarafından yazılır ve kullanıcıdan gelen
bir dosya asla yüklenmez.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import sqlite3
import time
import zlib
from collections.abc import Callable, Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alamet.stepper import StepSystem

SCHEMA_VERSION = 1
KEEP_LATEST = 20
KEEP_DAILY_DAYS = 30
SECONDS_PER_DAY = 86400

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS snapshots (
    bar_time INTEGER PRIMARY KEY, bar_index INTEGER NOT NULL, schema INTEGER NOT NULL,
    sha256 TEXT NOT NULL, state BLOB NOT NULL, created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS trades (
    trade_id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, side INTEGER, open_time INTEGER,
    close_time INTEGER, entry REAL, exit REAL, notional REAL, usd REAL, r REAL, pnl_pct REAL,
    reason INTEGER, rank INTEGER, method INTEGER, strength INTEGER, div_n INTEGER, mfe REAL,
    hold INTEGER, signal_pct REAL);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, bar_time INTEGER NOT NULL, phase INTEGER, seq INTEGER,
    kind TEXT NOT NULL, symbol TEXT, payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_time ON events (bar_time);
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trade_id INTEGER NOT NULL, kind TEXT NOT NULL,
    base TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL, quantity REAL, ref_price REAL,
    client_id TEXT NOT NULL UNIQUE, reduce_only INTEGER NOT NULL, reason TEXT, bar_time INTEGER,
    state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0, error TEXT);
CREATE TABLE IF NOT EXISTS orders (
    client_id TEXT PRIMARY KEY, trade_id INTEGER, kind TEXT, symbol TEXT, side TEXT,
    order_id INTEGER, status TEXT, quantity REAL, executed_qty REAL, avg_price REAL,
    ref_price REAL, slippage_pct REAL, commission REAL, realized_pnl REAL, latency_ms REAL,
    sent_at INTEGER, bar_time INTEGER);
CREATE TABLE IF NOT EXISTS positions (
    trade_id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, side INTEGER NOT NULL,
    quantity REAL NOT NULL, avg_price REAL NOT NULL, opened_at INTEGER, closed_at INTEGER);
"""
TRADE_COLUMNS = (
    "trade_id", "symbol", "side", "open_time", "close_time", "entry", "exit", "notional", "usd",
    "r", "pnl_pct", "reason", "rank", "method", "strength", "div_n", "mfe", "hold", "signal_pct",
)  # fmt: skip


class StateMismatch(RuntimeError):
    """Kayıtlı durum bu ayarlarla/sürümle uyumsuz: yeniden kurmadan devam edilemez."""


class StoreLocked(RuntimeError):
    """Depo başka bir süreç tarafından kullanılıyor."""


def fingerprint(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    panel_params,
    trade_params,
    symbols: Iterable[str],
    anchor_ms: int,
    daily_mode: str,
    first_day: int | None,
) -> dict:
    """Durumu üreten her ayarın parmak izi. `hard` alanlar uyuşmazsa durum kullanılamaz."""

    def digest(obj) -> str:
        return hashlib.sha256(repr(obj).encode("utf-8")).hexdigest()

    return {
        "hard": {
            "schema": SCHEMA_VERSION,
            "panel": digest(panel_params),
            "trade": digest(trade_params),
            "symbols": list(symbols),
            "anchor_ms": int(anchor_ms),
            "daily_mode": daily_mode,
            "first_day": first_day,
        },
        "soft": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def _clean(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return _clean(value.item())
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    return value


def event_payload(event) -> dict:
    data = asdict(event) if is_dataclass(event) else dict(vars(event))
    for skip in ("bar", "phase", "seq"):
        data.pop(skip, None)
    return {key: _clean(value) for key, value in data.items()}


class Store:
    """SQLite deposu. `with Store(path) as store:` kullanılır; dosya kilidi alınır."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_handle = self._acquire_lock()
        self.conn = sqlite3.connect(self.path, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self._migrate()
        if self.conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise sqlite3.DatabaseError(f"{self.path}: bütünlük denetimi başarısız")

    def _migrate(self) -> None:
        columns = {r[1] for r in self.conn.execute("PRAGMA table_info(outbox)")}
        if "stop_price" not in columns:
            self.conn.execute("ALTER TABLE outbox ADD COLUMN stop_price REAL NOT NULL DEFAULT 0")

    def _acquire_lock(self):
        import fcntl  # pylint: disable=import-outside-toplevel

        handle = open(
            f"{self.path}.lock", "w", encoding="utf-8"
        )  # pylint: disable=consider-using-with
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise StoreLocked(f"{self.path} başka bir süreçte açık") from exc
        return handle

    def close(self) -> None:
        self.conn.close()
        self._lock_handle.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ meta
    def _meta(self, key: str):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def _set_meta(self, key: str, value) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, json.dumps(value))
        )

    def check(self, config: dict) -> list[str]:
        """Kayıtlı ayarlarla karşılaştırır. Sert fark `StateMismatch`, yumuşak fark uyarı döner."""
        saved = self._meta("fingerprint")
        if saved is None:
            return []
        differences = [
            f"{key}: kayıtlı={saved['hard'].get(key)!r} şimdiki={value!r}"
            for key, value in config["hard"].items()
            if saved["hard"].get(key) != value
        ]
        if differences:
            raise StateMismatch(
                "Kayıtlı durum bu ayarlarla uyumsuz (yeniden kur gerekir): "
                + "; ".join(differences)
            )
        return [
            f"{key} sürümü değişmiş: {saved['soft'].get(key)} -> {value}"
            for key, value in config["soft"].items()
            if saved["soft"].get(key) != value
        ]

    # --------------------------------------------------------------- yazma
    def commit(
        self, system: StepSystem, symbols: list[str], config: dict, outbox: Iterable[dict] = ()
    ) -> None:
        """Kapanan işlemleri, olayları, emir niyetlerini (işlem kutusu) ve anlık görüntüyü TEK
        işlemde yazar: durum ile emir niyeti birlikte kaydedilir ya da hiç kaydedilmez."""
        bar_time = system.prev_time
        if bar_time is None:
            raise ValueError("Henüz hiç bar işlenmedi")
        closed, events = system.drain()
        times, system.bar_times = system.bar_times, {}
        try:
            state = zlib.compress(pickle.dumps(system, protocol=pickle.HIGHEST_PROTOCOL), 3)
            sha = hashlib.sha256(state).hexdigest()
            self.conn.execute("BEGIN IMMEDIATE")
            self._write_trades(closed, symbols)
            self._write_events(events, symbols, times)
            self._write_outbox(outbox)
            self.conn.execute(
                "INSERT OR REPLACE INTO snapshots VALUES (?, ?, ?, ?, ?, ?)",
                (bar_time, system.index, SCHEMA_VERSION, sha, state, int(time.time())),
            )
            self._set_meta("fingerprint", config)
            self._prune(bar_time)
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            system.undrain(closed, events, times)
            raise

    def _write_trades(self, trades, symbols: list[str]) -> None:
        rows = [
            (
                t.trade_id, symbols[t.symbol], t.side, t.open_time, t.close_time, t.entry, t.exit,
                t.notional, t.usd, t.r, t.pnl, t.reason, t.rank, t.method, t.strength, t.div_n,
                t.mfe, t.hold, t.signal_pct,
            )
            for t in trades
        ]  # fmt: skip
        rows = [tuple(_clean(v) for v in row) for row in rows]
        self.conn.executemany(
            f"INSERT OR REPLACE INTO trades({','.join(TRADE_COLUMNS)}) "
            f"VALUES ({','.join('?' * len(TRADE_COLUMNS))})",
            rows,
        )

    def _write_outbox(self, rows: Iterable[dict]) -> None:
        """Emir niyetleri; aynı `client_id` ikinci kez yazılmaz (kopya koruması)."""
        for row in rows:
            self.conn.execute(
                "INSERT OR IGNORE INTO outbox(trade_id, kind, base, symbol, side, quantity, "
                "ref_price, client_id, reduce_only, reason, bar_time, state, error, stop_price) "
                "VALUES (:trade_id, :kind, :base, :symbol, :side, :quantity, :ref_price, "
                ":client_id, :reduce_only, :reason, :bar_time, :state, :error, :stop_price)",
                {"reason": None, "error": None, "state": "pending", "stop_price": 0.0, **row},
            )

    def _write_events(self, events, symbols: list[str], times: dict[int, int]) -> None:
        rows = []
        for event in events:
            symbol = getattr(event, "symbol", None)
            rows.append(
                (
                    times[event.bar], event.phase, event.seq, type(event).__name__,
                    None if symbol is None else symbols[symbol],
                    json.dumps(event_payload(event), sort_keys=True),
                )  # fmt: skip
            )
        self.conn.executemany(
            "INSERT INTO events(bar_time, phase, seq, kind, symbol, payload) VALUES (?,?,?,?,?,?)",
            rows,
        )

    def _prune(self, newest: int) -> None:
        """Son `KEEP_LATEST` anlık görüntü + son `KEEP_DAILY_DAYS` günün her günün ilki kalır."""
        latest = [r[0] for r in self.conn.execute(
            "SELECT bar_time FROM snapshots ORDER BY bar_time DESC LIMIT ?", (KEEP_LATEST,)
        )]  # fmt: skip
        floor = newest - KEEP_DAILY_DAYS * SECONDS_PER_DAY
        daily = [r[0] for r in self.conn.execute(
            "SELECT MIN(bar_time) FROM snapshots WHERE bar_time >= ? GROUP BY bar_time / ?",
            (floor, SECONDS_PER_DAY),
        )]  # fmt: skip
        keep = set(latest) | set(daily)
        marks = ",".join("?" * len(keep))
        self.conn.execute(f"DELETE FROM snapshots WHERE bar_time NOT IN ({marks})", tuple(keep))

    # -------------------------------------------------------------- okuma
    def restore(self, config: dict) -> tuple[StepSystem | None, list[str]]:
        """En son sağlam anlık görüntüyü yükler. (sistem ya da None, uyarılar)."""
        warnings = self.check(config)
        rows = self.conn.execute(
            "SELECT bar_time, schema, sha256, state FROM snapshots ORDER BY bar_time DESC"
        ).fetchall()
        for bar_time, schema, sha, state in rows:
            if schema != SCHEMA_VERSION:
                raise StateMismatch(f"Anlık görüntü şeması {schema} != {SCHEMA_VERSION}")
            if hashlib.sha256(state).hexdigest() != sha:
                warnings.append(f"{bar_time} anlık görüntüsü bozuk, öncekine dönüldü")
                continue
            return pickle.loads(zlib.decompress(state)), warnings  # noqa: S301
        return None, warnings

    def trades(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM trades ORDER BY trade_id", self.conn)

    def events(self, kind: str | None = None) -> pd.DataFrame:
        query, args = "SELECT * FROM events", ()
        if kind is not None:
            query, args = query + " WHERE kind = ?", (kind,)
        return pd.read_sql_query(query + " ORDER BY id", self.conn, params=args)

    def note(self, kind: str, bar_time: int, symbol: str | None, payload: dict) -> None:
        """Adım dışı bir olayı (ör. veri düzeltmesi) hemen kaydeder."""
        self.conn.execute(
            "INSERT INTO events(bar_time, phase, seq, kind, symbol, payload) VALUES (?,?,?,?,?,?)",
            (bar_time, -1, 0, kind, symbol, json.dumps(_clean(payload), sort_keys=True)),
        )

    # --------------------------------------------------------- emir kayıtları
    def pending_outbox(self) -> list[dict]:
        cursor = self.conn.execute("SELECT * FROM outbox WHERE state='pending' ORDER BY id")
        names = [c[0] for c in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]

    def outbox(self, state: str | None = None) -> pd.DataFrame:
        query, args = "SELECT * FROM outbox", ()
        if state is not None:
            query, args = query + " WHERE state = ?", (state,)
        return pd.read_sql_query(query + " ORDER BY id", self.conn, params=args)

    def mark_outbox(
        self, row_id: int, state: str, error: str | None = None, attempts: int = 0
    ) -> None:
        self.conn.execute(
            "UPDATE outbox SET state=?, error=?, attempts=attempts+? WHERE id=?",
            (state, error, attempts, row_id),
        )

    def record_order(self, order: dict) -> None:
        names = list(order)
        self.conn.execute(
            f"INSERT OR REPLACE INTO orders({','.join(names)}) VALUES ({','.join('?' * len(names))})",
            [_clean(order[n]) for n in names],
        )

    def orders(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM orders ORDER BY sent_at, client_id", self.conn)

    def open_position(
        self, trade_id: int, symbol: str, side: int, qty: float, price: float, at: int
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO positions VALUES (?, ?, ?, ?, ?, ?, NULL)",
            (trade_id, symbol, side, qty, price, at),
        )

    def close_position(self, trade_id: int, at: int) -> None:
        self.conn.execute("UPDATE positions SET closed_at=? WHERE trade_id=?", (at, trade_id))

    def positions(self, open_only: bool = True) -> pd.DataFrame:
        query = "SELECT * FROM positions" + (" WHERE closed_at IS NULL" if open_only else "")
        return pd.read_sql_query(query + " ORDER BY trade_id", self.conn)

    def get_meta(self, key: str, default=None):
        value = self._meta(key)
        return default if value is None else value

    def set_meta(self, key: str, value) -> None:
        self._set_meta(key, value)

    def snapshot_times(self) -> list[int]:
        return [r[0] for r in self.conn.execute("SELECT bar_time FROM snapshots ORDER BY bar_time")]

    def problems(self) -> list[str]:
        """Tutarlılık denetimi: son anlık görüntüden sonraki olay/işlem olmamalı."""
        times = self.snapshot_times()
        if not times:
            return []
        latest = times[-1]
        out = []
        late_events = self.conn.execute(
            "SELECT COUNT(*) FROM events WHERE bar_time > ?", (latest,)
        ).fetchone()[0]
        late_trades = self.conn.execute(
            "SELECT COUNT(*) FROM trades WHERE close_time > ?", (latest,)
        ).fetchone()[0]
        if late_events:
            out.append(f"{late_events} olay son anlık görüntüden sonra")
        if late_trades:
            out.append(f"{late_trades} işlem son anlık görüntüden sonra")
        return out


def replay(
    system: StepSystem,
    universe,
    store: Store | None,
    symbols: list[str],
    config: dict,
    every: int = 2000,
    start: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """`universe` barlarını (start'tan) sırayla işler; her `every` barda ve sonunda kaydeder."""
    for i in range(start, universe.bars):
        system.step(
            universe.time[i], universe.open[i], universe.high[i], universe.low[i],
            universe.close[i], universe.volume[i],
        )  # fmt: skip
        last = i == universe.bars - 1
        if store is not None and ((i + 1 - start) % every == 0 or last):
            store.commit(system, symbols, config)
            if progress is not None:
                progress(i + 1, universe.bars)
