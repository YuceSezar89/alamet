"""
"Ya şöyle olsaydı": ayarları değiştirip GEÇMİŞ veride sonucu görmek (bota, veritabanına, borsaya dokunmaz).

Ayar yolları ve değer çevirisi `overrides.params_with` ile aynıdır (TradingView ile doğrulanan yol).
Panel (sinyal hattı) pahalıdır; yalnızca panel ayarları değişince yeniden hesaplanır, işlem
(motor) ayarları değişince son panel yeniden kullanılır.

  fields()                 -> arayüzdeki düzenlenebilir ayar alanları (+ varsayılanlar)
  run(overrides, window, cost, root) -> sonuç sözlüğü (ölçütler, kasa eğrisi, işlemler)
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alamet.ayar_etiketleri import ETIKETLER, GRUPLAR
from alamet.backtest import summarise
from alamet.binance_data import DEFAULT_ROOT, infer_tf, load_universe_binance
from alamet.daily import daily_from_bars, daily_pack, day_inputs, map_to_bars
from alamet.overrides import params_with
from alamet.panel import build_panel
from alamet.stages import stage_params
from alamet.system import run_system
from alamet.timeframe import SignalTf
from alamet.universe import SYMBOLS, Universe
from alamet.webviews import METHODS, SELECTOR_MODES, flatten

BASE_STAGE = "S17"
EXCLUDED = ("panel.symbol.signal_tf_seconds",)
WINDOWS = {"all": "Tüm geçmiş", "4000": "Son 4000 bar (Pine penceresi)"}
COSTS = {"ayar": "Ayarlardaki maliyet", "gerçekçi": "Gerçekçi (komisyon %0,05, kayma %0,02)"}
REALISTIC = ("trade.fee_in=0.05", "trade.fee_out=0.05", "trade.slippage=0.02")
WINDOW_BARS = 4000
MAX_POINTS = 600
MAX_TRADES = 400
TEXT_OPTIONS = {
    "panel.own_mode": ["Önceki hareket", "Ort + σ"],
    "panel.tgt_mode": ["Yakın hedef", "Büyük hedef (aylık+yıllık)", "Yakın + büyük ortalaması"],
    "panel.dr_src": ["Kendi geçmişi", "Emtialar arası", "Karma"],
    "panel.rank_mode": ["ATR cinsinden (volatilite normalize)", "Ham %"],
}
BOOL_LABELS = {
    "trade.risk_mode": ("Sabit marj × kaldıraç", "Risk %"),
    "panel.shadow.risk_mode": ("Sabit marj × kaldıraç", "Risk %"),
    "trade.cap_risk_mode": ("Marj (notional / maks. kaldıraç)", "Risk (1R kaybı)"),
}


class InvalidSetting(ValueError):
    """Bilinmeyen alan ya da geçersiz değer."""


def _label(path: str) -> tuple[str, str, int]:
    group, label, order = ETIKETLER.get(path, ("Diğer", path, 10_000))
    return group, label, order


def _field(path: str, value, label_suffix: str = "", order_shift: float = 0.0) -> dict:
    group, label, order = _label(path.rsplit(".", 1)[0] if label_suffix else path)
    row = {
        "code": path, "group": group, "label": label + label_suffix,
        "order": order + order_shift, "default": value,
    }  # fmt: skip
    if path in BOOL_LABELS:
        off, on = BOOL_LABELS[path]
        row.update(kind="select", options=[{"v": False, "l": off}, {"v": True, "l": on}])
    elif path == "panel.selector.mode":
        row.update(kind="select", options=[{"v": i, "l": n} for i, n in enumerate(SELECTOR_MODES)])
    elif path in TEXT_OPTIONS:
        row.update(kind="select", options=[{"v": o, "l": o} for o in TEXT_OPTIONS[path]])
    elif isinstance(value, bool):
        row["kind"] = "bool"
    elif isinstance(value, int):
        row["kind"] = "int"
    elif isinstance(value, float):
        row["kind"] = "float"
    else:
        row["kind"] = "text"
    return row


def fields() -> list[dict]:
    """Düzenlenebilir ayarlar: Hoca'nın başlık/grup sırasıyla, varsayılan değerleriyle."""
    panel, trade = stage_params(BASE_STAGE)
    rows: list[dict] = []
    for root, params in (("panel", panel), ("trade", trade)):
        for key, value in flatten(params):
            path = f"{root}.{key}"
            if path in EXCLUDED:
                continue
            if isinstance(value, (tuple, list)):
                for i, item in enumerate(value):
                    code = f"{path}.{i}"
                    if path == "panel.selector.seq_order":
                        row = _field(code, int(item), f" · {i + 1}. sıra", i / 100)
                        row.update(
                            kind="select", options=[{"v": j, "l": n} for j, n in enumerate(METHODS)]
                        )
                    elif path == "panel.selector.enabled":
                        row = _field(code, bool(item), f" · {METHODS[i]}", i / 100)
                    else:
                        row = _field(code, item, f" · {i + 1}", i / 100)
                    rows.append(row)
            else:
                rows.append(_field(path, value))
    rank = {name: i for i, name in enumerate([*GRUPLAR, "Diğer"])}
    rows.sort(key=lambda r: (rank.get(r["group"], len(rank)), r["order"]))
    return rows


def _index() -> dict[str, dict]:
    return {row["code"]: row for row in fields()}


def to_sets(overrides: dict[str, object], cost: str = "ayar") -> list[str]:
    """Arayüz değerlerini `params_with` için `yol=değer` metinlerine çevirir (doğrulayarak)."""
    known = _index()
    sets: list[str] = []
    for code, value in overrides.items():
        spec = known.get(code)
        if spec is None:
            raise InvalidSetting(f"bilinmeyen ayar: {code}")
        sets.append(f"{code}={_raw(spec, value)}")
    if cost == "gerçekçi":
        sets += list(REALISTIC)
    elif cost != "ayar":
        raise InvalidSetting(f"bilinmeyen maliyet: {cost}")
    return sets


def _raw(spec: dict, value) -> str:
    kind = spec["kind"]
    if kind == "select":
        if value not in [o["v"] for o in spec["options"]] or isinstance(value, bool) != isinstance(
            spec["options"][0]["v"], bool
        ):
            raise InvalidSetting(f"{spec['code']}: geçersiz seçim {value!r}")
        return str(value).lower() if isinstance(value, bool) else str(value)
    if kind == "bool":
        if not isinstance(value, bool):
            raise InvalidSetting(f"{spec['code']}: açık/kapalı olmalı")
        return "true" if value else "false"
    if kind in ("int", "float"):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise InvalidSetting(f"{spec['code']}: sayı olmalı")
        if kind == "int" and float(value) != int(value):
            raise InvalidSetting(f"{spec['code']}: tam sayı olmalı")
        return repr(float(value)) if kind == "float" else str(int(value))
    return str(value)


# ----------------------------------------------------------------------------- çalıştırma
@dataclass
class _Cache:
    root: str | None = None
    universe: Universe | None = None
    tf: SignalTf | None = None
    panel_key: tuple | None = None
    panel: object = None


_CACHE = _Cache()


def _universe(root: Path):
    if _CACHE.root != str(root):
        tf = infer_tf(root)
        _CACHE.universe = load_universe_binance(root, SYMBOLS, tf=tf)
        _CACHE.tf, _CACHE.root = tf, str(root)
        _CACHE.panel_key, _CACHE.panel = None, None
    return _CACHE.universe, _CACHE.tf


def run(
    overrides: dict[str, object],
    window: str = "all",
    cost: str = "ayar",
    root: str | Path = DEFAULT_ROOT,
) -> dict:
    """Seçilen ayarlarla geçmişi oynatır ve özet döner. Saf hesap: dışarıya hiçbir şey yazmaz."""
    if window not in WINDOWS:
        raise InvalidSetting(f"bilinmeyen pencere: {window}")
    sets = to_sets(overrides, cost)
    universe, tf = _universe(Path(root))
    panel_p, trade_p = params_with(BASE_STAGE, sets, tf.seconds)
    start = 0 if window == "all" else max(universe.bars - WINDOW_BARS, 0)
    daily = daily_from_bars(universe, tf.seconds)
    pack = daily_pack(daily, panel_p.dr_rsi_len, panel_p.dr_look)
    day = day_inputs(map_to_bars(pack, daily.time, universe.time, tf_seconds=tf.seconds))
    key = (repr(panel_p), window, str(root))
    if _CACHE.panel_key != key:
        _CACHE.panel = None
        _CACHE.panel = build_panel(universe, panel_p, True, start, None, day)
        _CACHE.panel_key = key
    result = run_system(universe, panel_p, trade_p, True, panel=_CACHE.panel, start=start, day=day)
    return _summary(result, tf, trade_p, window, cost, overrides)


def _summary(result, tf, trade_p, window, cost, overrides) -> dict:  # noqa: PLR0913
    """Motor sonuçları `start`'tan itibaren indekslidir (`panel.universe` kesilmiş evren)."""
    universe = result.panel.universe
    done = [t for t in result.trades if t.close_bar >= 0]
    bars = universe.bars
    metrics = summarise(BASE_STAGE, cost, done, result, bars, trade_p.start_cap, tf.bars_per_year)
    times = universe.time
    equity = np.asarray(result.realised, dtype=float)
    step = max(len(equity) // MAX_POINTS, 1)
    idx = list(range(0, len(equity), step))
    if idx[-1] != len(equity) - 1:
        idx.append(len(equity) - 1)
    ordered = sorted(done, key=lambda t: t.close_bar, reverse=True)[:MAX_TRADES]
    return {
        "window": window,
        "cost": cost,
        "overrides": overrides,
        "from": int(times[0]),
        "to": int(times[-1]),
        "bars": int(bars),
        "metrics": {
            "trades": metrics.trades, "start_cap": metrics.start_cap, "final": metrics.final,
            "ret": metrics.ret, "max_dd": metrics.max_dd, "win_rate": metrics.win_rate,
            "profit_factor": _finite(metrics.profit_factor), "avg_r": metrics.avg_r,
            "long_share": metrics.long_share, "top10_share": _finite(metrics.top10_share),
            "years": metrics.years,
        },  # fmt: skip
        "equity": [[int(times[i]), round(float(equity[i]), 4)] for i in idx],
        "trades": [
            {
                "symbol": universe.symbols[t.symbol], "side": int(t.side),
                "open": int(universe.time[t.open_bar]), "close": int(universe.time[t.close_bar]),
                "entry": _finite(t.entry), "exit": _finite(t.exit), "usd": _finite(t.usd),
                "r": _finite(t.r), "reason": int(t.reason),
            }  # fmt: skip
            for t in ordered
        ],
    }


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


# ------------------------------------------------------------------------ arka plan işleri
IDLE_SECONDS = 600
KEEP_JOBS = 30


def _init_worker() -> None:
    """Çalışan bota öncelik bırak."""
    import os  # pylint: disable=import-outside-toplevel

    try:
        os.nice(10)
    except OSError:
        pass


class Jobs:
    """Tek işçili kuyruk: koşular sırayla çalışır, bellek yoğun işçi boşta kalınca kapatılır.

    Sonuçlar (ayar, pencere, maliyet) anahtarıyla önbelleklenir; aynı istek anında döner.
    """

    def __init__(self, root: Path) -> None:
        import time  # pylint: disable=import-outside-toplevel

        self.root, self.clock = Path(root), time.time
        self.pool = None
        self.jobs: dict[str, dict] = {}
        self.results: dict[str, dict] = {}
        self.last_used = self.clock()

    @staticmethod
    def key(overrides: dict, window: str, cost: str) -> str:
        import json  # pylint: disable=import-outside-toplevel

        return json.dumps([sorted(overrides.items()), window, cost], default=str)

    def submit(self, loop, overrides: dict, window: str, cost: str) -> dict:
        import uuid  # pylint: disable=import-outside-toplevel

        to_sets(overrides, cost)  # doğrula: geçersizse InvalidSetting
        if window not in WINDOWS:
            raise InvalidSetting(f"bilinmeyen pencere: {window}")
        key = self.key(overrides, window, cost)
        job = {"id": uuid.uuid4().hex[:12], "status": "queued", "key": key, "created": self.clock()}
        if key in self.results:
            job.update(status="done", result=self.results[key])
        else:
            self.jobs[job["id"]] = job
            loop.create_task(self._work(loop, job, overrides, window, cost))
        self.jobs[job["id"]] = job
        for old in list(self.jobs)[:-KEEP_JOBS]:
            del self.jobs[old]
        return job

    async def _work(self, loop, job, overrides, window, cost) -> None:
        if self.pool is None:
            self.pool = ProcessPoolExecutor(max_workers=1, initializer=_init_worker)
        self.last_used = self.clock()
        job["status"] = "running"
        try:
            result = await loop.run_in_executor(
                self.pool, run, overrides, window, cost, str(self.root)
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            job.update(status="error", error=f"{type(exc).__name__}: {exc}")
        else:
            self.results[job["key"]] = result
            for old in list(self.results)[:-KEEP_JOBS]:
                del self.results[old]
            job.update(status="done", result=result)
        self.last_used = self.clock()

    def get(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)

    def busy(self) -> bool:
        return any(j["status"] in ("queued", "running") for j in self.jobs.values())

    def reap(self) -> bool:
        """Boştaysa işçiyi kapatır (bellek serbest kalır). Kapattıysa True."""
        if (
            self.pool is not None
            and not self.busy()
            and self.clock() - self.last_used > IDLE_SECONDS
        ):
            self.pool.shutdown(wait=False)
            self.pool = None
            return True
        return False

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None
