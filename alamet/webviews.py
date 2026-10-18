"""
Arayüz verisi: kayıtlı canlı sistem durumundan Hoca'nın 5 tablosu, tüm semboller, ayarlar ve grafikler.

Tabloların sayıları `views.py`'den gelir (TradingView'de doğrulanan istatistik motoru). Burada yalnızca
canlı sistemden (`StepSystem`) bağlam kurulur ve sonuçlar JSON'a uygun hale getirilir. Hiçbir şey
yazılmaz, bota dokunulmaz.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd

from alamet.ayar_etiketleri import ETIKETLER, GRUPLAR
from alamet.views import (
    ViewContext,
    cash_view,
    engine_view,
    method_symbol_view,
    reason_view,
    trades_view,
)

LOG_STATUS = {0: "AÇIK", 1: "Kâr kilidi / TP", 2: "Sıralama", 3: "Sinyal", 4: "Stop / zaman"}
R_EDGES = (-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
DAY = 86400


def clean(value):
    """NumPy ve özel kayan nokta değerlerini JSON'a uygun yapar (NaN → None, ∞ → "∞")."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "∞" if value > 0 else "-∞"
    return value


def clean_dict(data: dict) -> dict:
    return {key: clean(value) for key, value in data.items()}


def records(frame: pd.DataFrame) -> list[dict]:
    return [clean_dict(row) for row in frame.to_dict("records")]


def shadow_state(system) -> dict:
    book = system.book
    return {
        "side": book.side, "notional": book.notional, "entry": book.entry, "net": book.net,
        "count": book.count, "wins": book.wins, "auto": system.selector.auto_m,
    }  # fmt: skip


def view_context(system, names: Sequence[str], close: np.ndarray) -> ViewContext:
    selector = system.pp.selector
    return ViewContext(
        system.engine, np.asarray(close, dtype=float), list(names), system.tp,
        shadow_state(system), system.pp.shadow, tuple(selector.enabled), selector.mode == 6,
        selector.fallback_method,
    )  # fmt: skip


def hoca_tables(system, names: Sequence[str], close: np.ndarray) -> dict:
    """Pine'ın 5 tablosu: İşlemler, Kasa, Sebep analizi, Kasa motoru, Yöntem × Emtia."""
    ctx = view_context(system, names, close)
    trades = trades_view(ctx)
    log_rows = records(trades.rows)
    for row in log_rows:
        row["name"] = names[row["symbol"]]
        row["status_text"] = LOG_STATUS.get(row["status"], str(row["status"]))
    reasons = reason_view(ctx)
    engine = engine_view(ctx)
    methods = method_symbol_view(ctx)
    return {
        "trades": {
            "rows": log_rows, "totals": clean_dict(trades.totals),
            "header": clean_dict(trades.header),
        },
        "cash": clean_dict(cash_view(ctx)),
        "reasons": {"summary": clean_dict(reasons.summary), "rows": records(reasons.rows)},
        "engine": {"header": clean_dict(engine.header), "rows": records(engine.rows)},
        "methods": {
            "cells": records(methods.cells), "best": records(methods.best),
            "engine": records(methods.engine), "totals": records(methods.totals),
            "real_total": clean(methods.real_total),
        },
    }  # fmt: skip


def symbols_table(system, names: Sequence[str], close: np.ndarray) -> list[dict]:
    """Her sembolün anlık durumu: sinyal, seçili yöntem, ayrışma takibi, açık işlem."""
    selector, tracker, state = system.selector, system.tracker, system.engine.s
    out = []
    for k, name in enumerate(names):
        side = int(selector.cb_dir[k])
        start = float(selector.cb_start[k])
        price = float(close[k]) if k < len(close) else math.nan
        move = (price / start - 1.0) * 100.0 * side if side and start > 0 else math.nan
        step = system.last[k]
        out.append(
            clean_dict(
                {
                    "symbol": name,
                    "side": side,
                    "method": int(selector.cb_m[k]),
                    "power": float(step.final_pct),
                    "move_pct": move,
                    "number": int(tracker.count[k]),
                    "top": bool(tracker.active[k]),
                    "double": bool(tracker.last_double(k)),
                    "lock": int(tracker.lock(k)),
                    "open": bool(state.active[k]),
                    "price": price,
                }
            )
        )
    return out


def flatten(value, prefix: str = "") -> list[tuple[str, object]]:
    """İç içe ayar kayıtlarını (nokta yoluyla) düz listeye çevirir; değerler ham kalır."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        items = [(f.name, getattr(value, f.name)) for f in dataclasses.fields(value)]
    elif isinstance(value, dict):
        items = list(value.items())
    else:
        return [(prefix, value)]
    out: list[tuple[str, object]] = []
    for key, item in items:
        out += flatten(item, f"{prefix}.{key}" if prefix else str(key))
    return out


SELECTOR_MODES = (
    "TREND (tek)", "MOMENTUM (tek)", "VOLATİLİTE (tek)", "HACİM (tek)", "Aynı anda conf",
    "Sıralı conf", "Otomatik (emtia başına en iyi)",
)  # fmt: skip
METHODS = ("TREND", "MOMENTUM", "VOLATİLİTE", "HACİM")
MINUTE, HOUR = 60, 3600


def number_text(value: float) -> str:
    return f"{value:g}"


def format_setting(path: str, value) -> str:
    """Ayar değerini okunur Türkçe metne çevirir."""
    if path == "panel.selector.mode":
        return SELECTOR_MODES[int(value)]
    if path == "panel.selector.enabled":
        return ", ".join(m for m, on in zip(METHODS, value) if on) or "-"
    if path == "panel.selector.seq_order":
        return " → ".join(METHODS[int(i)] for i in value)
    if path in ("trade.risk_mode", "panel.shadow.risk_mode"):
        return "Risk %" if value else "Sabit marj × kaldıraç"
    if path == "trade.cap_risk_mode":
        return "Risk (1R kaybı)" if value else "Marj (notional / maks. kaldıraç)"
    if path == "panel.symbol.signal_tf_seconds":
        seconds = int(value)
        return f"{seconds // HOUR} saat" if seconds % HOUR == 0 else f"{seconds // MINUTE} dk"
    if isinstance(value, bool):
        return "Açık" if value else "Kapalı"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return number_text(float(value)) if isinstance(value, (float, np.floating)) else str(value)
    if isinstance(value, (tuple, list)):
        return ", ".join(format_setting("", item) for item in value)
    return "-" if value is None else str(value)


def settings_table(system) -> list[dict]:
    """Hoca'nın başlık ve gruplarıyla, okunur değerlerle. Kod adı yalnızca `code` alanındadır."""
    rows = []
    for root, params in (("panel", system.pp), ("trade", system.tp)):
        for key, value in flatten(params):
            path = f"{root}.{key}"
            group, label, order = ETIKETLER.get(path, ("Diğer", path, 10_000))
            rows.append(
                {
                    "group": group, "label": label, "value": format_setting(path, value),
                    "code": path, "order": order,
                }
            )  # fmt: skip
    rank = {name: i for i, name in enumerate([*GRUPLAR, "Diğer"])}
    rows.sort(key=lambda r: (rank.get(r["group"], len(rank)), r["order"]))
    return rows


def charts(trade_rows: list[dict], start_cap: float, reason_names: dict[int, str]) -> dict:
    """Kapanan işlemlerden kasa eğrisi, R dağılımı, günlük işlem sayısı ve çıkış nedenleri."""
    ordered = sorted(trade_rows, key=lambda t: (t["close_time"] or 0, t["trade_id"]))
    equity, total = [], start_cap
    for t in ordered:
        total += t["usd"] or 0.0
        equity.append({"t": t["close_time"], "v": round(total, 4)})
    bins = [0] * (len(R_EDGES) + 1)
    for t in ordered:
        r = t["r"]
        if r is not None:
            bins[int(np.searchsorted(R_EDGES, r, side="right"))] += 1
    labels = [f"< {R_EDGES[0]:g}"]
    labels += [f"{a:g} … {b:g}" for a, b in zip(R_EDGES, R_EDGES[1:])]
    labels += [f"≥ {R_EDGES[-1]:g}"]
    per_day = Counter((t["close_time"] or 0) // DAY * DAY for t in ordered if t["close_time"])
    reasons = Counter(reason_names.get(t["reason"], str(t["reason"])) for t in ordered)
    return {
        "equity": equity, "start_cap": start_cap,
        "r_hist": [{"label": a, "n": n} for a, n in zip(labels, bins)],
        "per_day": [{"t": d, "n": n} for d, n in sorted(per_day.items())],
        "reasons": [{"label": a, "n": n} for a, n in reasons.most_common()],
    }  # fmt: skip


def leg_stats(tracker, k: int) -> dict:
    """Bir sembolün ayrışma bacakları: son `max_div` bacağın hareketi (güncel yön lehine), çift
    ayrışma eşiği ve yön kilidi toplamları. `tracker.py` ile aynı hesap."""
    p = tracker.p
    n, md = int(tracker.count[k]), p.max_div
    sign = -1 if tracker.dir[k] == -1 else 1
    legs = []
    for j in range(max(1, n - md + 1), n + 1):
        value = float(tracker.res_mv[k, (j - 1) % md]) * sign
        legs.append({"no": j, "move": clean(value), "current": j == n})
    previous = [leg["move"] for leg in legs if not leg["current"] and leg["move"] is not None]
    if p.dbl_pos_only:
        previous = [v for v in previous if v > 0]
    threshold = mean = std = None
    if len(previous) >= p.min_prev:
        mean = sum(previous) / len(previous)
        var = sum(v * v for v in previous) / len(previous) - mean * mean
        std = math.sqrt(max(var, 0.0)) if len(previous) > 1 else 0.0
        threshold = mean + p.z_min * std
    up = down = 0.0
    for j in range(max(1, n - md + 1), n + 1):
        value = float(tracker.res_mv[k, (j - 1) % md])
        if not math.isnan(value) and value != 0.0:
            if value > 0:
                up += value
            else:
                down += -value
    total = up + down
    return {
        "number": n, "active": bool(tracker.active[k]), "side": int(tracker.dir[k]),
        "current": legs[-1]["move"] if legs else None, "legs": legs, "mean": mean, "std": std,
        "threshold": threshold, "z_min": p.z_min, "min_prev": p.min_prev,
        "double": bool(tracker.last_double(k)), "lock": int(tracker.lock(k)),
        "up": up, "down": down,
        "up_share": up / total * 100.0 if total > 0 else None,
        "down_share": down / total * 100.0 if total > 0 else None,
        "min_share": system_min_share(tracker),
    }  # fmt: skip


def system_min_share(tracker) -> float:
    return float(tracker.p.min_share)


def divergence_overview(system, names: Sequence[str]) -> list[dict]:
    """Şu an ayrışmada (ilk-2) olan semboller: güncel hareketin eşiğe oranı."""
    out = []
    for k, name in enumerate(names):
        if not system.tracker.active[k]:
            continue
        leg = leg_stats(system.tracker, k)
        cur, thr = leg["current"], leg["threshold"]
        ratio = cur / thr if cur is not None and thr not in (None, 0) and thr > 0 else None
        out.append(
            clean_dict(
                {
                    "symbol": name,
                    "side": leg["side"],
                    "number": leg["number"],
                    "current": cur,
                    "threshold": thr,
                    "ratio": ratio,
                    "double": leg["double"],
                    "ready": leg["double"] and leg["lock"] == leg["side"],
                }
            )
        )
    return sorted(out, key=lambda r: (not r["ready"], -(r["ratio"] or -9), r["symbol"]))


def divergence_detail(system, names: Sequence[str], k: int, series: dict) -> dict:
    """Seçilen sembolün ayrışma grafiği verisi. `series` fiyat, olay ve işlem listelerini taşır."""
    leg = leg_stats(system.tracker, k)
    tracker, state = system.tracker, system.engine.s
    leg["start_price"] = clean(float(tracker.entry[k])) if tracker.active[k] else None
    return {
        "symbol": names[k], "leg": clean_dict(leg) | {
            "legs": [clean_dict(x) for x in leg["legs"]],
        },
        "open_trade": clean_dict(
            {"side": int(state.side[k]), "entry": float(state.entry[k])}
        ) if state.active[k] else None,
        **series,
    }  # fmt: skip
