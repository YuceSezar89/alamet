"""Arayüz verisi: canlı sistem durumundan Hoca'nın 5 tablosu, tüm semboller, ayarlar, grafikler."""

import json

import numpy as np

from alamet.stages import stage_params
from alamet.stepper import StepSystem
from alamet.webviews import charts, clean, hoca_tables, settings_table, symbols_table
from tests.test_stepper import random_universe


def stepped_system(n=6, bars=96 * 8):
    universe = random_universe(n, bars, 7)
    panel, trade = stage_params("S17")
    system = StepSystem(n, panel, trade, "history", None, True, False)
    for i in range(bars):
        system.step(
            int(universe.time[i]), universe.open[i], universe.high[i], universe.low[i],
            universe.close[i], universe.volume[i],
        )  # fmt: skip
    return system, universe


def test_clean_makes_values_json_safe():
    assert clean(float("nan")) is None
    assert clean(float("inf")) == "∞" and clean(float("-inf")) == "-∞"
    assert clean(np.int64(3)) == 3 and clean(np.float64(1.5)) == 1.5


def test_five_hoca_tables_are_built_and_json_serialisable():
    system, universe = stepped_system()
    names = [f"S{i}" for i in range(universe.n)]
    tables = hoca_tables(system, names, universe.close[-1])
    assert set(tables) == {"trades", "cash", "reasons", "engine", "methods"}
    json.dumps(tables, allow_nan=False)
    assert tables["cash"]["start_cap"] == system.tp.start_cap
    assert {r["section"] for r in tables["reasons"]["rows"]} >= {1, 2, 3}
    assert {c["account"] for c in tables["methods"]["cells"]} == {0, 1, 2, 3, 4}


def test_symbols_table_has_one_row_per_symbol():
    system, universe = stepped_system()
    names = [f"S{i}" for i in range(universe.n)]
    rows = symbols_table(system, names, universe.close[-1])
    assert [r["symbol"] for r in rows] == names
    assert all(r["side"] in (-1, 0, 1) for r in rows)
    json.dumps(rows, allow_nan=False)


def test_settings_use_hocas_titles_groups_and_readable_values():
    system, _ = stepped_system(n=6, bars=96)
    rows = settings_table(system)
    by_code = {r["code"]: r for r in rows}
    assert "Diğer" not in {r["group"] for r in rows}
    start = by_code["trade.start_cap"]
    assert start["label"] == "Başlangıç kasası (USDT)" and start["value"] == "1000"
    assert by_code["trade.use_give"]["value"] == "Açık"
    assert by_code["panel.symbol.signal_tf_seconds"]["value"] == "15 dk"
    assert by_code["trade.risk_mode"]["label"] == "Pozisyon büyüklüğü modu"
    assert by_code["trade.risk_mode"]["value"] == "Risk %"
    assert by_code["panel.selector.mode"]["label"] == "Sinyal modu"
    assert by_code["panel.selector.mode"]["value"] == "Otomatik (emtia başına en iyi)"
    assert by_code["panel.selector.enabled"]["value"] == "TREND, MOMENTUM, VOLATİLİTE, HACİM"
    assert by_code["panel.selector.seq_order"]["value"] == "TREND → MOMENTUM → HACİM → VOLATİLİTE"
    assert not any("_" in r["label"] and "." in r["label"] for r in rows)


def test_every_setting_is_labelled_so_no_code_name_is_shown():
    from alamet.ayar_etiketleri import ETIKETLER  # pylint: disable=import-outside-toplevel
    from alamet.webviews import flatten  # pylint: disable=import-outside-toplevel

    system, _ = stepped_system(n=6, bars=96)
    paths = {f"panel.{k}" for k, _ in flatten(system.pp)} | {
        f"trade.{k}" for k, _ in flatten(system.tp)
    }
    assert paths <= set(ETIKETLER), sorted(paths - set(ETIKETLER))


def test_charts_build_equity_histogram_and_daily_counts():
    trades = [
        {"trade_id": 1, "close_time": 1_000, "usd": 2.0, "r": 1.2, "reason": 1},
        {"trade_id": 2, "close_time": 2_000, "usd": -1.0, "r": -0.8, "reason": 2},
        {"trade_id": 3, "close_time": 90_000, "usd": 3.0, "r": 4.0, "reason": 1},
    ]
    out = charts(trades, 1000.0, {1: "Kâr kilidi", 2: "Ters sinyal"})
    assert [p["v"] for p in out["equity"]] == [1002.0, 1001.0, 1004.0]
    assert sum(b["n"] for b in out["r_hist"]) == 3
    assert [d["n"] for d in out["per_day"]] == [2, 1]
    assert out["reasons"][0] == {"label": "Kâr kilidi", "n": 2}


def test_divergence_overview_and_detail_follow_the_tracker():
    from alamet.webviews import divergence_detail, divergence_overview  # pylint: disable=import-outside-toplevel

    system, universe = stepped_system()
    names = [f"S{i}" for i in range(universe.n)]
    overview = divergence_overview(system, names)
    json.dumps(overview, allow_nan=False)
    assert {r["symbol"] for r in overview} == {
        names[k] for k in range(len(names)) if system.tracker.active[k]
    }
    k = int(np.argmax(system.tracker.count))
    detail = divergence_detail(system, names, k, {"price": [], "events": [], "trades": []})
    json.dumps(detail, allow_nan=False)
    leg = detail["leg"]
    assert leg["number"] == int(system.tracker.count[k])
    assert 1 <= len(leg["legs"]) <= system.tracker.p.max_div
    assert leg["legs"][-1]["current"] is True
    if leg["threshold"] is not None:
        previous = [x["move"] for x in leg["legs"][:-1] if x["move"] is not None]
        assert abs(leg["threshold"] - (leg["mean"] + leg["z_min"] * leg["std"])) < 1e-9
        assert len(previous) >= leg["min_prev"]
