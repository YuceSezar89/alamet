"""Ya şöyle olsaydı: ayar alanları, değer doğrulama, geçmiş koşusu, panel önbelleği, iş kuyruğu."""

import asyncio
import json
import math
from concurrent.futures import ThreadPoolExecutor

import pytest
from aiohttp.test_utils import TestClient, TestServer

from alamet import whatif
from alamet.stages import stage_params
from alamet.timeframe import SignalTf
from alamet.webui import make_app
from alamet.webviews import flatten
from tests.test_stepper import random_universe


@pytest.fixture
def synthetic(monkeypatch):
    universe = random_universe(6, 96 * 10, 7)
    monkeypatch.setattr(whatif, "_universe", lambda _root: (universe, SignalTf(900)))
    monkeypatch.setattr(whatif, "_CACHE", whatif._Cache())  # noqa: SLF001
    return universe


def test_fields_cover_the_settings_with_their_defaults():
    rows = whatif.fields()
    codes = [r["code"] for r in rows]
    assert len(codes) == len(set(codes)) and len(rows) > 150
    assert "panel.symbol.signal_tf_seconds" not in codes
    assert "panel.symbol.ha_multipliers.2" in codes and "panel.selector.enabled.3" in codes
    panel, trade = stage_params("S17")
    expected = {f"panel.{k}": v for k, v in flatten(panel)} | {
        f"trade.{k}": v for k, v in flatten(trade)
    }
    for row in rows:
        assert not isinstance(row["default"], (tuple, list))
        if row["kind"] == "select":
            assert row["default"] in [o["v"] for o in row["options"]]
        if row["code"] in expected and not isinstance(expected[row["code"]], tuple):
            assert row["default"] == expected[row["code"]]


def test_values_become_override_strings_after_validation():
    sets = whatif.to_sets(
        {
            "trade.use_tp": True, "trade.tp_r": 1, "trade.max_slots": 3.0,
            "panel.selector.mode": 2, "panel.own_mode": "Ort + σ", "trade.risk_mode": False,
        }  # fmt: skip
    )
    assert sets == [
        "trade.use_tp=true", "trade.tp_r=1.0", "trade.max_slots=3", "panel.selector.mode=2",
        "panel.own_mode=Ort + σ", "trade.risk_mode=false",
    ]  # fmt: skip
    assert whatif.to_sets({}, "gerçekçi") == list(whatif.REALISTIC)


@pytest.mark.parametrize(
    "bad",
    [
        {"trade.nope": 1},
        {"trade.tp_r": float("nan")},
        {"trade.tp_r": True},
        {"trade.tp_r": "2"},
        {"trade.max_slots": 2.5},
        {"trade.use_tp": 1},
        {"panel.selector.mode": 99},
        {"trade.risk_mode": 1},
        {"panel.own_mode": "yok"},
    ],
)
def test_invalid_values_are_rejected(bad):
    with pytest.raises(whatif.InvalidSetting):
        whatif.to_sets(bad)


def test_unknown_cost_is_rejected():
    with pytest.raises(whatif.InvalidSetting):
        whatif.to_sets({}, "bedava")


def test_run_returns_json_safe_summary_and_overrides_change_the_result(synthetic):
    base = whatif.run({}, "all", "ayar", "x")
    json.dumps(base)
    assert base["bars"] == synthetic.bars and base["metrics"]["start_cap"] == 1000.0
    assert base["equity"][0][0] == int(synthetic.time[0]) and base["equity"][-1][0] == int(
        synthetic.time[-1]
    )
    assert all(math.isfinite(v) for _, v in base["equity"])
    bigger = whatif.run({"trade.start_cap": 2000.0}, "all", "ayar", "x")
    assert bigger["metrics"]["start_cap"] == 2000.0
    assert bigger["metrics"]["trades"] == base["metrics"]["trades"]
    realistic = whatif.run({}, "all", "gerçekçi", "x")
    assert realistic["cost"] == "gerçekçi"
    if base["metrics"]["trades"]:
        assert realistic["metrics"]["final"] != base["metrics"]["final"]


def test_default_run_equals_stage_run(synthetic):
    from alamet.daily import daily_from_bars, daily_pack, day_inputs, map_to_bars
    from alamet.system import run_system

    panel_p, trade_p = stage_params("S17")
    day = day_inputs(
        map_to_bars(
            daily_pack(daily_from_bars(synthetic, 900), panel_p.dr_rsi_len, panel_p.dr_look),
            daily_from_bars(synthetic, 900).time, synthetic.time, tf_seconds=900,
        )
    )  # fmt: skip
    direct = run_system(synthetic, panel_p, trade_p, True, start=0, day=day)
    mine = whatif.run({}, "all", "ayar", "x")
    assert mine["metrics"]["trades"] == len([t for t in direct.trades if t.close_bar >= 0])
    assert mine["metrics"]["final"] == pytest.approx(float(direct.realised[-1]))


def test_panel_is_reused_unless_panel_or_shadow_settings_change(synthetic):
    whatif.run({}, "all", "ayar", "x")
    first = whatif._CACHE.panel  # noqa: SLF001
    fast = {"trade.use_tp": True, "trade.tp_r": 1.0, "trade.quality.max_ext": 6.0}
    whatif.run(fast, "all", "ayar", "x")
    assert whatif._CACHE.panel is first  # noqa: SLF001
    whatif.run({"trade.start_cap": 3000.0}, "all", "ayar", "x")  # gölge kasa bundan türer
    assert whatif._CACHE.panel is not first  # noqa: SLF001
    first = whatif._CACHE.panel  # noqa: SLF001
    whatif.run({"panel.n_top": 3}, "all", "ayar", "x")
    assert whatif._CACHE.panel is not first  # noqa: SLF001


def test_short_window_on_short_data_starts_at_zero(synthetic):
    result = whatif.run({}, "4000", "ayar", "x")
    assert result["window"] == "4000" and result["bars"] == synthetic.bars


def test_jobs_run_cache_and_validate(synthetic, tmp_path):
    async def go():
        jobs = whatif.Jobs(tmp_path)
        jobs.pool = ThreadPoolExecutor(1)
        loop = asyncio.get_running_loop()
        job = jobs.submit(loop, {"trade.use_tp": True}, "all", "ayar")
        assert job["status"] == "queued"
        for _ in range(200):
            if jobs.get(job["id"])["status"] in ("done", "error"):
                break
            await asyncio.sleep(0.05)
        done = jobs.get(job["id"])
        assert done["status"] == "done" and done["result"]["metrics"]["trades"] >= 0
        again = jobs.submit(loop, {"trade.use_tp": True}, "all", "ayar")
        assert again["status"] == "done" and again["id"] != job["id"]
        with pytest.raises(whatif.InvalidSetting):
            jobs.submit(loop, {"trade.nope": 1}, "all", "ayar")
        with pytest.raises(whatif.InvalidSetting):
            jobs.submit(loop, {}, "yok", "ayar")
        jobs.last_used -= whatif.IDLE_SECONDS + 1
        assert jobs.reap() is True and jobs.pool is None

    asyncio.run(go())


def test_job_errors_are_reported_not_raised(synthetic, tmp_path, monkeypatch):
    def boom(*_args):
        raise ValueError("geçersiz birleşim")

    monkeypatch.setattr(whatif, "run", boom)

    async def go():
        jobs = whatif.Jobs(tmp_path)
        jobs.pool = ThreadPoolExecutor(1)
        job = jobs.submit(asyncio.get_running_loop(), {}, "all", "ayar")
        for _ in range(100):
            if jobs.get(job["id"])["status"] == "error":
                break
            await asyncio.sleep(0.05)
        assert "geçersiz birleşim" in jobs.get(job["id"])["error"]

    asyncio.run(go())


def test_http_endpoints(tmp_path):
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"symbols": [{"base": "BTC"}]}), encoding="utf-8")

    async def go():
        app = make_app(tmp_path / "t.sqlite", uni, tmp_path / "HALT")
        async with TestClient(TestServer(app)) as client:
            fields = await (await client.get("/api/whatif/fields")).json()
            bad = await client.post("/api/whatif", json={"overrides": {"trade.nope": 1}})
            worse = await client.post("/api/whatif", data="saçma")
            wrong = await client.post("/api/whatif", json={"overrides": [1]})
            unknown = await client.get("/api/whatif/yok")
            return fields, bad.status, worse.status, wrong.status, unknown.status

    fields, bad, worse, wrong, unknown = asyncio.run(go())
    assert len(fields["fields"]) > 150 and set(fields["windows"]) == {"all", "4000"}
    assert (bad, worse, wrong, unknown) == (400, 400, 400, 404)
