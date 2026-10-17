"""Canlı yürütücü: sahte akışla (ağsız) toplu hesapla aynı, kapat-aç, hata durumunda durma."""

import asyncio

import numpy as np
import pandas as pd
import pytest

from alamet import binance_data as bd
from alamet.aggregator import Candle
from alamet.livefeed import BarSet, Correction
from alamet.runner import PROFILES, Runner, bar_arrays, profile_params
from alamet.stages import stage_params
from alamet.stepper import StepSystem
from alamet.timeframe import SignalTf
from tests.test_stepper import random_universe

TF = SignalTf(900)
SYMBOLS = tuple(f"S{i}" for i in range(8))


def write_cache(root, universe, upto):
    for k, base in enumerate(universe.symbols):
        frame = pd.DataFrame(
            {
                "open_time": universe.time[:upto].astype("int64") * 1000,
                "open": universe.open[:upto, k], "high": universe.high[:upto, k],
                "low": universe.low[:upto, k], "close": universe.close[:upto, k],
                "volume": universe.volume[:upto, k], "quote_volume": universe.volume[:upto, k] * 10,
            }
        )  # fmt: skip
        bd.save_cache(root, base, frame[bd.COLUMNS])


def bar_set(universe, i):
    candles = {
        base: Candle(
            int(universe.time[i]) * 1000, universe.open[i, k], universe.high[i, k],
            universe.low[i, k], universe.close[i, k], universe.volume[i, k], 1.0,
        )
        for k, base in enumerate(universe.symbols)
    }  # fmt: skip
    return BarSet(int(universe.time[i]) * 1000, candles)


def make_runner(tmp_path, universe, anchor_ms, sinks=(), name="r.sqlite", stage="S17"):
    panel, trade = stage_params(stage)
    logs = []
    runner = Runner(
        tmp_path / "cache", universe.symbols, TF, tmp_path / name, panel, trade, anchor_ms,
        "history", sinks, logs.append,
    )  # fmt: skip
    return runner, logs


def straight_run(universe, stage="S17"):
    panel, trade = stage_params(stage)
    system = StepSystem(universe.n, panel, trade, "history", None, False, True)
    return system, [
        system.step(
            universe.time[i], universe.open[i], universe.high[i], universe.low[i],
            universe.close[i], universe.volume[i],
        )
        for i in range(universe.bars)
    ]  # fmt: skip


async def go(coro):
    return await coro


def prepare_offline(runner):
    """Ağsız hazırlık: LiveFeed.prepare yerine önbellekten eşitleme."""
    system = runner.restore_or_create()
    runner.sync(system)
    runner.system = system
    return system


@pytest.fixture(scope="module")
def universe():
    return random_universe(8, 96 * 10, 5)


def test_history_replay_then_live_bars_equal_one_uninterrupted_run(tmp_path, universe):
    cut = 96 * 4
    write_cache(tmp_path / "cache", universe, cut)
    runner, _ = make_runner(tmp_path, universe, int(universe.time[0]) * 1000)
    system = prepare_offline(runner)
    assert system.index == cut
    for i in range(cut, universe.bars):
        asyncio.run(runner.on_bar(bar_set(universe, i)))
    straight, records = straight_run(universe)
    assert system.index == straight.index == universe.bars
    assert np.array_equal(system.engine.s.ks, straight.engine.s.ks)
    assert system.engine.stats.id_seq == straight.engine.stats.id_seq > 0
    trades = runner.store.trades()
    done = [t for t in straight.engine.trades if t.close_bar >= 0]
    assert len(trades) + sum(1 for t in system.engine.trades if t.close_bar >= 0) == len(done)
    runner.store.close()


def test_close_and_reopen_continues_exactly(tmp_path, universe):
    first_cut, second_cut = 96 * 3, 96 * 6
    write_cache(tmp_path / "cache", universe, first_cut)
    anchor = int(universe.time[0]) * 1000
    runner, _ = make_runner(tmp_path, universe, anchor)
    prepare_offline(runner)
    for i in range(first_cut, first_cut + 50):
        asyncio.run(runner.on_bar(bar_set(universe, i)))
    runner.store.close()
    write_cache(tmp_path / "cache", universe, second_cut)
    again, logs = make_runner(tmp_path, universe, anchor)
    system = prepare_offline(again)
    assert any("Durum yüklendi" in line for line in logs)
    assert system.index == second_cut
    for i in range(second_cut, universe.bars):
        asyncio.run(again.on_bar(bar_set(universe, i)))
    straight, _ = straight_run(universe)
    assert np.array_equal(system.engine.s.ks, straight.engine.s.ks)
    assert system.engine.stats.id_seq == straight.engine.stats.id_seq
    again.store.close()


def test_listeners_see_the_events_before_they_are_stored(tmp_path, universe):
    seen = []
    write_cache(tmp_path / "cache", universe, 96 * 5)
    runner, _ = make_runner(
        tmp_path, universe, int(universe.time[0]) * 1000,
        sinks=(lambda events, record, system: seen.append((len(events), record.time_s)),),
    )  # fmt: skip
    prepare_offline(runner)
    for i in range(96 * 5, 96 * 7):
        asyncio.run(runner.on_bar(bar_set(universe, i)))
    assert len(seen) == 96 * 2 and seen[0][1] == int(universe.time[96 * 5])
    events = runner.store.events()
    live_start = int(universe.time[96 * 5])
    assert sum(n for n, _ in seen) == int((events.bar_time >= live_start).sum()) > 0
    assert int((events.bar_time < live_start).sum()) > 0
    runner.store.close()


def test_an_error_during_a_step_stops_the_runner_and_keeps_the_last_state(
    tmp_path, universe, monkeypatch
):
    write_cache(tmp_path / "cache", universe, 96 * 5)
    runner, _ = make_runner(tmp_path, universe, int(universe.time[0]) * 1000)
    prepare_offline(runner)
    asyncio.run(runner.on_bar(bar_set(universe, 96 * 5)))
    committed = runner.store.snapshot_times()[-1]

    def broken(*args, **kwargs):
        raise ArithmeticError("boom")

    monkeypatch.setattr(StepSystem, "step", broken)
    with pytest.raises(ArithmeticError):
        asyncio.run(runner.on_bar(bar_set(universe, 96 * 5 + 1)))
    assert runner.store.snapshot_times()[-1] == committed and runner.store.problems() == []
    runner.store.close()


def test_a_repeated_bar_is_skipped_not_applied_twice(tmp_path, universe):
    write_cache(tmp_path / "cache", universe, 96 * 5)
    runner, logs = make_runner(tmp_path, universe, int(universe.time[0]) * 1000)
    system = prepare_offline(runner)
    asyncio.run(runner.on_bar(bar_set(universe, 96 * 5)))
    index = system.index
    asyncio.run(runner.on_bar(bar_set(universe, 96 * 5)))
    assert system.index == index and any("atlandı" in line for line in logs)
    runner.store.close()


def test_data_corrections_are_recorded_as_notes(tmp_path, universe):
    write_cache(tmp_path / "cache", universe, 96 * 5)
    runner, _ = make_runner(tmp_path, universe, int(universe.time[0]) * 1000)
    prepare_offline(runner)
    old = Candle(1000, 1.0, 2.0, 0.5, 1.5, 3.0, 4.0)
    runner.on_correction(Correction("S1", 900_000, old, Candle(1000, 1.0, 2.0, 0.5, 1.6, 3.0, 4.0)))
    runner.on_correction(Correction("S2", 1_800_000, None, old))
    notes = runner.store.events("VeriDüzeltmesi")
    assert list(notes.symbol) == ["S1", "S2"] and runner.corrections == 2
    assert '"eski": null' in notes.payload[1]
    runner.store.close()


def test_missing_symbols_become_nan_rows():
    universe = random_universe(4, 2, 1)
    bar = bar_set(universe, 0)
    del bar.candles["S2"]
    open_, high, low, close, volume = bar_arrays(bar, universe.symbols)
    assert np.isnan(close[2]) and np.isnan(volume[2]) and not np.isnan(close[[0, 1, 3]]).any()


def test_the_production_profile_is_the_full_original_defaults():
    assert PROFILES["orijinal"] == "S17"
    panel, trade = profile_params("orijinal")
    assert (panel, trade) == stage_params("S17")
    assert profile_params("S9") == stage_params("S9")
