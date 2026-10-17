"""Adım yöneticisi: toplu `run_system` ile bar bar birebir aynı olmalı (altın referans dahil)."""

import os
from pathlib import Path

import numpy as np
import pytest

from alamet.daily import (
    daily_from_bars,
    daily_pack,
    day_inputs,
    map_live,
    window_daily,
)
from alamet.golden import load_window, run_stage
from alamet.stages import STAGES, stage_params
from alamet.stepper import StepSystem
from alamet.system import run_system
from alamet.universe import Universe

GOLDEN = Path(__file__).parent / "data" / "golden" / "window.npz"
DAY = 86400
REPRESENTATIVE = ("S0", "S5", "S9", "S11", "S12", "S13", "S17", "S29", "S36", "S43", "S44", "S48")
LAST_ONES = ("S49", "S50")

requires_golden = pytest.mark.skipif(not GOLDEN.exists(), reason="altın referans penceresi yok")


def stages():
    if os.environ.get("ALAMET_GOLDEN") == "all":
        return list(STAGES)
    return [*REPRESENTATIVE, *LAST_ONES]


def first_day_of(universe):
    return -(-int(universe.time[0]) // DAY) * DAY


def run_steps(universe, stage, mode="history", stats=True, events=True):
    panel, trade = stage_params(stage)
    system = StepSystem(universe.n, panel, trade, mode, first_day_of(universe), stats, events)
    records = [
        system.step(
            universe.time[i], universe.open[i], universe.high[i], universe.low[i],
            universe.close[i], universe.volume[i],
        )
        for i in range(universe.bars)
    ]  # fmt: skip
    return system, records


def same(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def assert_equal_to_batch(system, records, batch):
    def column(name):
        return np.array([getattr(r, name) for r in records])

    ks = np.column_stack([batch.realised, batch.peak, batch.max_dd])
    checks = {
        "sides": (column("sides"), batch.sides),
        "entries": (column("entries"), batch.entries),
        "notionals": (column("notionals"), batch.notionals),
        "kasa": (column("ks"), ks),
        "kapanan": (column("closed_count"), batch.closed_count),
        "tip": (column("types"), batch.panel.types),
        "başlangıç": (column("starts"), batch.panel.starts),
        "güç": (column("strengths"), batch.panel.strengths),
        "yöntem": (column("methods"), batch.panel.methods),
        "puan": (column("scores"), batch.panel.scores),
        "kapı": (column("own_ok"), batch.panel.own_ok),
        "hedef sırası": (column("t_rank"), batch.panel.divergence["t_rank"]),
        "ayrışma sayacı": (column("count"), batch.panel.divergence["count"]),
        "uygun": (column("eligible"), batch.panel.divergence["eligible"]),
    }
    if records[0].stats is not None and batch.stats is not None:
        checks["istatistik"] = (column("stats"), batch.stats)
    for name, (got, want) in checks.items():
        assert same(got, want), name
    assert len(system.engine.trades) == len(batch.trades)
    for mine, theirs in zip(system.engine.trades, batch.trades):
        assert (mine.symbol, mine.side, mine.open_bar, mine.close_bar, mine.reason) == (
            theirs.symbol, theirs.side, theirs.open_bar, theirs.close_bar, theirs.reason
        )  # fmt: skip
        assert same([mine.entry, mine.exit, mine.notional, mine.usd, mine.r],
                    [theirs.entry, theirs.exit, theirs.notional, theirs.usd, theirs.r])  # fmt: skip
    if system.events is not None and batch.events is not None:
        assert [type(e).__name__ for e in system.events.ordered()] == [
            type(e).__name__ for e in batch.events.ordered()
        ]


@requires_golden
@pytest.mark.parametrize("stage", stages())
def test_stepper_equals_batch_on_the_tradingview_window(stage):
    window = load_window(GOLDEN)
    batch = run_stage(window, stage, collect_stats=True, collect_events=True)
    system, records = run_steps(window, stage)
    assert len(system.engine.trades) > 0
    assert_equal_to_batch(system, records, batch)


@requires_golden
def test_live_daily_mode_equals_batch_with_live_day_inputs():
    window = load_window(GOLDEN)
    panel, trade = stage_params("S13")
    daily = window_daily(daily_from_bars(window), window.time[0])
    pack = daily_pack(daily, panel.dr_rsi_len, panel.dr_look)
    day = day_inputs(map_live(pack, daily, window))
    batch = run_system(
        window, panel, trade, True, start=0, day=day, collect_stats=True, collect_events=True
    )
    system, records = run_steps(window, "S13", mode="live")
    assert len(system.engine.trades) > 0
    assert_equal_to_batch(system, records, batch)


def random_universe(n, bars, seed):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, (bars, n)), axis=0))
    open_ = np.vstack([close[:1], close[:-1]])
    spread = close * 0.003
    time = 1_767_571_200 - (1_767_571_200 % DAY) + np.arange(bars) * 900
    arrays = [open_, np.maximum(open_, close) + spread, np.minimum(open_, close) - spread, close]
    volume = rng.uniform(100, 200, (bars, n))
    return Universe(tuple(f"S{i}" for i in range(n)), time, *arrays, volume)


def with_holes(universe, late_symbol, listed_bar, gap_symbol, gap):
    arrays = {
        name: getattr(universe, name).copy() for name in ("open", "high", "low", "close", "volume")
    }
    for array in arrays.values():
        array[:listed_bar, late_symbol] = np.nan
        array[gap[0] : gap[1], gap_symbol] = np.nan
    return Universe(universe.symbols, universe.time, **arrays)


@pytest.mark.parametrize("stage", ["S9", "S13", "S49"])
def test_stepper_equals_batch_for_late_listed_and_gapped_symbols(stage):
    universe = with_holes(random_universe(14, 96 * 18, 3), 5, 96 * 5 + 30, 8, (96 * 9, 96 * 9 + 40))
    batch = run_stage(universe, stage, collect_stats=True, collect_events=True)
    system, records = run_steps(universe, stage)
    assert_equal_to_batch(system, records, batch)
    assert (np.array([r.sides for r in records])[: 96 * 5 + 30, 5] == 0).all()


def test_a_repeated_or_older_bar_is_refused():
    universe = random_universe(6, 200, 1)
    panel, trade = stage_params("S9")
    system = StepSystem(universe.n, panel, trade)
    args = [
        universe.open[0],
        universe.high[0],
        universe.low[0],
        universe.close[0],
        universe.volume[0],
    ]
    system.step(int(universe.time[3]), *args)
    with pytest.raises(ValueError, match="ilerlemeli"):
        system.step(int(universe.time[3]), *args)
    with pytest.raises(ValueError, match="ilerlemeli"):
        system.step(int(universe.time[2]), *args)
    system.step(int(universe.time[4]), *args)


def test_unknown_daily_mode_is_rejected():
    panel, trade = stage_params("S9")
    with pytest.raises(ValueError):
        StepSystem(5, panel, trade, "ileri")
