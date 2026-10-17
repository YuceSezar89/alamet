"""Kalıcı depo: kapat-aç sonrası kesintisiz çalışmayla birebir aynı, tutarlılık, reddetme, kilit."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from alamet.golden import load_window
from alamet.stages import stage_params
from alamet.stepper import StepSystem
from alamet.store import (
    SCHEMA_VERSION,
    StateMismatch,
    Store,
    StoreLocked,
    fingerprint,
    replay,
)
from tests.test_stepper import first_day_of, random_universe

GOLDEN = Path(__file__).parent / "data" / "golden" / "window.npz"
DAY = 86400


def build(universe, stage="S49", events=True, mode="history"):
    panel, trade = stage_params(stage)
    first_day = first_day_of(universe)
    system = StepSystem(universe.n, panel, trade, mode, first_day, False, events)
    config = fingerprint(
        panel, trade, universe.symbols, int(universe.time[0]) * 1000, mode, first_day
    )
    return system, list(universe.symbols), config


def feed(system, universe, begin, end):
    return [
        system.step(
            universe.time[i], universe.open[i], universe.high[i], universe.low[i],
            universe.close[i], universe.volume[i],
        )
        for i in range(begin, end)
    ]  # fmt: skip


def small_universe(bars=96 * 9, n=10, seed=4):
    return random_universe(n, bars, seed)


def test_stop_and_restart_equals_an_uninterrupted_run(tmp_path):
    universe = small_universe()
    straight, symbols, config = build(universe)
    full = feed(straight, universe, 0, universe.bars)
    cut = 400
    first, _, _ = build(universe)
    with Store(tmp_path / "a.sqlite") as store:
        feed(first, universe, 0, cut)
        store.commit(first, symbols, config)
    with Store(tmp_path / "a.sqlite") as store:
        restored, warnings = store.restore(config)
        assert restored is not None and warnings == [] and restored.index == cut
        rest = feed(restored, universe, cut, universe.bars)
        store.commit(restored, symbols, config)
        trades, events = store.trades(), store.events()
    for mine, theirs in zip(rest, full[cut:]):
        assert np.array_equal(mine.sides, theirs.sides)
        assert np.array_equal(mine.ks, theirs.ks)
        assert np.array_equal(mine.scores, theirs.scores, equal_nan=True)
        assert mine.closed_count == theirs.closed_count
    done = [t for t in straight.engine.trades if t.close_bar >= 0]
    assert len(trades) == len(done) > 0
    by_id = trades.set_index("trade_id")
    for t in done:
        row = by_id.loc[t.trade_id]
        assert (row.symbol, row.side, row.open_time, row.close_time) == (
            symbols[t.symbol], t.side, t.open_time, t.close_time
        )  # fmt: skip
        assert row.entry == t.entry and row.usd == t.usd and row.reason == t.reason
    assert len(events) == len(straight.events.ordered())
    assert restored.engine.stats.id_seq == straight.engine.stats.id_seq


@pytest.mark.skipif(not GOLDEN.exists(), reason="altın referans penceresi yok")
def test_restart_on_the_tradingview_window_matches_every_bar_including_statistics(tmp_path):
    window = load_window(GOLDEN)
    panel, trade = stage_params("S49")
    first_day = first_day_of(window)
    config = fingerprint(
        panel, trade, window.symbols, int(window.time[0]) * 1000, "history", first_day
    )
    straight = StepSystem(window.n, panel, trade, "history", first_day, True, True)
    full = feed(straight, window, 0, window.bars)
    one = StepSystem(window.n, panel, trade, "history", first_day, True, True)
    with Store(tmp_path / "g.sqlite") as store:
        feed(one, window, 0, 2000)
        store.commit(one, list(window.symbols), config)
    with Store(tmp_path / "g.sqlite") as store:
        restored, _ = store.restore(config)
        rest = feed(restored, window, 2000, window.bars)
    for mine, theirs in zip(rest, full[2000:]):
        assert np.array_equal(mine.stats, theirs.stats, equal_nan=True)
        assert np.array_equal(mine.sides, theirs.sides)


def test_memory_holds_only_open_trades_and_no_events_after_a_commit(tmp_path):
    universe = small_universe()
    system, symbols, config = build(universe)
    feed(system, universe, 0, universe.bars)
    assert any(t.close_bar >= 0 for t in system.engine.trades) and len(system.events.ordered()) > 0
    with Store(tmp_path / "m.sqlite") as store:
        store.commit(system, symbols, config)
        assert store.problems() == []
    assert all(t.close_bar < 0 for t in system.engine.trades)
    assert system.events.ordered() == [] and system.bar_times == {}


def test_a_failed_write_rolls_back_and_loses_nothing(tmp_path, monkeypatch):
    universe = small_universe()
    system, symbols, config = build(universe)
    feed(system, universe, 0, universe.bars)
    before_trades = len(system.engine.trades)
    before_events = len(system.events.ordered())
    with Store(tmp_path / "c.sqlite") as store:
        monkeypatch.setattr(
            Store, "_write_events", lambda *a: (_ for _ in ()).throw(OSError("disk"))
        )
        with pytest.raises(OSError):
            store.commit(system, symbols, config)
        assert store.snapshot_times() == [] and len(store.trades()) == 0
        assert len(system.engine.trades) == before_trades
        assert len(system.events.ordered()) == before_events
        monkeypatch.undo()
        store.commit(system, symbols, config)
        assert len(store.snapshot_times()) == 1 and len(store.trades()) > 0
        assert store.problems() == []


def test_different_settings_are_refused_and_new_libraries_only_warn(tmp_path):
    universe = small_universe(96 * 3)
    system, symbols, config = build(universe)
    feed(system, universe, 0, universe.bars)
    path = tmp_path / "d.sqlite"
    with Store(path) as store:
        store.commit(system, symbols, config)
    other_panel, other_trade = stage_params("S9")
    cases = {
        "ayar": fingerprint(other_panel, other_trade, symbols, config["hard"]["anchor_ms"], "history", config["hard"]["first_day"]),
        "sembol": fingerprint(*stage_params("S49"), symbols[:-1], config["hard"]["anchor_ms"], "history", config["hard"]["first_day"]),
        "çapa": fingerprint(*stage_params("S49"), symbols, config["hard"]["anchor_ms"] + 900_000, "history", config["hard"]["first_day"]),
        "günlük mod": fingerprint(*stage_params("S49"), symbols, config["hard"]["anchor_ms"], "live", config["hard"]["first_day"]),
    }  # fmt: skip
    with Store(path) as store:
        for name, wrong in cases.items():
            with pytest.raises(StateMismatch):
                store.restore(wrong)
        newer = json.loads(json.dumps(config))
        newer["soft"]["numpy"] = "999.0.0"
        restored, warnings = store.restore(newer)
        assert restored is not None and any("numpy" in w for w in warnings)


def test_a_second_process_cannot_open_the_same_store(tmp_path):
    path = tmp_path / "e.sqlite"
    with Store(path):
        with pytest.raises(StoreLocked):
            Store(path)
    Store(path).close()


def test_snapshot_retention_keeps_the_latest_and_one_per_recent_day(tmp_path):
    universe = small_universe(96 * 40, n=6, seed=7)
    system, symbols, config = build(universe, events=False)
    with Store(tmp_path / "r.sqlite") as store:
        for chunk in range(universe.bars // 48):
            feed(system, universe, chunk * 48, chunk * 48 + 48)
            store.commit(system, symbols, config)
        times = store.snapshot_times()
        newest = system.prev_time
    assert times[-1] == newest and len(times) < universe.bars // 48
    latest = sorted(times)[-20:]
    assert latest == times[-20:]
    older = [t for t in times if t not in latest]
    assert len({t // DAY for t in older}) == len(older)
    assert min(times) >= newest - 31 * DAY


def test_a_corrupted_snapshot_falls_back_to_the_previous_one(tmp_path):
    universe = small_universe(96 * 3)
    system, symbols, config = build(universe)
    path = tmp_path / "f.sqlite"
    with Store(path) as store:
        feed(system, universe, 0, 100)
        store.commit(system, symbols, config)
        feed(system, universe, 100, 150)
        store.commit(system, symbols, config)
        newest = store.snapshot_times()[-1]
    raw = sqlite3.connect(path)
    raw.execute("UPDATE snapshots SET state = zeroblob(10) WHERE bar_time = ?", (newest,))
    raw.commit()
    raw.close()
    with Store(path) as store:
        restored, warnings = store.restore(config)
    assert restored.index == 100 and any("bozuk" in w for w in warnings)


def test_replay_helper_saves_periodically_and_at_the_end(tmp_path):
    universe = small_universe(96 * 4)
    system, symbols, config = build(universe)
    seen = []
    with Store(tmp_path / "p.sqlite") as store:
        replay(
            system,
            universe,
            store,
            symbols,
            config,
            every=100,
            progress=lambda i, n: seen.append(i),
        )
        assert seen[-1] == universe.bars and len(seen) == universe.bars // 100 + 1
        restored, _ = store.restore(config)
    assert restored.index == universe.bars
    assert SCHEMA_VERSION == 1
