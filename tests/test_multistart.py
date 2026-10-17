from types import SimpleNamespace

import numpy as np
import pytest

from alamet.multistart import BARS_PER_DAY, start_bars, window_metrics


def test_start_bars_cover_range_with_step():
    starts = start_bars(total=1000, warmup=100, horizon=300, step=200)
    assert starts == [100, 300, 500, 700]
    assert all(s - 100 >= 0 and s + 300 <= 1000 for s in starts)


def test_window_metrics_return_drawdown_and_idle_gap():
    realised = np.array([900.0, 1000.0, 1100.0, 990.0, 1210.0, 1210.0])
    trades = [SimpleNamespace(open_bar=1), SimpleNamespace(open_bar=2), SimpleNamespace(open_bar=0)]
    result = SimpleNamespace(realised=realised, trades=trades)
    ret, dd, opened, idle = window_metrics(result, begin=1, bars_after=5)
    assert ret == pytest.approx(21.0)
    assert dd == pytest.approx(10.0)
    assert opened == 2
    assert idle == pytest.approx(4 / BARS_PER_DAY)


def test_run_one_uses_fixed_length_window(monkeypatch):
    from alamet import multistart

    seen = {}

    class Fake:
        def __init__(self, bars):
            self.bars = bars
            self.time = np.arange(bars)
            self.bar_seconds = 900

        def slice_from(self, start):
            return Fake(self.bars - start)

        def slice_to(self, stop):
            seen["bars"] = stop
            return Fake(stop)

    monkeypatch.setattr(
        multistart, "_data", lambda root: (Fake(10_000), SimpleNamespace(time=None))
    )
    monkeypatch.setattr(multistart, "daily_pack", lambda daily, *args: None)
    monkeypatch.setattr(multistart, "map_to_bars", lambda *a, **k: None)
    monkeypatch.setattr(multistart, "day_inputs", lambda mapped: None)
    monkeypatch.setattr(
        multistart,
        "run_system",
        lambda *a, **k: SimpleNamespace(realised=np.ones(seen["bars"]), trades=[]),
    )
    row = multistart.run_one(("S0", "zero", 4000, 4000, 960, "x"))
    assert seen["bars"] == 4000 + 960
    assert row.trades == 0 and row.idle_days == pytest.approx(960 / BARS_PER_DAY)
