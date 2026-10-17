import numpy as np

from alamet.daily import DailyUniverse, daily_pack
from alamet.panel import PanelParams, build_panel
from alamet.symbol import SymbolParams
from alamet.system import day_changes
from alamet.universe import Universe

DAY = 86400


def _universe(n=3, bars=700, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, (bars, n)), axis=0))
    spread = close * 0.003
    open_ = np.vstack([close[:1], close[:-1]])
    time = 1_767_571_200 + np.arange(bars) * 900
    return Universe(
        tuple(f"S{i}" for i in range(n)), time, open_, np.maximum(open_, close) + spread,
        np.minimum(open_, close) - spread, close, rng.uniform(100, 200, (bars, n)),
    )  # fmt: skip


def test_day_changes_default_and_custom_open_hour():
    time = 1_767_571_200 + np.arange(96 * 2 + 20) * 900
    assert day_changes(time).sum() == 2
    custom = day_changes(time, 3)
    assert list(np.flatnonzero(custom)) == [12, 108, 204]
    assert ((time[custom] % DAY) // 3600 == 3).all()


def test_symbol_params_change_atr_and_signals():
    universe = _universe()
    base = build_panel(universe, PanelParams(), True)
    slow = build_panel(universe, PanelParams(symbol=SymbolParams(atr_len=30)), True)
    assert not np.allclose(np.nan_to_num(base.atr), np.nan_to_num(slow.atr))


def test_daily_open_hour_moves_the_daily_open():
    universe = _universe()
    base = build_panel(universe, PanelParams(), True)
    shifted = build_panel(universe, PanelParams(daily_open_hour=3), True)
    assert not np.array_equal(np.nan_to_num(base.daily_open), np.nan_to_num(shifted.daily_open))


def test_daily_pack_uses_devisso_parameters():
    closes = 100 + np.cumsum(np.random.default_rng(1).normal(0, 1, (120, 2)), axis=0)
    daily = DailyUniverse(("A", "B"), 1_767_571_200 + np.arange(120) * DAY, closes, closes + 1,
                          closes - 1, closes)  # fmt: skip
    base = daily_pack(daily)
    custom = daily_pack(daily, rsi_len=6, look=30)
    assert base.shape == custom.shape
    assert not np.allclose(np.nan_to_num(base[:, :, 18]), np.nan_to_num(custom[:, :, 18]))
