"""Sembol sayısından bağımsızlık: 20'nin ötesinde çökme yok, sembole özgü çıktılar evrenden bağımsız,
kova düzeni 20 sembolde Pine ile aynı."""

import numpy as np
import pytest

from alamet.engine import TradeParams
from alamet.events import CloseEvent
from alamet.kelly import KellyParams, bucket_count, ease_base
from alamet.panel import build_panel
from alamet.pool import PoolParams
from alamet.stages import panel_params
from alamet.stats import SYMBOL_BASE, cause_buckets, stats_layout
from alamet.system import run_system
from alamet.universe import Universe
from alamet.views import cash_view, engine_view, method_symbol_view, reason_view, view_context


def market(n, bars=1500, seed=5):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, (bars, n)), axis=0))
    spread = close * 0.003
    open_ = np.vstack([close[:1], close[:-1]])
    time = 1_767_571_200 + np.arange(bars) * 900
    return Universe(
        tuple(f"S{i}" for i in range(n)), time, open_, np.maximum(open_, close) + spread,
        np.minimum(open_, close) - spread, close, rng.uniform(100, 200, (bars, n)),
    )  # fmt: skip


def test_bucket_layout_is_pines_at_twenty_symbols_and_grows_with_the_universe():
    assert (cause_buckets(20), ease_base(20), bucket_count(20)) == (56, 56, 60)
    assert (cause_buckets(150), ease_base(150), bucket_count(150)) == (186, 186, 190)
    assert sum(size for name, size in stats_layout(8, 20) if name == "lsW") == 56
    assert dict(stats_layout(8, 150))["symN"] == 150 and dict(stats_layout(8, 150))["lsW"] == 186


@pytest.mark.parametrize("n", [60, 150])
def test_whole_system_runs_with_every_learning_layer_on_a_wide_universe(n):
    trade = TradeParams(
        max_slots=4, use_dyn=True, kelly=KellyParams(on=True), pool=PoolParams(on=True), dr_on=True
    )
    panel = panel_params(n_top=n // 6, max_slots=4)
    data = market(n)
    result = run_system(data, panel, trade, True, collect_stats=True, collect_events=True)
    closes = result.events.of(CloseEvent)
    assert len(closes) >= 5
    engine = result.engine
    assert engine.stats.symbol_count.sum() == len(closes) == engine.stats.count.sum()
    assert engine.stats.cause_wins.size == SYMBOL_BASE + n == engine.stats.cause_losses.size
    assert engine.kelly.n.size == bucket_count(n)
    assert result.stats.shape[1] == sum(size for _, size in stats_layout(8, n))
    names = [f"X{k}" for k in range(n)]
    ctx = view_context(result, panel, trade, names)
    reasons = reason_view(ctx).rows
    assert len(reasons[reasons.section == 12]) == n
    assert reasons[reasons.section == 12].trades.sum() == len(closes)
    assert len(engine_view(ctx).rows) == 2 + 4 + 3 + 2 + 4 + 2 + 2 + 4 + n
    assert len(method_symbol_view(ctx).cells) == 5 * n
    assert cash_view(ctx)["trades"] == len(closes)
    assert result.sides.shape == (data.bars, n)


def test_per_symbol_signal_outputs_do_not_depend_on_the_rest_of_the_universe():
    wide, narrow_n = market(60), 20
    narrow = Universe(
        wide.symbols[:narrow_n], wide.time, wide.open[:, :narrow_n], wide.high[:, :narrow_n],
        wide.low[:, :narrow_n], wide.close[:, :narrow_n], wide.volume[:, :narrow_n],
    )  # fmt: skip
    params = panel_params(n_top=4, max_slots=3)
    big, small = build_panel(wide, params, True), build_panel(narrow, params, True)
    for name in ("types", "starts", "strengths", "methods", "pcts", "base_scores", "atr", "power"):
        assert np.array_equal(
            getattr(big, name)[:, :narrow_n], getattr(small, name), equal_nan=True
        ), name
    assert big.rank.idx_long.shape[1] == 60 and small.rank.idx_long.shape[1] == 20


def test_cross_sectional_ranking_does_depend_on_the_universe():
    wide, narrow_n = market(60), 20
    narrow = Universe(
        wide.symbols[:narrow_n], wide.time, wide.open[:, :narrow_n], wide.high[:, :narrow_n],
        wide.low[:, :narrow_n], wide.close[:, :narrow_n], wide.volume[:, :narrow_n],
    )  # fmt: skip
    params = panel_params(n_top=4, max_slots=3)
    big, small = build_panel(wide, params, True), build_panel(narrow, params, True)
    assert big.rank.top.sum(axis=1).max() <= 2 * 4 and small.rank.top.sum(axis=1).max() <= 2 * 4
    in_wide = big.rank.top[:, :narrow_n].sum()
    assert in_wide < small.rank.top.sum()


def listed_late(n, late, first_bar, bars=1500, seed=5):
    base = market(n, bars, seed)
    close, high, low, open_, volume = (
        a.copy() for a in (base.close, base.high, base.low, base.open, base.volume)
    )
    for array in (close, high, low, open_, volume):
        array[:first_bar, late:] = np.nan
    return Universe(base.symbols, base.time, open_, high, low, close, volume)


def sliced_from(data, first):
    return Universe(
        data.symbols, data.time[first:], data.open[first:], data.high[first:], data.low[first:],
        data.close[first:], data.volume[first:],
    )  # fmt: skip


def test_symbols_listed_later_never_trade_before_listing_but_do_trade_after():
    n, late, first_bar = 40, 20, 700
    data = listed_late(n, late, first_bar)
    trade = TradeParams(
        max_slots=4, use_dyn=True, kelly=KellyParams(on=True), pool=PoolParams(on=True), dr_on=True
    )
    result = run_system(
        data, panel_params(n_top=8, max_slots=4), trade, True, collect_stats=True,
        collect_events=True,
    )  # fmt: skip
    assert (result.sides[:first_bar, late:] == 0).all()
    assert (result.notionals[:first_bar, late:] == 0).all()
    assert not result.panel.rank.top[:first_bar, late:].any()
    late_trades = [t for t in result.trades if t.symbol >= late]
    assert late_trades and all(t.open_bar >= first_bar for t in late_trades)
    assert {t.symbol for t in result.trades} & set(range(late)) and np.isfinite(
        result.realised
    ).all()


def test_a_late_listed_symbols_series_starts_at_its_own_first_bar():
    n, late, first_bar = 24, 12, 600
    data = listed_late(n, late, first_bar, bars=1400)
    params = panel_params(n_top=4, max_slots=3)
    wide = build_panel(data, params, True)
    reference = build_panel(sliced_from(data, first_bar), params, True)
    for name in ("types", "starts", "strengths", "methods", "pcts", "base_scores", "atr", "power"):
        assert np.array_equal(
            getattr(wide, name)[first_bar:, late:],
            getattr(reference, name)[:, late:],
            equal_nan=True,
        ), name
    assert np.isnan(wide.atr[:first_bar, late:]).all()
    assert np.isfinite(wide.atr[first_bar + 50 :, late:]).all()
