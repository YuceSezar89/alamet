"""Görünüm tabloları: elle hesaplanmış senaryolar ve motor durumuyla tutarlılık."""

import math
from pathlib import Path

import numpy as np
import pytest

from alamet.engine import TradeEngine
from alamet.golden import load_window, run_stage
from alamet.kelly import KellyParams, ease_base
from alamet.shadow import ShadowParams
from alamet.stages import panel_params, stage_params
from alamet.system import run_system
from alamet.views import (
    ViewContext,
    cash_view,
    engine_view,
    method_symbol_view,
    reason_view,
    trades_view,
    view_context,
)
from tests.test_engine import N, mkbar, params
from tests.test_events import _named
from tests.test_stats import _universe

NAMES = [f"S{i}USDT" for i in range(N)]


def make(engine, close, **kw):
    return ViewContext(engine, np.array(close + [100.0] * (N - len(close))), NAMES, engine.p, **kw)


def played(**kw):
    """Sembol 0 LONG açılır (100), sembol 1 SHORT açılır; sembol 0 ters sinyalle 104'te kapanır."""
    engine = TradeEngine(N, params(use_give=False, **kw))
    engine.step(0, mkbar([100, 100, 100], types=(1, -1, 0), target=(1, 2, 0), scores=(2, 2, 0)))
    engine.step(1, mkbar([104, 101, 100], types=(-1, -1, 0), target=(0, 2, 0)))
    return engine


def test_trades_view_open_row_uses_live_price_and_closed_row_uses_exit():
    engine = played()
    view = trades_view(make(engine, [104, 98]))
    newest, oldest = view.rows.iloc[0], view.rows.iloc[1]
    assert (newest.trade_id, newest.symbol, newest.side, newest.status) == (2, 1, -1, 0)
    assert newest.price == 98.0 and newest.entry == 100.0
    assert newest.r == pytest.approx(2.0 / 1.5) and newest.pnl_pct == pytest.approx(2.0)
    notional = engine.trades[1].notional
    assert newest.usd == pytest.approx(notional * 0.02)
    assert (oldest.trade_id, oldest.status, oldest.price) == (1, 3, 104.0)
    assert oldest.r == pytest.approx(4.0 / 1.5) and oldest.usd == pytest.approx(
        engine.trades[0].usd
    )
    assert oldest.signal_pct == pytest.approx(engine.stats.log.signal_pct[0])


def test_trades_view_totals_split_closed_and_open():
    engine = played()
    view = trades_view(make(engine, [104, 98]))
    t = view.totals
    assert t["closed_count"] == 1 and t["closed_usd"] == pytest.approx(engine.trades[0].usd)
    assert t["closed_usd_long"] == pytest.approx(t["closed_usd"]) and t["closed_usd_short"] == 0
    assert t["open_count"] == 1 and t["open_short"] == 1 and t["open_long"] == 0
    assert t["open_usd"] == pytest.approx(engine.trades[1].notional * 0.02)
    assert t["total_usd"] == pytest.approx(t["closed_usd"] + t["open_usd"])
    assert t["total_pct"] == pytest.approx(t["total_usd"] / 1000.0 * 100.0)


def test_trades_view_empty_and_missing_price():
    engine = TradeEngine(N, params())
    assert trades_view(make(engine, [])).rows.empty
    engine.step(0, mkbar([100, 100, 100]))
    view = trades_view(make(engine, [math.nan]))
    assert math.isnan(view.rows.price[0]) and math.isnan(view.rows.usd[0])


def test_cash_view_numbers():
    engine = played()
    cash = cash_view(make(engine, [104, 98]))
    won = engine.trades[0].usd
    open_usd = engine.trades[1].notional * 0.02
    assert cash["realised"] == pytest.approx(1000 + won) and cash["open_usd"] == pytest.approx(
        open_usd
    )
    assert cash["equity"] == pytest.approx(1000 + won + open_usd)
    assert cash["net_pct"] == pytest.approx((won + open_usd) / 10.0)
    assert (cash["wins"], cash["losses"], cash["trades"], cash["win_pct"]) == (1, 0, 1, 100.0)
    assert math.isnan(cash["profit_factor"]) and cash["best_usd"] == pytest.approx(won)
    assert cash["long_usd"] == pytest.approx(won) and cash["short_usd"] == pytest.approx(open_usd)
    assert cash["peak"] >= cash["equity"] and cash["drawdown_now"] >= 0


def test_cash_view_profit_factor_and_streak_after_a_loss():
    engine = TradeEngine(N, params(use_give=False))
    engine.step(0, mkbar([100, 100, 100], types=(1, 0, 0)))
    engine.step(1, mkbar([104, 100, 100], types=(-1, 0, 0), target=(0, 0, 0)))
    engine.step(2, mkbar([104, 100, 100], types=(-1, 0, 0), target=(1, 0, 0)))
    engine.step(3, mkbar([108, 100, 100], types=(1, 0, 0), target=(0, 0, 0)))
    cash = cash_view(make(engine, [108]))
    gain, loss = engine.trades[0].usd, engine.trades[1].usd
    assert gain > 0 > loss
    assert cash["profit_factor"] == pytest.approx(gain / -loss) and cash["max_loss_streak"] == 1
    assert (cash["wins"], cash["losses"]) == (1, 1) and cash["worst_usd"] == pytest.approx(loss)


def test_reason_view_sections_each_cover_every_closed_trade():
    engine = played()
    view = reason_view(make(engine, [104, 98]))
    assert view.summary["closed"] == 1 and view.summary["gross_profit"] > 0
    assert (
        math.isnan(view.summary["profit_factor"]) is False
        and view.summary["profit_factor"] == math.inf
    )
    frame = view.rows
    for section in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12):
        assert frame[frame.section == section].trades.sum() == 1, section
    signal = frame[(frame.section == 1) & (frame.label.str.startswith("Ters"))].iloc[0]
    assert signal.trades == 1 and signal.share_pct == 100.0 and signal.win_pct == 100.0
    assert signal.is_top_win and not signal.is_top_loss
    assert signal.avg_r == pytest.approx(engine.trades[0].r) and signal.avg_hold == 1
    assert signal.win_usd_share == pytest.approx(100.0)
    empty = frame[(frame.section == 1) & (frame.label == "TP hedefi")].iloc[0]
    assert empty.trades == 0 and empty.share_pct == 0.0 and math.isnan(empty.avg_r)
    assert list(frame[frame.section == 12].label)[:2] == ["1-S0USDT", "2-S1USDT"]
    assert list(frame[frame.section == 7].label) == [
        "Tutma ≤ 3 bar",
        "Tutma 4–12 bar",
        "Tutma > 12 bar",
    ]


def test_reason_view_marks_the_largest_loss_bucket():
    engine = TradeEngine(N, params(use_give=False))
    engine.step(0, mkbar([100, 100, 100], types=(1, 0, 0)))
    engine.step(1, mkbar([96, 100, 100], types=(-1, 0, 0), target=(0, 0, 0)))
    view = reason_view(make(engine, [96]))
    loss_rows = view.rows[view.rows.is_top_loss]
    assert not loss_rows.empty and (loss_rows.loss_usd > 0).all()
    assert view.summary["losses"] == 1 and view.summary["payoff"] != view.summary["payoff"]
    assert view.summary["win_rate"] == 0.0 and view.summary["expectancy"] < 0


def feed_kelly(engine, bucket_set, rewards):
    for r in rewards:
        engine.kelly.accumulate(bucket_set, r)


def test_engine_view_matches_hand_computed_kelly_row():
    kp = KellyParams(on=True, prior=4.0, z=0.5, kelly=0.25, min_n=6.0, block=0.15)
    engine = TradeEngine(N, params(kelly=kp))
    direction = [9, 11, 15, 18, 20, 31, 33, ease_base(N) + 1, 36]
    feed_kelly(engine, direction, [1.0] * 8 + [0.5] * 2)
    view = engine_view(make(engine, [100]))
    long_row = view.rows[(view.rows.section == "YÖN") & (view.rows.bucket == 9)].iloc[0]
    n, total = 10.0, 9.0
    mu = total / (n + 4.0)
    sd = max(math.sqrt(max(((8 * 1.0 + 2 * 0.25) / n) - (total / n) ** 2, 0.25)), 0.0)
    assert view.header["sd"] == pytest.approx(sd) and view.header["learned"] == n
    assert long_row.n == n and long_row.raw_r == pytest.approx(0.9)
    assert long_row.expected_r == pytest.approx(mu)
    safe = mu - 0.5 * sd / math.sqrt(n)
    kelly = 0.25 * (safe / (sd * sd)) / 0.005
    assert long_row.kelly_x == pytest.approx(min(3.0, max(0.25, kelly)))
    assert bool(long_row.boosted) == (kelly > 1.0) and not long_row.blocked
    short_row = view.rows[(view.rows.section == "YÖN") & (view.rows.bucket == 10)].iloc[0]
    assert short_row.n == 0 and math.isnan(short_row.raw_r) and math.isnan(short_row.kelly_x)
    assert len(view.rows) == 2 + 4 + 3 + 2 + 4 + 2 + 2 + 4 + N


def test_engine_view_blocks_a_bucket_with_bad_expectancy():
    kp = KellyParams(on=True, prior=4.0, min_n=6.0, block=0.15)
    engine = TradeEngine(N, params(kelly=kp))
    feed_kelly(engine, [10, 12, 15, 19, 21, 31, 34, ease_base(N), 37], [-1.0] * 8)
    row = engine_view(make(engine, [100])).rows
    short = row[(row.section == "YÖN") & (row.bucket == 10)].iloc[0]
    assert short.blocked and short.kelly_x == pytest.approx(0.25)


def shadow_state(n=N):
    shape = (n, 5)
    return {
        "side": np.zeros(shape, int), "count": np.zeros(shape, int), "wins": np.zeros(shape, int),
        "net": np.zeros(shape), "entry": np.full(shape, np.nan), "notional": np.zeros(shape),
        "auto": np.full(n, -1),
    }  # fmt: skip


def test_method_symbol_view_adds_open_virtual_position_and_finds_best_method():
    state = shadow_state()
    state["net"][0, :] = [5, -2, 0, 0, 3]
    state["count"][0, :] = [2, 1, 0, 0, 2]
    state["wins"][0, :] = [1, 0, 0, 0, 1]
    state["side"][0, 1], state["entry"][0, 1], state["notional"][0, 1] = 1, 100.0, 100.0
    engine = TradeEngine(N, params())
    ctx = make(engine, [110], shadow=state, shadow_params=ShadowParams(fee_in=0, fee_out=0))
    view = method_symbol_view(ctx)
    cells = view.cells[view.cells.symbol == 0].set_index("account")
    assert cells.net_usd[1] == pytest.approx(-2 + 10.0) and cells.open_side[1] == 1
    assert cells.net_usd[0] == 5 and cells.win_pct[0] == 50.0 and math.isnan(cells.win_pct[2])
    assert list(cells.is_best) == [False, True, False, False, False]
    best = view.best.set_index("symbol").loc[0]
    assert best.best_method == 1 and best.best_usd == pytest.approx(8.0)
    assert view.best.set_index("symbol").loc[3].best_method == -1
    disabled = method_symbol_view(
        make(engine, [110], shadow=state, shadow_params=ShadowParams(fee_in=0, fee_out=0),
             enabled=(True, False, True, True))  # fmt: skip
    )
    assert disabled.best.set_index("symbol").loc[0].best_method == 0
    totals = view.totals.set_index("account")
    assert totals.net_usd[1] == pytest.approx(8.0) and totals.trades[0] == 2
    assert bool(totals.is_best[1]) and not totals.is_best[4]


def test_method_symbol_view_real_engine_column_includes_open_trade():
    engine = played()
    state = shadow_state()
    view = method_symbol_view(make(engine, [104, 98], shadow=state))
    real = view.engine.set_index("symbol")
    assert real.net_usd[0] == pytest.approx(engine.trades[0].usd) and not real.open[0]
    open_usd = engine.trades[1].notional * 0.02
    assert real.net_usd[1] == pytest.approx(open_usd) and bool(real.open[1])
    assert view.real_total == pytest.approx(engine.trades[0].usd + open_usd)


def test_method_symbol_view_auto_mode_reports_chosen_or_fallback_method():
    state = shadow_state()
    state["auto"][0], state["auto"][1] = 2, -1
    engine = TradeEngine(N, params())
    view = method_symbol_view(make(engine, [100], shadow=state, auto_mode=True, fallback_method=3))
    best = view.best.set_index("symbol")
    assert (best.auto_method[0], bool(best.auto_is_fallback[0])) == (2, False)
    assert (best.auto_method[1], bool(best.auto_is_fallback[1])) == (3, True)


def test_method_symbol_view_requires_shadow_state():
    with pytest.raises(ValueError):
        method_symbol_view(make(TradeEngine(N, params()), [100]))


def test_views_on_a_full_run_agree_with_each_other():
    trade = params(max_slots=2, use_give=True)
    panel = panel_params(n_top=3, max_slots=2)
    result = run_system(_universe(), panel, trade, True, collect_stats=True)
    ctx = view_context(result, panel, trade, names=[f"X{i}" for i in range(6)])
    trades, cash = trades_view(ctx), cash_view(ctx)
    reasons, engine = reason_view(ctx), engine_view(ctx)
    methods = method_symbol_view(ctx)
    assert trades.totals["total_usd"] == pytest.approx(cash["net_usd"])
    assert cash["trades"] == trades.totals["closed_count"] == reasons.summary["closed"]
    assert reasons.summary["net_usd"] == pytest.approx(trades.totals["closed_usd"])
    assert methods.real_total == pytest.approx(cash["net_usd"])
    assert methods.engine.trades.sum() == cash["trades"]
    frame = reasons.rows
    assert frame[frame.section == 1].trades.sum() == cash["trades"] > 0
    assert len(trades.rows) == min(8, len(result.trades))
    assert not engine.header["on"] and len(methods.cells) == 6 * 5
    last = result.stats[-1]
    assert last[0] == cash["realised"]


GOLDEN = Path(__file__).parent / "data" / "golden"


@pytest.mark.skipif(not (GOLDEN / "window.npz").exists(), reason="altın referans yok")
@pytest.mark.parametrize("stage", ["S9", "S49", "S50"])
def test_views_match_the_tv_validated_statistics_vector(stage):
    window = load_window(GOLDEN / "window.npz")
    result = run_stage(window, stage, collect_stats=True)
    panel, trade = stage_params(stage)
    ctx = view_context(result, panel, trade)
    stats = _named(result, trade.log_size)
    cash, reasons = cash_view(ctx), reason_view(ctx)
    assert cash["trades"] == stats["tCnt"].sum() > 0 and cash["wins"] == stats["tWin"].sum()
    assert cash["realised"] == stats["ks"][0] and cash["best_usd"] == stats["ks"][6]
    assert cash["max_loss_streak"] == stats["ks"][9]
    frame = reasons.rows
    for name, column in (
        ("lsW", "wins"),
        ("lsL", "losses"),
        ("lsWu", "win_usd"),
        ("lsLu", "loss_usd"),
    ):
        assert np.allclose(frame[column].to_numpy(float), stats[name]), name
    rows = trades_view(ctx).rows
    ring = stats["lgId"]
    assert set(rows.trade_id) == {int(v) for v in ring if v > 0}
    emtia = frame[frame.section == 12]
    assert np.array_equal(emtia.trades.to_numpy(), stats["symN"].astype(int))
