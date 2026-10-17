"""Yapılandırılmış olaylar: elle hesaplanmış senaryolar, TV'de doğrulanmış sayılarla çapraz kontrol."""

# pylint: disable=redefined-outer-name
import math
from pathlib import Path

import numpy as np
import pytest

from alamet.engine import (
    REASON_DAY,
    REASON_HARD_DD,
    REASON_LOCK,
    REASON_SIGNAL,
    STATUS_SIGNAL,
    STATUS_STOP,
    TradeEngine,
)
from alamet.events import (
    PHASE_ENTRY,
    PHASE_EXIT,
    PHASE_PROTECT,
    CloseEvent,
    DayLossEvent,
    DivergenceEvent,
    DoubleDivergenceEvent,
    EquityLockEvent,
    EventLog,
    HardDrawdownEvent,
    OpenEvent,
    fmt_move,
    fmt_price,
    fmt_r,
    render,
    strength_text,
)
from alamet.golden import load_window, run_stage
from alamet.kelly import KellyParams
from alamet.stages import panel_params, stage_params
from alamet.stats import stats_layout
from alamet.system import run_system
from tests.test_engine import N, dyn_bar, mkbar, params
from tests.test_stats import _universe

GOLDEN = Path(__file__).parent / "data" / "golden"


def engine_with_log(**kw):
    engine = TradeEngine(N, params(**kw))
    engine.events = EventLog()
    return engine


def test_log_is_empty_and_inert_by_default():
    engine = TradeEngine(N, params())
    engine.step(0, mkbar([100, 100, 100]))
    assert engine.events is None and len(engine.trades) == 1


def test_open_event_carries_hand_computed_size_chain():
    engine = engine_with_log()
    engine.step(0, mkbar([100, 100, 100], types=(1, 1, 0), target=(1, 2, 0), scores=(2, 1, 0)))
    first, second = engine.events.of(OpenEvent)
    base = 1000 * 0.005 / 0.015
    assert (first.symbol, first.side, first.rank, first.entry) == (0, 1, 1, 100.0)
    assert math.isnan(first.take) and first.phase == PHASE_ENTRY and first.bar == 0
    assert first.notional == pytest.approx(base) and first.risk_pct == pytest.approx(0.5)
    assert first.multiplier == pytest.approx(1.0) and first.rank_mult == 1.0
    assert second.rank == 2 and second.rank_mult == 0.5
    assert second.notional == pytest.approx(base / 2) and second.risk_pct == pytest.approx(0.25)
    assert second.multiplier == pytest.approx(0.5)
    assert not (first.dyn_on or first.kelly_on or first.dr_on)
    assert first.trade_id == engine.trades[0].trade_id == 1 and second.trade_id == 2


def test_open_event_carries_take_profit_price():
    engine = engine_with_log(use_tp=True, tp_r=2.0)
    engine.step(0, mkbar([100, 100, 100]))
    assert engine.events.of(OpenEvent)[0].take == pytest.approx(100 + 2.0 * 1.5)


def test_open_event_devisso_parts():
    engine = engine_with_log(dr_on=True)
    inputs = mkbar([100, 100, 100])
    inputs.dr_pct = np.full(N, 0.0)
    inputs.dr_ratio = np.full(N, 0.123)
    engine.step(0, inputs)
    event = engine.events.of(OpenEvent)[0]
    assert (
        event.dr_on
        and event.dr_mult == pytest.approx(2.0)
        and event.multiplier == pytest.approx(2.0)
    )
    assert event.dr_ratio == 0.123 and event.dr_percentile == 0.0
    assert event.notional == pytest.approx(1000 * 0.005 / 0.015 * 2.0)


def test_open_event_dynamic_parts_follow_buckets_and_streak():
    engine = TradeEngine(N, params(use_dyn=True, lock_on=False))
    engine.events = EventLog()
    for bucket in (0, 2, 4):
        engine.s.bne[bucket], engine.s.bsr[bucket] = 5.0, 5.0
    engine.s.ks[8] = 3.0
    engine.step(1, dyn_bar(1))
    event = engine.events.of(OpenEvent)[0]
    assert event.dyn_on and event.dyn_score == pytest.approx(0.5)
    assert event.streak_factor == 0.5 and event.dd_factor == 1.0
    assert event.dyn_mult == pytest.approx(0.5 * 1.75) == pytest.approx(event.multiplier)


def test_open_event_kelly_exploration_parts():
    engine = engine_with_log(kelly=KellyParams(on=True))
    engine.step(0, mkbar([100, 100, 100]))
    event = engine.events.of(OpenEvent)[0]
    assert event.kelly_on and event.kelly_exploring
    assert event.kelly_effective_n == 0.0 and event.kelly_mean == 0.0
    assert event.kelly_mult == pytest.approx(0.5)


def test_close_event_matches_trade_and_goes_in_exit_phase():
    engine = engine_with_log(use_give=False)
    engine.step(0, mkbar([100, 100, 100]))
    engine.step(1, mkbar([96, 100, 100], types=(-1, 0, 0), target=(0, 0, 0)))
    (event,) = engine.events.of(CloseEvent)
    trade = engine.trades[0]
    assert (event.bar, event.phase, event.symbol, event.side) == (1, PHASE_EXIT, 0, 1)
    assert (event.status, event.reason, event.hold) == (STATUS_SIGNAL, REASON_SIGNAL, 1)
    assert event.entry == trade.entry and event.exit == 96.0
    assert event.r == trade.r and event.pnl_pct == trade.pnl and event.usd == trade.usd


def test_hard_drawdown_event_follows_the_closes_it_caused():
    engine = engine_with_log(dd_hard_on=True, dd_hard_pct=1.0, dd_hard_bars=5, risk_pct=2.0,
                             cap_on=False, use_give=False, day_loss_on=False)  # fmt: skip
    engine.step(0, mkbar([100, 100, 100]))
    engine.step(1, mkbar([95, 100, 100], types=(1, 0, 0)))
    close, hard = [e for e in engine.events.ordered() if e.bar == 1]
    assert isinstance(close, CloseEvent) and close.reason == REASON_HARD_DD
    assert close.status == STATUS_STOP and close.phase == PHASE_PROTECT
    assert isinstance(hard, HardDrawdownEvent) and (hard.closed, hard.wait_bars) == (1, 5)
    assert hard.drawdown_pct >= 1.0


def test_daily_loss_event_reports_whether_it_closes_trades():
    for closes in (False, True):
        engine = engine_with_log(day_loss_on=True, day_loss_pct=0.5, risk_pct=2.0, cap_on=False,
                                 use_give=False, day_loss_close=closes)  # fmt: skip
        engine.step(0, mkbar([100, 100, 100], day=True))
        engine.step(1, mkbar([95, 100, 100], day=False, types=(1, 0, 0), target=(0, 0, 0)))
        (event,) = engine.events.of(DayLossEvent)
        assert event.closes_trades is closes and event.closed == (1 if closes else 0)
        assert event.loss_pct >= 0.5
        reasons = [e.reason for e in engine.events.of(CloseEvent)]
        assert reasons == ([REASON_DAY] if closes else [])


def test_equity_lock_event_reports_peak_profit_before_reset():
    engine = engine_with_log(lock_on=True, lock_arm=1.0, lock_frac=0.4, lock_bars=4, risk_pct=2.0,
                             cap_on=False, use_give=False, day_loss_on=False)  # fmt: skip
    engine.step(0, mkbar([100, 100, 100]))
    engine.step(1, mkbar([102, 100, 100]))
    peak = engine.s.kl[1]
    engine.step(2, mkbar([100.5, 100, 100]))
    (event,) = engine.events.of(EquityLockEvent)
    assert event.peak_profit == pytest.approx(peak - 1000.0) and event.wait_bars == 4
    assert event.closed == 1 and engine.events.of(CloseEvent)[0].reason == REASON_LOCK


def test_ordering_follows_pine_phases_inside_one_bar():
    log = EventLog()
    log.add(DivergenceEvent, 3, 1, symbol=0, number=2, direction=1)
    log.add(HardDrawdownEvent, 3, PHASE_PROTECT, drawdown_pct=1.0, closed=0, wait_bars=1)
    log.add(CloseEvent, 3, PHASE_EXIT, symbol=0, side=1, trade_id=1, status=1, reason=0,
            entry=1.0, exit=1.0, r=0.0, pnl_pct=0.0, usd=0.0, hold=0)  # fmt: skip
    log.add(DivergenceEvent, 2, 1, symbol=1, number=2, direction=-1)
    kinds = [type(e).__name__ for e in log.ordered()]
    assert kinds == ["DivergenceEvent", "CloseEvent", "DivergenceEvent", "HardDrawdownEvent"]


def test_time_of_needs_times():
    log = EventLog()
    event = log.add(HardDrawdownEvent, 1, PHASE_PROTECT, drawdown_pct=1.0, closed=0, wait_bars=1)
    with pytest.raises(ValueError):
        log.time_of(event)
    log.times = np.array([10, 20, 30])
    assert log.time_of(event) == 20


def _named(result, log_size=8):
    sizes = stats_layout(log_size, len(result.panel.universe.symbols))
    row, out, offset = result.stats[-1], {}, 0
    for name, size in sizes:
        out[name] = row[offset : offset + size]
        offset += size
    return out


def _check_against_validated_stats(result, log_size=8):
    log, stats = result.events, _named(result, log_size)
    closes, opens = log.of(CloseEvent), log.of(OpenEvent)
    assert len(closes) == stats["tCnt"].sum() and len(opens) == len(result.trades)
    assert sum(e.usd for e in closes) == pytest.approx(stats["tSumUsd"].sum())
    for side, index in ((1, 0), (-1, 1)):
        assert sum(1 for e in closes if e.side == side) == stats["tCnt"][index]
        assert sum(e.usd for e in closes if e.side == side) == pytest.approx(
            stats["tSumUsd"][index]
        )
    for symbol in range(len(stats["symN"])):
        mine = [e for e in closes if e.symbol == symbol]
        assert len(mine) == stats["symN"][symbol]
        assert sum(e.usd for e in mine) == pytest.approx(stats["symNet"][symbol])
    for reason in range(9):
        count = sum(1 for e in closes if e.reason == reason)
        assert count == stats["lsW"][reason] + stats["lsL"][reason]
    size = len(stats["lgId"])
    by_id = {e.trade_id: e for e in opens}
    exits = {e.trade_id: e for e in closes}
    for trade_id in range(max(1, len(opens) - size + 1), len(opens) + 1):
        slot = (trade_id - 1) % size
        event = by_id[trade_id]
        assert stats["lgId"][slot] == trade_id and stats["lgSym"][slot] == event.symbol
        assert stats["lgDir"][slot] == event.side and stats["lgEnt"][slot] == event.entry
        assert stats["lgNot"][slot] == event.notional
        same_take = (np.isnan(stats["lgTp"][slot]) and math.isnan(event.take)) or (
            stats["lgTp"][slot] == event.take
        )
        assert same_take
        if trade_id in exits:
            assert stats["lgExit"][slot] == exits[trade_id].exit
            assert stats["lgSt"][slot] == exits[trade_id].status
        else:
            assert stats["lgSt"][slot] == 0 and np.isnan(stats["lgExit"][slot])


def test_events_agree_with_validated_statistics_on_synthetic_data():
    result = run_system(
        _universe(), panel_params(n_top=3, max_slots=2), params(max_slots=2, use_give=True), True,
        collect_stats=True, collect_events=True,
    )  # fmt: skip
    assert len(result.events.of(CloseEvent)) > 5
    _check_against_validated_stats(result)


def test_divergence_events_match_the_tracker_counts():
    universe = _universe(bars=1500, seed=5)
    result = run_system(
        universe, panel_params(n_top=3, max_slots=2), params(max_slots=2), True,
        collect_events=True,
    )  # fmt: skip
    count = result.panel.divergence["count"]
    starts = result.events.of(DivergenceEvent)
    doubles = result.events.of(DoubleDivergenceEvent)
    assert starts and all(e.number >= 2 and e.direction in (1, -1) for e in starts)
    assert all(count[e.bar, e.symbol] == e.number for e in starts + doubles)
    assert all(e.phase == 1 for e in starts + doubles)


requires_golden = pytest.mark.skipif(
    not (GOLDEN / "window.npz").exists(), reason="altın referans yok"
)


@pytest.fixture(scope="module")
def window():
    return load_window(GOLDEN / "window.npz")


@requires_golden
@pytest.mark.parametrize("stage", ["S9", "S13", "S43", "S49", "S50"])
def test_events_change_nothing_and_match_stats_on_the_tv_window(stage, window):
    plain = run_stage(window, stage, collect_stats=True)
    logged = run_stage(window, stage, collect_stats=True, collect_events=True)
    for name in ("sides", "entries", "notionals", "realised", "peak", "max_dd", "closed_count"):
        assert np.array_equal(getattr(plain, name), getattr(logged, name), equal_nan=True), name
    assert np.array_equal(plain.stats, logged.stats, equal_nan=True)
    assert plain.events is None
    _check_against_validated_stats(logged, stage_params(stage)[1].log_size)


def test_render_open_close_and_divergence_texts():
    names = ["BTCUSDT", "ETHUSDT"]
    opened = OpenEvent(
        bar=0, phase=PHASE_ENTRY, symbol=0, side=1, trade_id=1, method=0, strength=1, rank=1,
        div_n=2, entry=78000.5, take=math.nan, notional=333.3333, risk_pct=0.5, multiplier=1.0,
        dyn_mult=1.0, kelly_mult=1.0, dr_mult=1.0, rank_mult=1.0, dyn_on=False, dyn_score=math.nan,
        dd_factor=1.0, streak_factor=1.0, below_start_cap=False, kelly_on=False, kelly_mean=0.0,
        kelly_effective_n=0.0, kelly_exploring=False, dr_on=False, dr_ratio=math.nan,
        dr_percentile=math.nan,
    )  # fmt: skip
    assert render(opened, names, "Otomatik") == (
        "AÇ | BTCUSDT | LONG (Otomatik · TREND) ⚡ Strong | Hedef #1 | Giriş 78000.5 | TP - | "
        "Büyüklük 333.33 USDT | 1R risk %0.5 kasa | Dinamik x1 (normal)"
    )
    closed = CloseEvent(
        bar=1, phase=PHASE_EXIT, symbol=1, side=-1, trade_id=1, status=3, reason=2, entry=1.0,
        exit=1.0, r=-1.234, pnl_pct=0.0, usd=-4.5678, hold=3,
    )  # fmt: skip
    assert render(closed, names) == (
        "KAPAT | ETHUSDT | SHORT | TERS SİNYAL | R -1.23R | PNL +0% | K/Z -4.57 USDT"
    )
    assert render(DivergenceEvent(bar=1, phase=1, symbol=0, number=3, direction=-1), names) == (
        "BTCUSDT · 3. AYRIŞMA ▼S"
    )
    double = DoubleDivergenceEvent(bar=1, phase=1, symbol=1, number=4, move=1.236)
    assert render(double, names) == "ETHUSDT · 4. AYRIŞMA ★ ÇİFT (kendi içinde de) +1.24%"


def test_render_open_reason_parts_and_protection_texts():
    base = dict(
        bar=0, phase=PHASE_ENTRY, symbol=0, side=-1, trade_id=1, method=3, strength=4, rank=2,
        div_n=2, entry=2.5, take=2.4, notional=10.0, risk_pct=0.25, multiplier=0.875,
        dyn_mult=0.875, kelly_mult=1.234, dr_mult=0.4, rank_mult=0.5, dyn_on=True, dyn_score=0.5,
        dd_factor=0.5, streak_factor=1.0, below_start_cap=False, kelly_on=True, kelly_mean=0.1234,
        kelly_effective_n=7.26, kelly_exploring=False, dr_on=True, dr_ratio=0.12345,
        dr_percentile=math.nan,
    )  # fmt: skip
    text = render(OpenEvent(**base), ["X"], "M")
    assert "SHORT (M · HACİM) ⚡ Strong ②" in text and "Giriş 2.5 | TP 2.4" in text
    assert text.endswith(
        "(DD kısıtı skor +0.5R · MOTOR x1.23 E[R] +0.12R · n≈7.3 · DR x0.4 "
        "(ratio 0.123 · yüzdelik -))"
    )
    explore = render(OpenEvent(**{**base, "kelly_exploring": True, "dd_factor": 1.0}), ["X"])
    assert "MOTOR x1.23 keşif (n≈7.3)" in explore and "DD kısıtı" not in explore
    assert render(
        HardDrawdownEvent(bar=0, phase=2, drawdown_pct=15.04, closed=3, wait_bars=96), []
    ) == ("⛔ SERT DD ACİL ÇIKIŞ | zirveden -15% | 3 işlem kapatıldı | 96 SS barı bekleme")
    assert render(
        DayLossEvent(bar=0, phase=2, loss_pct=3.26, closed=2, closes_trades=True), []
    ) == ("⛔ GÜNLÜK KAYIP LİMİTİ | -3.3% | gün sonuna kadar yeni işlem yok | 2 işlem kapatıldı")
    assert render(
        DayLossEvent(bar=0, phase=2, loss_pct=3.0, closed=0, closes_trades=False), []
    ) == ("⛔ GÜNLÜK KAYIP LİMİTİ | -3% | gün sonuna kadar yeni işlem yok")
    assert render(
        EquityLockEvent(bar=0, phase=2, peak_profit=31.04, closed=1, wait_bars=48), []
    ) == ("🔒 KASA KÂR KİLİDİ | zirve kâr 31 USDT | 1 işlem kapatıldı | 48 SS barı bekleme")


def test_number_formats_follow_pine_helpers():
    assert fmt_price(78000.0) == "78000" and fmt_price(2.5) == "2.5"
    assert fmt_price(0.00012345678) == "0.00012346" and fmt_price(math.nan) == "-"
    assert fmt_r(1.5) == "+1.5R" and fmt_r(-0.004) == "0R" and fmt_r(math.nan) == "-"
    assert fmt_move(0.0) == "+0%" and fmt_move(-2.345) == "-2.35%" and fmt_move(math.nan) == "-"
    assert [strength_text(c) for c in (0, 1, 2, 3, 5)] == [
        "● Normal", "⚡ Strong", "🚀 Çok güçlü", "● Normal ②", "🚀 Çok güçlü ②",
    ]  # fmt: skip
