"""D2–D5 — işlem motoru: kural başına elle hesaplanmış senaryolar."""

import numpy as np
import pytest

from alamet.engine import (
    REASON_DAY,
    REASON_GIVE,
    REASON_HARD_DD,
    REASON_LOCK,
    REASON_RANK,
    REASON_SIGNAL,
    REASON_STOP,
    REASON_TIME,
    REASON_TP,
    BarInputs,
    TradeEngine,
    TradeParams,
)
from alamet.entry_quality import EntryQualityParams

NAN = np.nan
N = 6
LOOSE = EntryQualityParams(use_ext=False, cool_bars=0, use_daily_open=False)


def params(**kw):
    base = {"fee_in": 0.0, "fee_out": 0.0, "quality": LOOSE, "lock_on": False, "dd_hard_on": False}
    base.update(kw)
    return TradeParams(**base)


def pad(values, fill=0.0):
    values = list(values)
    return np.array(values + [fill] * (N - len(values)), dtype=float)


def mkbar(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    close, types=(1,), target=(1,), atr=1.0, high=None, low=None, open_=None,
        scores=(1.0,), idx=None, day=False):  # fmt: skip
    close = pad(close, 100.0)
    t_rank = pad(target).astype(int)
    order = (
        np.arange(N) if idx is None else np.array(list(idx) + [k for k in range(N) if k not in idx])
    )
    return BarInputs(
        open=close.copy() if open_ is None else pad(open_, 100.0),
        high=close.copy() if high is None else pad(high, 100.0),
        low=close.copy() if low is None else pad(low, 100.0),
        close=close,
        atr=np.full(N, atr),
        types=pad(types),
        starts=np.full(N, 99.0),
        power=np.zeros(N),
        scores=pad(scores),
        eligible=t_rank > 0,
        t_rank=t_rank,
        idx_long=order,
        idx_short=order,
        daily_open=np.full(N, NAN),
        weekly_open=np.full(N, NAN),
        monthly_open=np.full(N, NAN),
        day_changed=day,
    )


def test_risk_based_notional_and_one_r_distance():
    engine = TradeEngine(N, params())
    engine.step(0, mkbar([100, 100, 100]))
    trade = engine.trades[0]
    assert trade.risk == pytest.approx(1.5)
    assert trade.notional == pytest.approx(1000 * 0.005 / 0.015)


def test_hard_cap_limits_notional():
    engine = TradeEngine(N, params(risk_pct=5.0))
    engine.step(0, mkbar([100, 100, 100]))
    assert engine.trades[0].notional == pytest.approx(1000 * 0.01 / 0.015)
    uncapped = TradeEngine(N, params(risk_pct=5.0, cap_on=False, max_exp=10.0))
    uncapped.step(0, mkbar([100, 100, 100]))
    assert uncapped.trades[0].notional == pytest.approx(1000 * 0.05 / 0.015)


def test_max_leverage_caps_notional_when_cap_is_off():
    engine = TradeEngine(N, params(risk_pct=50.0, cap_on=False, max_lev=2.0))
    engine.step(0, mkbar([100, 100, 100]))
    assert engine.trades[0].notional == pytest.approx(2000.0)


def test_second_target_uses_rank2_multiplier():
    engine = TradeEngine(N, params())
    engine.step(0, mkbar([100, 100, 100], types=(1, 1, 0), target=(1, 2, 0), scores=(2, 1, 0)))
    first, second = engine.trades[0], engine.trades[1]
    assert second.notional == pytest.approx(first.notional * 0.5, rel=1e-3)


def test_rank2_mult_zero_blocks_second_target():
    engine = TradeEngine(N, params(rank2_mult=0.0))
    engine.step(0, mkbar([100, 100, 100], types=(1, 1, 0), target=(1, 2, 0)))
    assert len(engine.trades) == 1


def test_signal_exit_closes_at_close_and_books_pnl_with_fees():
    engine = TradeEngine(N, params(fee_in=0.02, fee_out=0.02))
    engine.step(0, mkbar([100, 100, 100]))
    notional = engine.trades[0].notional
    engine.step(1, mkbar([104, 100, 100], types=(-1, 0, 0), target=(0, 0, 0)))
    trade = engine.trades[0]
    assert trade.reason == REASON_SIGNAL and trade.exit == 104
    expected = notional * 0.04 - notional * 0.0002 - notional * 1.04 * 0.0002
    assert trade.usd == pytest.approx(expected)
    assert engine.s.ks[0] == pytest.approx(1000 + expected)


def test_give_back_exit_after_arm():
    engine = TradeEngine(N, params())
    engine.step(0, mkbar([100, 100, 100]))
    engine.step(1, mkbar([102, 100, 100], high=[102.5, 100, 100]))
    assert engine.s.mfe[0] == pytest.approx(2.5 / 1.5)
    engine.step(2, mkbar([101.2, 100, 100], high=[101.5, 100, 100], target=(1, 0, 0)))
    trade = engine.trades[0]
    assert trade.reason == REASON_GIVE and trade.exit == 101.2


def test_give_back_not_armed_below_threshold():
    engine = TradeEngine(N, params())
    engine.step(0, mkbar([100, 100, 100]))
    engine.step(1, mkbar([101, 100, 100], high=[101.5, 100, 100]))
    engine.step(2, mkbar([100.2, 100, 100]))
    assert engine.trades[0].close_bar == -1


def test_stop_triggers_at_stop_price_and_gap_uses_open():
    engine = TradeEngine(N, params(use_stop=True, stop_r=2.0, use_give=False))
    engine.step(0, mkbar([100, 100, 100]))
    engine.step(1, mkbar([97.5, 100, 100], low=[96.5, 100, 100]))
    assert engine.trades[0].reason == REASON_STOP and engine.trades[0].exit == pytest.approx(97.0)
    gap = TradeEngine(N, params(use_stop=True, stop_r=2.0, use_give=False))
    gap.step(0, mkbar([100, 100, 100]))
    gap.step(1, mkbar([95, 100, 100], open_=[95.0, 100, 100], low=[94, 100, 100]))
    assert gap.trades[0].exit == 95.0


def test_time_stop_closes_after_max_hold():
    engine = TradeEngine(N, params(use_time_stop=True, max_hold=3, use_give=False))
    engine.step(0, mkbar([100, 100, 100]))
    engine.step(2, mkbar([100, 100, 100]))
    assert engine.trades[0].close_bar == -1
    engine.step(3, mkbar([100, 100, 100], target=(0, 0, 0)))
    assert engine.trades[0].reason == REASON_TIME


def test_take_profit_sets_target_and_closes_when_high_reaches_it():
    off = TradeEngine(N, params(use_give=False))
    off.step(0, mkbar([100, 100, 100]))
    assert np.isnan(off.s.take[0])
    engine = TradeEngine(N, params(use_tp=True, tp_r=2.0, use_give=False))
    engine.step(0, mkbar([100, 100, 100]))
    assert engine.s.take[0] == pytest.approx(103.0)
    engine.step(1, mkbar([102, 100, 100], high=[102.9, 100, 100]))
    assert engine.trades[0].close_bar == -1
    engine.step(2, mkbar([102, 100, 100], high=[103.2, 100, 100]))
    assert engine.trades[0].reason == REASON_TP and engine.trades[0].exit == pytest.approx(103.0)


def test_stop_wins_over_take_profit_in_same_bar():
    engine = TradeEngine(
        N, params(use_tp=True, tp_r=1.0, use_stop=True, stop_r=1.0, use_give=False)
    )
    engine.step(0, mkbar([100, 100, 100]))
    engine.step(1, mkbar([100, 100, 100], high=[102, 100, 100], low=[98, 100, 100]))
    assert engine.trades[0].reason == REASON_STOP


def test_swap_replaces_weakest_non_target_when_slots_full():
    engine = TradeEngine(N, params(max_slots=1, use_give=False))
    engine.step(0, mkbar([100, 100, 100], scores=(1.0, 0.0, 0.0)))
    assert engine.s.active[0]
    engine.step(1, mkbar([100, 100, 100], types=(1, 1, 0), target=(0, 1, 0),
                       scores=(1.0, 1.5, 0.0), idx=(1, 0, 2)))  # fmt: skip
    assert engine.trades[0].reason == REASON_RANK
    assert engine.s.active[1] and not engine.s.active[0]


def test_swap_needs_enough_score_edge_and_keeps_old_trade_otherwise():
    engine = TradeEngine(N, params(max_slots=1, use_give=False))
    engine.step(0, mkbar([100, 100, 100], scores=(1.0, 0.0, 0.0)))
    engine.step(1, mkbar([100, 100, 100], types=(1, 1, 0), target=(0, 1, 0),
                       scores=(1.0, 1.2, 0.0), idx=(1, 0, 2)))  # fmt: skip
    assert engine.s.active[0] and not engine.s.active[1]


def test_swap_never_closes_trade_that_is_still_a_target():
    engine = TradeEngine(N, params(max_slots=1, use_give=False))
    engine.step(0, mkbar([100, 100, 100]))
    engine.step(1, mkbar([100, 100, 100], types=(1, 1, 0), target=(2, 1, 0),
                       scores=(0.0, 5.0, 0.0), idx=(1, 0, 2)))  # fmt: skip
    assert engine.s.active[0] and not engine.s.active[1]


def test_heat_limit_shrinks_and_blocks_new_notional():
    engine = TradeEngine(N, params(max_exp=0.5, risk_pct=2.0, cap_on=False))
    engine.step(0, mkbar([100, 100, 100], types=(1, 1, 0), target=(1, 2, 0), scores=(2, 1, 0)))
    assert engine.trades[0].notional == pytest.approx(500.0)
    assert len(engine.trades) == 1


def test_cooldown_after_losing_close_blocks_reentry():
    quality = EntryQualityParams(use_ext=False, cool_bars=3, use_daily_open=False)
    engine = TradeEngine(N, params(quality=quality, use_give=False))
    engine.step(0, mkbar([100, 100, 100]))
    engine.step(1, mkbar([98, 100, 100], types=(-1, 0, 0), target=(0, 0, 0)))
    assert engine.s.last_loss[0] == 1
    engine.step(2, mkbar([98, 100, 100], target=(1, 0, 0)))
    assert len(engine.trades) == 1
    engine.step(4, mkbar([98, 100, 100], target=(1, 0, 0)))
    assert len(engine.trades) == 2


def test_stay_used_blocks_reentry_until_symbol_leaves_target():
    engine = TradeEngine(N, params(re_entry=False, use_give=False))
    engine.step(0, mkbar([100, 100, 100]))
    engine.s.active[0] = False
    engine.step(1, mkbar([100, 100, 100]))
    assert len(engine.trades) == 1
    for i in range(2, 5):
        engine.step(i, mkbar([100, 100, 100], target=(0, 0, 0)))
    engine.step(5, mkbar([100, 100, 100]))
    assert len(engine.trades) == 2


def test_negative_equity_scales_risk_when_lock_is_on():
    engine = TradeEngine(N, params(lock_on=True, day_loss_on=False))
    engine.s.ks[0] = 900.0
    engine.step(0, mkbar([100, 100, 100]))
    assert engine.trades[0].notional == pytest.approx(900 * 0.005 * 0.6 / 0.015)


def test_hard_drawdown_closes_everything_and_halts_entries():
    engine = TradeEngine(N, params(dd_hard_on=True, dd_hard_pct=1.0, dd_hard_bars=5,
                                   risk_pct=2.0, cap_on=False, use_give=False,
                                   day_loss_on=False))  # fmt: skip
    engine.step(0, mkbar([100, 100, 100]))
    engine.step(1, mkbar([95, 100, 100], types=(1, 0, 0)))
    assert engine.trades[0].reason == REASON_HARD_DD
    engine.step(2, mkbar([95, 100, 100]))
    assert len(engine.trades) == 1
    engine.step(7, mkbar([95, 100, 100]))
    assert len(engine.trades) == 2


def test_daily_loss_limit_blocks_new_entries_until_next_day():
    engine = TradeEngine(N, params(day_loss_on=True, day_loss_pct=0.5, risk_pct=2.0,
                                   cap_on=False, use_give=False))  # fmt: skip
    engine.step(0, mkbar([100, 100, 100], day=True))
    engine.step(1, mkbar([95, 100, 100], types=(-1, 0, 0), target=(0, 0, 0), day=False))
    assert engine.trades[0].reason == REASON_SIGNAL
    engine.step(2, mkbar([95, 100, 100], types=(1, 0, 0), target=(1, 0, 0)))
    assert len(engine.trades) == 1
    engine.step(3, mkbar([95, 100, 100], day=True))
    assert len(engine.trades) == 2
    assert REASON_DAY == 7


def test_cash_lock_closes_all_after_profit_pullback():
    engine = TradeEngine(N, params(lock_on=True, lock_arm=1.0, lock_frac=0.4, lock_bars=4,
                                   risk_pct=2.0, cap_on=False, use_give=False,
                                   day_loss_on=False))  # fmt: skip
    engine.step(0, mkbar([100, 100, 100]))
    engine.step(1, mkbar([102, 100, 100]))
    assert engine.s.kl[1] > 1000
    engine.step(2, mkbar([100.5, 100, 100]))
    assert engine.trades[0].reason == REASON_LOCK
    engine.step(3, mkbar([100.5, 100, 100]))
    assert len(engine.trades) == 1


def test_unconfirmed_bar_changes_nothing():
    engine = TradeEngine(N, params())
    engine.step(0, mkbar([100, 100, 100]), confirmed=False)
    assert not engine.trades
    assert engine.s.ks[0] == 1000


def test_peak_and_max_drawdown_tracking():
    engine = TradeEngine(N, params(use_give=False, lock_on=False))
    engine.step(0, mkbar([100, 100, 100]))
    engine.step(1, mkbar([102, 100, 100]))
    peak = engine.s.ks[1]
    engine.step(2, mkbar([99, 100, 100]))
    assert engine.s.ks[1] == pytest.approx(peak)
    assert engine.s.ks[2] > 0


def test_short_trade_profits_when_price_falls():
    engine = TradeEngine(N, params(use_give=False))
    engine.step(0, mkbar([100, 100, 100], types=(-1, 0, 0)))
    engine.step(1, mkbar([96, 100, 100], types=(1, 0, 0), target=(0, 0, 0)))
    assert engine.trades[0].side == -1 and engine.trades[0].usd > 0


def test_devisso_multiplier_scales_notional_and_blocks_hardest_tier():
    def opened(percentile, **kw):
        engine = TradeEngine(N, params(dr_on=True, **kw))
        inputs = mkbar([100, 100, 100])
        inputs.dr_pct = np.full(N, percentile)
        engine.step(0, inputs)
        return engine.trades

    base = opened(50.0, dr_min_m=1.0, dr_max_m=1.0)[0].notional
    assert opened(0.0)[0].notional == pytest.approx(base * 2.0)
    assert opened(100.0)[0].notional == pytest.approx(base * 0.4)
    assert opened(50.0)[0].notional == pytest.approx(base * 1.2)
    assert opened(95.0, dr_block_on=True) == []
    assert len(opened(85.0, dr_block_on=True)) == 1
    unknown = opened(np.nan)[0].notional
    assert unknown == pytest.approx(base * 1.2)


def test_devisso_off_ignores_percentile_entirely():
    engine = TradeEngine(N, params())
    inputs = mkbar([100, 100, 100])
    inputs.dr_pct = np.full(N, 100.0)
    engine.step(0, inputs)
    assert engine.trades[0].notional == pytest.approx(1000 * 0.005 / 0.015)


def dyn_engine(**kw):
    return TradeEngine(N, params(use_dyn=True, **kw))


def dyn_bar(div_count=1):
    inputs = mkbar([100, 100, 100])
    inputs.div_count = np.full(N, div_count)
    return inputs


def test_dynamic_buckets_follow_side_rank_and_divergence_group():
    assert TradeEngine.dyn_buckets(1, 1, 2) == (0, 2, 4)
    assert TradeEngine.dyn_buckets(-1, 2, 4) == (1, 3, 5)
    assert TradeEngine.dyn_buckets(1, 1, 6) == (0, 2, 6)


def test_closed_trade_writes_its_r_into_three_buckets():
    engine = dyn_engine(use_give=False)
    engine.step(0, dyn_bar(div_count=2))
    engine.step(1, mkbar([98, 100, 100], types=(-1,), target=(0,)))
    r = engine.trades[0].r
    assert r < 0
    assert engine.s.bne[[0, 2, 4]].tolist() == [1.0, 1.0, 1.0]
    assert engine.s.bsr[[0, 2, 4]] == pytest.approx([r, r, r])
    assert engine.s.bne[[1, 3, 5, 6]].sum() == 0


def test_dynamic_multiplier_is_clamped_and_uses_confidence_weight():
    engine = dyn_engine()
    engine.s.bne[[0, 2, 4]] = 10.0
    engine.s.bsr[[0, 2, 4]] = 15.0
    mult, kill = engine.dyn_score(1, 1, 1)
    assert mult == pytest.approx(min(2.0, 1.0 + 1.5 * (15.0 / 15.0)))
    assert not kill
    engine.s.bsr[[0, 2, 4]] = -3.0
    assert engine.dyn_score(1, 1, 1)[0] == pytest.approx(max(0.25, 1.0 + 1.5 * (-3.0 / 15.0)))
    engine.s.bne[[0, 2, 4]] = 0.0
    engine.s.bsr[[0, 2, 4]] = 0.0
    assert engine.dyn_score(1, 1, 1) == (1.0, False)


def test_bad_rank_or_divergence_bucket_blocks_entry_but_direction_bucket_does_not():
    engine = dyn_engine()
    engine.s.bne[0], engine.s.bsr[0] = 20.0, -30.0
    assert engine.dyn_score(1, 1, 1)[1] is False
    engine.s.bne[2], engine.s.bsr[2] = 6.0, -3.0
    assert engine.dyn_score(1, 1, 1)[1] is True
    engine.step(1, dyn_bar())
    assert engine.trades == []
    quiet = dyn_engine()
    quiet.s.bne[4], quiet.s.bsr[4] = 5.0, -30.0
    assert quiet.dyn_score(1, 1, 1)[1] is False


def test_drawdown_throttle_and_loss_streak_scale_risk_independently():
    engine = dyn_engine(lock_on=False)
    assert engine.entry_multiplier(1000.0, 1, 1, 1, 0.0) == (1.0, False)
    assert engine.entry_multiplier(1000.0, 1, 1, 1, 5.0)[0] == pytest.approx(0.5)
    engine.s.ks[8] = 3.0
    assert engine.entry_multiplier(1000.0, 1, 1, 1, 5.0)[0] == pytest.approx(0.25)
    assert engine.entry_multiplier(1000.0, 1, 1, 1, 0.0)[0] == pytest.approx(0.5)


def test_dynamic_off_ignores_buckets_throttle_and_streak():
    engine = TradeEngine(N, params())
    engine.s.bne[2], engine.s.bsr[2] = 50.0, -90.0
    engine.s.ks[8] = 10.0
    assert engine.entry_multiplier(1000.0, 1, 1, 1, 50.0) == (1.0, False)


def test_dynamic_drawdown_halt_blocks_entries_then_releases():
    engine = dyn_engine(dd_halt=1.0, halt_bars=5, risk_pct=2.0, cap_on=False, use_give=False,
                        day_loss_on=False)  # fmt: skip
    engine.step(0, dyn_bar())
    engine.step(1, mkbar([95, 100, 100], types=(1,), target=(1,)))
    assert engine.s.dyn[1] == 6.0 and engine.s.dyn[2] == 1.0
    engine.s.active[0] = False
    engine.step(2, mkbar([95, 100, 100], types=(1,), target=(1,)))
    assert len(engine.trades) == 1
    engine.step(6, mkbar([95, 100, 100], types=(1,), target=(1,)))
    assert len(engine.trades) == 2
    assert engine.s.dyn[0] == pytest.approx(engine.equity(mkbar([95, 100, 100]).close), abs=1e-6)


def test_buckets_decay_on_schedule_only_when_dynamic_is_on():
    engine = dyn_engine(dyn_decay=0.5, dyn_decay_bars=4)
    engine.s.bne[:], engine.s.bsr[:] = 8.0, 8.0
    engine.step(1, mkbar([100, 100, 100], types=(0,), target=(0,)))
    assert engine.s.bne[0] == 8.0
    engine.step(4, mkbar([100, 100, 100], types=(0,), target=(0,)))
    assert engine.s.bne[0] == 4.0 and engine.s.bsr[0] == 4.0
    off = TradeEngine(N, params(dyn_decay=0.5, dyn_decay_bars=4))
    off.s.bne[:] = 8.0
    off.step(4, mkbar([100, 100, 100], types=(0,), target=(0,)))
    assert off.s.bne[0] == 8.0
