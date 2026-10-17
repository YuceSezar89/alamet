import pytest

from alamet.stages import STAGES, stage_params
from alamet.symbol import SymbolParams


@pytest.mark.parametrize("name", list(STAGES))
def test_every_stage_builds_consistent_params(name):
    panel, trade = stage_params(name)
    assert panel.n_top >= 1
    assert panel.divergence.max_slots == trade.max_slots
    assert trade.start_cap > 0


def test_coverage_closing_stages_set_the_intended_inputs():
    panel, trade = stage_params("S19")
    assert not trade.allow_long and not trade.re_entry and not panel.divergence.allow_long
    assert stage_params("S18")[1].max_exp == 0.5 and stage_params("S18")[1].swap_edge == 0.0
    assert stage_params("S20")[1].quality.min_power == 30.0
    assert stage_params("S21")[0].symbol == SymbolParams(tf_on=False)
    assert stage_params("S21")[0].selector.mode == 1
    assert stage_params("S25")[0].selector.mode == 5
    assert stage_params("S24")[0].divergence.dbl_pos_only


def test_state_mode_stages_set_the_intended_inputs():
    p26, t26 = stage_params("S26")
    assert p26.own == {"latch": False, "allow_no_hist": True} and not p26.divergence.need_dbl
    assert not t26.cap_risk_mode and t26.cap_pct == 5.0
    p27 = stage_params("S27")[0]
    assert p27.rank_mode == "Ham %" and not p27.dr_rank and p27.own == {"use_own_first": False}
    p28, t28 = stage_params("S28")
    assert not t28.kelly.replace and not t28.pool.prob and t28.pool.top1 and not t28.use_kill
    assert p28.selector.enabled == (True, True, True, False)
    t29 = stage_params("S29")[1]
    assert (t29.start_cap, t29.risk_pct, t29.max_lev, t29.fee_in, t29.slippage) == (
        5000.0, 1.0, 3.0, 0.05, 0.02,
    )  # fmt: skip
    assert t29.kelly.on
    assert not stage_params("S30")[0].selector.auto_block
    assert not stage_params("S31")[1].pool.top1 and stage_params("S31")[1].pool.prob


def test_gap_closing_stages_set_the_intended_inputs():
    assert (
        stage_params("S32")[0].symbol.td_len == 18 and stage_params("S32")[0].symbol.mo_thr == 0.8
    )
    assert stage_params("S33")[0].symbol.hv_flow == 0.2
    s34 = stage_params("S34")[0].symbol
    assert s34.ha_multipliers == (3, 4, 6, 8) and s34.tf_need == 2
    s35 = stage_params("S35")[0].symbol
    assert (s35.cool, s35.pend_max, s35.conf_bars, s35.atr_len) == (6, 8, 2, 14)
    s36 = stage_params("S36")[0]
    assert (s36.daily_open_hour, s36.dr_rsi_len, s36.dr_look) == (3, 5, 90)
    assert stage_params("S37")[0].selector.enabled == (False, True, True, True)
    assert stage_params("S38")[0].selector.seq_order == (3, 2, 1, 0)
    assert stage_params("S39")[0].selector.auto_min_n == 1
    p40, t40 = stage_params("S40")
    assert p40.n_top == 3 and p40.own["min_prev"] == 3 and p40.divergence.min_prev == 3
    assert t40.quality.max_ext == 2.5 and t40.r_unit == 1.0
    assert stage_params("S41")[1].neg_mult == 0.8
    assert not stage_params("S42")[1].risk_mode
    assert stage_params("S43")[1].kelly.kelly == 0.5 and stage_params("S43")[1].pool.decay == 0.5


def test_shadow_params_follow_the_trade_params():
    panel, trade = stage_params("S44")
    shadow = panel.shadow
    assert (shadow.fee_in, shadow.fee_out, shadow.slippage) == (0.05, 0.05, 0.02)
    assert (shadow.r_unit, shadow.risk_pct, shadow.start_cap) == (1.0, 1.0, 5000.0)
    assert (shadow.decay, shadow.decay_bars) == (trade.dyn_decay, trade.dyn_decay_bars)
    overridden, params = stage_params("S7", fee_in=0.07, slippage=0.03)
    assert params.fee_in == 0.07 and overridden.shadow.fee_in == 0.07
    assert overridden.shadow.slippage == 0.03


def test_s47_disables_the_middle_methods():
    selector = stage_params("S47")[0].selector
    assert selector.enabled == (True, False, False, True) and selector.mode == 4


def test_s48_and_s49_select_the_uncovered_rule_branches():
    assert stage_params("S48")[0].selector.mode == 6
    assert stage_params("S48")[0].selector.enabled == (False, True, True, True)
    assert stage_params("S48")[0].selector.fallback_method == 1
    assert not stage_params("S49")[0].divergence.need_dbl and stage_params("S49")[1].kelly.on


def test_s50_changes_only_the_bookkeeping_thresholds():
    trade = stage_params("S50")[1]
    assert (trade.hold_short, trade.hold_long, trade.log_size) == (2, 6, 5)
    assert stage_params("S50")[0] == stage_params("S9")[0]
