"""Ayar geçersiz kılma: yol çözümleme, tür çevirme, varsayılanın korunması."""

import pytest

from alamet.overrides import params_with
from alamet.stages import stage_params


def test_no_overrides_equals_the_stage_defaults():
    panel, trade = params_with("S17", [])
    ref_panel, ref_trade = stage_params("S17")
    assert panel == ref_panel and trade == ref_trade


def test_nested_and_dictionary_paths_are_applied_with_types():
    panel, trade = params_with(
        "S17",
        ["trade.use_tp=true", "trade.tp_r=3", "trade.quality.max_ext=6", "panel.n_top=3",
         "panel.divergence.z_min=0.5", "panel.own.mult=0.7", "trade.kelly.decay=0.9"],
        900,
    )  # fmt: skip
    assert trade.use_tp is True and trade.tp_r == 3.0 and trade.quality.max_ext == 6.0
    assert panel.n_top == 3 and panel.divergence.z_min == 0.5
    assert panel.own == {"mult": 0.7} and trade.kelly.decay == 0.9
    assert panel.symbol.signal_tf_seconds == 900


def test_simple_trade_fields_rederive_the_shadow_book():
    panel, _ = params_with("S17", ["trade.risk_pct=1.0"])
    ref_panel, _ = stage_params("S17")
    assert panel.shadow != ref_panel.shadow


def test_unknown_names_are_rejected():
    with pytest.raises(KeyError):
        params_with("S17", ["trade.quality.yok=1"])
    with pytest.raises(ValueError):
        params_with("S17", ["engine.x=1"])


def test_tuple_fields_are_set_by_index():
    panel, _ = params_with(
        "S17", ["panel.symbol.ha_multipliers.2=5", "panel.symbol.ha_weights.3=5"]
    )
    assert panel.symbol.ha_multipliers == (2, 3, 5, 6)
    assert panel.symbol.ha_weights == (2.0, 3.0, 4.0, 5.0)


def test_ease_ranking_bonus_needs_the_devisso_multiplier_on():
    panel, _ = params_with("S17", [])
    assert panel.dr_rank is True
    panel, _ = params_with("S17", ["trade.dr_on=false"])
    assert panel.dr_rank is False
    panel, _ = stage_params("S17", dr_on=False)
    assert panel.dr_rank is False
