from dataclasses import replace

import pytest

from alamet.divergence import DivergenceParams
from alamet.engine import TradeParams
from alamet.own_divergence import MODE_SIGMA
from alamet.panel import PanelParams
from alamet.stages import STAGES, stage_params
from alamet.validate import validate_params


@pytest.mark.parametrize("name", list(STAGES))
def test_every_stage_is_valid(name):
    validate_params(*stage_params(name))


def test_min_prev_must_fit_the_divergence_history():
    bad = PanelParams(divergence=DivergenceParams(max_div=4, min_prev=4))
    with pytest.raises(ValueError, match="maxDiv"):
        validate_params(bad, TradeParams())


def test_sigma_mode_min_prev_must_not_exceed_own_n():
    bad = PanelParams(own_mode=MODE_SIGMA, own={"own_n": 3, "min_prev": 4})
    with pytest.raises(ValueError, match="ownN"):
        validate_params(bad, TradeParams())
    validate_params(replace(bad, own={"own_n": 5, "min_prev": 4}), TradeParams())


def test_devisso_multipliers_must_be_ordered():
    with pytest.raises(ValueError, match="Devisso"):
        validate_params(PanelParams(), TradeParams(dr_min_m=2.0, dr_max_m=0.5))
