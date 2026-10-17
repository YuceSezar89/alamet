import numpy as np

from alamet.convergence import Agreement, agreement, table
from alamet.panel import PanelParams, build_panel
from alamet.universe import Universe


def _universe(n=4, bars=900, seed=5):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, (bars, n)), axis=0))
    spread = close * 0.003
    open_ = np.vstack([close[:1], close[:-1]])
    time = 1_767_571_200 + np.arange(bars) * 900
    return Universe(
        tuple(f"S{i}" for i in range(n)), time, open_, np.maximum(open_, close) + spread,
        np.minimum(open_, close) - spread, close, rng.uniform(100, 200, (bars, n)),
    )  # fmt: skip


def test_identical_panels_fully_agree_from_the_first_bar():
    panel = build_panel(_universe(), PanelParams(), True, 0)
    result = agreement(panel, panel, 300, 0)
    assert (result.types, result.t_rank, result.scores) == (1.0, 1.0, 1.0)
    assert result.converged_after == 0


def test_a_late_start_is_compared_on_the_common_window_and_reports_convergence():
    universe = _universe()
    reference = build_panel(universe, PanelParams(), True, 0)
    late = build_panel(universe, PanelParams(), True, 500)
    result = agreement(late, reference, 300, 400)
    assert 0.0 <= result.types <= 1.0 and 0.0 <= result.scores <= 1.0
    assert result.converged_after is None or 0 <= result.converged_after < 300


def test_table_formats_percentages_and_unconverged_rows():
    text = table(
        "S9", [Agreement(0, 0.5, 0.25, 0.75, None), Agreement(10, 1.0, 1.0, 1.0, 0)], 100, 50
    )
    assert "| 0 | 50.00 | 25.00 | 75.00 | yakınsamadı |" in text
    assert "| 10 | 100.00 | 100.00 | 100.00 | 0 |" in text
