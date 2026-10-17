"""
Altın referans testleri: TradingView ile doğrulanmış pencerede (S47-S49 CSV'siyle aynı başlangıç)
her basamağın işlem listesi ve özet metrikleri dondurulmuş Python çıktısıyla aynı olmalıdır.

Varsayılan: temsilci basamaklar. Tümü için: ALAMET_GOLDEN=all pytest tests/test_golden.py
Bilinçli davranış değişikliğinde: python -m alamet.golden update --root data/binance_15m
"""

# pylint: disable=redefined-outer-name,import-outside-toplevel
import json
import os
from pathlib import Path

import numpy as np
import pytest

from alamet.golden import PANEL_STAGES, load_window, panel_arrays, run_stage, summary
from alamet.stages import STAGES

GOLDEN = Path(__file__).parent / "data" / "golden"
REPRESENTATIVE = ("S0", "S5", "S9", "S12", "S13", "S17", "S29", "S36", "S43", "S44", "S48")
TV_ANCHORS = {  # TradingView `KS0`/`KS2`/`CL` (s47 s48 s49.csv, pencerenin son barı)
    "S47": (1033.653459060912, 9.409128766110593, 120),
    "S48": (999.1291044261646, 3.675645531646904, 67),
    "S49": (1007.8787411502312, 10.894174236200277, 578),
}

pytestmark = pytest.mark.skipif(not (GOLDEN / "trades.json").exists(), reason="altın referans yok")


def _stages() -> list[str]:
    return list(STAGES) if os.environ.get("ALAMET_GOLDEN") == "all" else list(REPRESENTATIVE)


@pytest.fixture(scope="module")
def window():
    return load_window(GOLDEN / "window.npz")


@pytest.fixture(scope="module")
def golden():
    return json.loads((GOLDEN / "trades.json").read_text(encoding="utf-8"))


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return bool(np.isclose(a, b, rtol=1e-9, atol=1e-9))


@pytest.mark.parametrize("stage", _stages())
def test_stage_matches_the_frozen_reference(stage, window, golden):
    expected = golden[stage]
    actual = summary(run_stage(window, stage))
    for key in ("count", "closed"):
        assert actual[key] == expected[key], (stage, key)
    for key in ("final", "peak", "max_dd"):
        assert _same(actual[key], expected[key]), (stage, key, actual[key], expected[key])
    assert len(actual["trades"]) == len(expected["trades"])
    for index, (got, want) in enumerate(zip(actual["trades"], expected["trades"])):
        assert all(_same(g, w) for g, w in zip(got, want)), (stage, index, got, want)


@pytest.mark.parametrize("stage", list(PANEL_STAGES))
def test_panel_decisions_match_the_frozen_reference(stage, window):
    frozen = np.load(GOLDEN / "panel.npz")
    arrays = panel_arrays(run_stage(window, stage))
    assert np.array_equal(arrays["types"], frozen[f"{stage}_types"])
    assert np.array_equal(arrays["t_rank"], frozen[f"{stage}_t_rank"])
    assert np.allclose(arrays["scores"], frozen[f"{stage}_scores"], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("stage", list(TV_ANCHORS))
def test_frozen_reference_is_anchored_to_tradingview(stage, golden):
    final, max_dd, closed = TV_ANCHORS[stage]
    assert _same(golden[stage]["final"], final)
    assert _same(golden[stage]["max_dd"], max_dd)
    assert golden[stage]["closed"] == closed


def test_every_stage_has_a_frozen_reference(golden):
    assert set(golden) == set(STAGES)


def test_reference_detects_a_behavior_change(window, golden, monkeypatch):
    from dataclasses import replace

    from alamet import golden as golden_module

    original = golden_module.stage_params

    def tampered(name):
        panel, trade = original(name)
        return panel, replace(trade, risk_pct=trade.risk_pct * 1.05)

    monkeypatch.setattr(golden_module, "stage_params", tampered)
    actual = summary(run_stage(window, "S9"))
    assert actual["trades"] != golden["S9"]["trades"] or not _same(
        actual["final"], golden["S9"]["final"]
    )
