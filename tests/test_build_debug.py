import numpy as np
import pandas as pd

from alamet import build_debug
from alamet.build_debug import (
    DISPLAY_MARKER,
    _cases,
    _effective,
    _ranges,
    _statement_end,
    build_master,
    literal,
)
from alamet.stages import STAGES, Stage
from alamet.universe import SYMBOLS
from alamet.verify_system import stage_blocks

SOURCE = "\n".join(
    [
        "//@version=6",
        'indicator("Devis\'So Otomatik Bot Eğitim PaneliV2", calc_bars_count = 4000)',
        *[f't{k} = input.symbol("MEXC:X{k}USDT.P", "Symbol {k}")' for k in range(1, 21)],
        "calc_bars_count = 4000",
        'sigMode = input.string("Otomatik", "Mod", options = ["A", "B"])',
        '     tooltip = "çok satırlı"',
        'bool allowL = sigMode != "B"',
        'maxExp = input.float(3.0, "Maks", minval = 0.5)',
        'ownN = input.int(8, "n")',
        "plot(1.0)",
        "// =====",
        f"//  {DISPLAY_MARKER} (etiketler)",
        "f_live() =>",
        "    1",
    ]
)


def test_statement_end_skips_indented_continuation_lines():
    lines = ["a = input.int(1,", "     tooltip = 'x')", "b = 2"]
    assert _statement_end(lines, 0) == 2


def test_ranges_compress_consecutive_stage_indices():
    assert _ranges([0, 1, 2, 5, 7, 8]) == (
        "stgIdx >= 0 and stgIdx <= 2 or stgIdx == 5 or stgIdx >= 7 and stgIdx <= 8"
    )
    assert _ranges([4]) == "stgIdx == 4"


def test_effective_uses_the_most_common_value_as_the_fallback_arm():
    per_stage = ["maxExp_in", "0.5", "0.5", "0.5", "1.0"]
    assert _effective("maxExp", per_stage) == (
        "maxExp = (stgIdx == 0) ? maxExp_in : ((stgIdx == 4) ? 1.0 : (0.5))"
    )


def test_every_real_input_resolves_to_its_stage_value():
    for key in sorted({key for stage in STAGES.values() for key in stage.pine}):
        per_stage = [
            literal(stage.pine[key]) if key in stage.pine else f"{key}_in"
            for stage in STAGES.values()
        ]
        cases, fallback = _cases(per_stage)
        for index, expected in enumerate(per_stage):
            chosen = fallback
            for cond, text in cases:
                if eval(cond, {"stgIdx": index}):  # pylint: disable=eval-used
                    chosen = text
                    break
            assert chosen == expected, (key, index)


def test_master_renames_overridden_inputs_and_adds_selector_and_stage_plot(monkeypatch):
    stages = {
        "S1": Stage({"maxExp": 0.5, "sigMode": "B"}, 2, None),
        "S2": Stage({"maxExp": 0.5}, 2, None),
    }
    monkeypatch.setattr(build_debug, "STAGES", stages)
    master = build_master(SOURCE, 4000)
    assert "string stg = input.string(" in master and 'int stgIdx = stg == "S1" ? 0 : (1)' in master
    assert master.index("string stg") < master.index("t1 = input.symbol")
    assert "sigMode_in = input.string(" in master
    assert master.index("sigMode_in") < master.index("sigMode = ") < master.index("bool allowL")
    assert master.index('     tooltip = "çok satırlı"') < master.index("sigMode = ")
    assert "maxExp_in = input.float(3.0" in master
    assert "maxExp = 0.5" in master
    assert 'sigMode = (stgIdx == 1) ? sigMode_in : ("B")' in master
    assert 'options = ["S1", "S2"]' in master
    assert 'ownN = input.int(8, "n")' in master
    assert "f_live" not in master and DISPLAY_MARKER not in master
    assert 'plot(1.0 * stgIdx, "STG")' in master and 'plot(ks.get(1), "KS1")' not in master
    assert f'"{SYMBOLS[0]}USDT' not in master and "BINANCE:BTCUSDT.P" in master


def test_stage_blocks_reads_stage_names_from_the_stg_column(tmp_path):
    order = list(STAGES)
    frame = pd.DataFrame(
        {
            "time": np.arange(5),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "STG": [np.nan, order.index("S9")] + [order.index("S9")] * 3,
            "STG.1": [np.nan, np.nan] + [order.index("S32")] * 3,
        }
    )
    path = tmp_path / "all.csv"
    frame.to_csv(path, index=False)
    assert stage_blocks(str(path)) == [(0, "S9"), (1, "S32")]
