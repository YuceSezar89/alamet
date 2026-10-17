"""
Altın referans (golden master): TradingView ile birebir doğrulanmış Python çıktısını dondurur.

Her basamak (S0-S49) için sabit bir pencerede (20 sembol, son 4000 bar, `window.npz`) işlem listesi
ve özet metrikler `trades.json`'a, birkaç basamağın panel kararları `panel.npz`'ye yazılır. Test
(`tests/test_golden.py`) aynı pencereyi yeniden koşturup karşılaştırır; bu sayede yeni
sinyal/mekanizma eklerken ya da kod değiştirirken doğrulanmış davranışın bozulmadığı kanıtlanır.

Güncelleme (yalnızca bilinçli davranış değişikliğinde):
  python -m alamet.golden update --root data/binance_15m --out tests/data/golden
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from alamet.binance_data import DEFAULT_ROOT, load_universe_binance
from alamet.daily import daily_from_bars, daily_pack, day_inputs, map_to_bars, window_daily
from alamet.stages import STAGES, stage_params
from alamet.system import run_system
from alamet.universe import SYMBOLS, Universe

START_TIME = 1788584400
END_TIME = 1792161000
PANEL_STAGES = ("S9", "S17", "S36")
DEFAULT_OUT = Path("tests/data/golden")


def save_window(path: Path, universe: Universe) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time=universe.time,
        open=universe.open,
        high=universe.high,
        low=universe.low,
        close=universe.close,
        volume=universe.volume,
    )


def load_window(path: Path) -> Universe:
    data = np.load(path)
    return Universe(
        SYMBOLS,
        data["time"],
        data["open"],
        data["high"],
        data["low"],
        data["close"],
        data["volume"],
    )


def cache_window(root: Path) -> Universe:
    """TV ile doğrulanmış sabit pencere (zaman damgasıyla; önbellek büyüse de değişmez)."""
    universe = load_universe_binance(root, SYMBOLS)
    first = int(np.searchsorted(universe.time, START_TIME))
    last = int(np.searchsorted(universe.time, END_TIME, side="right"))
    if universe.time[first] != START_TIME or universe.time[last - 1] != END_TIME:
        raise ValueError("Önbellek altın pencereyi kapsamıyor")
    return universe.slice_from(first).slice_to(last - first)


def run_stage(universe: Universe, stage: str, **options):
    panel_p, trade_p = stage_params(stage)
    tf = panel_p.symbol.signal_tf_seconds
    daily = window_daily(daily_from_bars(universe, tf), universe.time[0], tf_seconds=tf)
    pack = daily_pack(daily, panel_p.dr_rsi_len, panel_p.dr_look)
    day = day_inputs(map_to_bars(pack, daily.time, universe.time, tf_seconds=tf))
    return run_system(universe, panel_p, trade_p, True, start=0, day=day, **options)


def _clean(value):
    return None if isinstance(value, float) and np.isnan(value) else value


def summary(result) -> dict:
    trades = [
        [
            t.symbol, t.side, t.open_bar, t.close_bar, t.rank, t.reason, t.entry, t.exit,
            t.notional, t.usd, t.r,
        ]
        for t in result.trades
    ]  # fmt: skip
    return {
        "count": len(result.trades),
        "final": float(result.realised[-1]),
        "peak": float(result.peak[-1]),
        "max_dd": float(result.max_dd[-1]),
        "closed": int(result.closed_count[-1]),
        "trades": [[_clean(v) for v in row] for row in trades],
    }


def panel_arrays(result) -> dict[str, np.ndarray]:
    panel = result.panel
    return {
        "types": np.nan_to_num(panel.types).astype(np.int8),
        "t_rank": panel.divergence["t_rank"].astype(np.int8),
        "scores": np.nan_to_num(panel.scores).astype(np.float32),
    }


def update(root: Path, out: Path, stages: list[str] | None = None) -> None:
    """Verilen basamakları (varsayılan: hepsi) yeniden üretir; mevcut diğerleri korunur."""
    universe = cache_window(root)
    save_window(out / "window.npz", universe)
    trades_path, panel_path = out / "trades.json", out / "panel.npz"
    golden = (
        json.loads(trades_path.read_text(encoding="utf-8"))
        if stages and trades_path.exists()
        else {}
    )
    panels = dict(np.load(panel_path)) if stages and panel_path.exists() else {}
    for stage in stages or list(STAGES):
        result = run_stage(universe, stage)
        golden[stage] = summary(result)
        if stage in PANEL_STAGES:
            for key, array in panel_arrays(result).items():
                panels[f"{stage}_{key}"] = array
        print(
            f"{stage}: {golden[stage]['count']} işlem, kasa {golden[stage]['final']:.4f}",
            flush=True,
        )
    ordered = {name: golden[name] for name in STAGES if name in golden}
    trades_path.write_text(json.dumps(ordered, separators=(",", ":")), encoding="utf-8")
    np.savez_compressed(panel_path, **panels)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DEPV2 altın referans üretici")
    parser.add_argument("command", choices=["update"])
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--stages", nargs="+")
    args = parser.parse_args(argv)
    update(Path(args.root), Path(args.out), args.stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
