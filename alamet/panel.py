"""
Sinyal paneli: 20 sembol için A→C hattının bar bar çıktıları (işlem durumundan bağımsız kısım).

Sıra (DEPV2 ile aynı): sembol sinyalleri (A1–A8) → seçici (C5) + gölge kasa (C6) → temel skor (C1)
→ kendi-içi kapı (C3) → sıralama (C2) → ayrışma takibi (C4). Havuz/Kelly/dinamik kasa kapalı
olduğunda bu zincir hiçbir işlem sonucuna bağlı değildir; bu yüzden topluca hesaplanır.
Kapanmamış son bar (`confirm_last=False`) durum ilerletmez, yalnızca değerlendirilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from alamet.daily import DayInputs
from alamet.devisso import LOOK, RSI_LEN
from alamet.divergence import DivergenceParams, run_divergence
from alamet.gate import GateResult
from alamet.levels import daily_open, period_ids, period_levels
from alamet.own_divergence import MODE_PREV, OwnDivergence
from alamet.primitives import pine_atr
from alamet.ranking import N_TOP, RankResult, rank_all
from alamet.score import (
    DR_MIX,
    DR_RANK_W,
    RANK_ATR,
    TGT_CAP,
    TGT_MIX,
    TGT_W,
    atr_percent,
    base_score,
    combine_percentile,
    combine_score,
    cross_percentile,
    ease_bonus,
    signal_pct,
    target_bonus,
    target_distance,
)
from alamet.selector import SelectorParams
from alamet.shadow import ShadowParams, run_with_shadow
from alamet.symbol import SymbolParams, symbol_signals
from alamet.universe import Universe


@dataclass(frozen=True)
class PanelParams:  # pylint: disable=too-many-instance-attributes
    selector: SelectorParams = SelectorParams(mode=4, conf_n=2)
    shadow: ShadowParams = ShadowParams()
    divergence: DivergenceParams = DivergenceParams()
    n_top: int = N_TOP
    gap_min: float = 0.0
    own_mode: str = MODE_PREV
    own: dict = field(default_factory=dict)
    use_target: bool = False
    tgt_mode: str = TGT_MIX
    tgt_w: float = TGT_W
    tgt_cap: float = TGT_CAP
    dr_rank: bool = False
    dr_rank_w: float = DR_RANK_W
    dr_src: str = DR_MIX
    symbol: SymbolParams = SymbolParams()
    rank_mode: str = RANK_ATR
    daily_open_hour: int = 0
    dr_rsi_len: int = RSI_LEN
    dr_look: int = LOOK


@dataclass
class SignalPanel:  # pylint: disable=too-many-instance-attributes
    universe: Universe
    atr: np.ndarray
    daily_open: np.ndarray
    weekly_open: np.ndarray
    monthly_open: np.ndarray
    power: np.ndarray
    types: np.ndarray
    starts: np.ndarray
    strengths: np.ndarray
    methods: np.ndarray
    pcts: np.ndarray
    base_scores: np.ndarray
    scores: np.ndarray
    own_ok: np.ndarray
    rank: RankResult
    divergence: dict
    confirmed: np.ndarray
    dr_percentile: np.ndarray
    dr_ratio: np.ndarray | None = None
    shadow_last: dict | None = None


def _pad_front(array: np.ndarray, rows: int) -> np.ndarray:
    if rows == 0:
        return array
    return np.concatenate([np.full((rows,) + array.shape[1:], np.nan), array])


def _one_symbol(universe: Universe, k: int, params: SymbolParams, open_hour: int):
    """Tek sembolün gösterge zinciri. Henüz işlem görmeyen başlangıç barları (kapanış `NaN`)
    atlanır ve sonuç başa `NaN` olarak eklenir: sembolün kendi serisi ilk barından başlar
    (TradingView'da `request.security` o sembolde olmayan geçmişi görmez)."""
    o, h, l, c = (getattr(universe, f)[:, k] for f in ("open", "high", "low", "close"))
    valid = ~np.isnan(c)
    first = int(np.argmax(valid)) if valid.any() else len(c)
    if first == len(c):
        empty = np.full(len(c), np.nan)
        blank = GateResult(empty, empty, empty, empty, empty)
        return (blank,) * 4, empty, empty, empty, empty, empty
    o, h, l, c = o[first:], h[first:], l[first:], c[first:]
    time = universe.time[first:]
    result = symbol_signals(o, h, l, c, universe.volume[first:, k], time, params)
    methods = tuple(
        GateResult(*(_pad_front(getattr(g, f.name), first) for f in fields(GateResult)))
        for g in result.methods
    )
    return (
        methods,
        _pad_front(result.mtf.final_pct, first),
        _pad_front(pine_atr(h, l, c, params.atr_len), first),
        _pad_front(daily_open(time, o, open_hour), first),
        _pad_front(period_levels(o, h, l, c, period_ids(time, "W")).open, first),
        _pad_front(period_levels(o, h, l, c, period_ids(time, "M")).open, first),
    )


def _symbol_arrays(universe: Universe, params: SymbolParams = SymbolParams(), open_hour: int = 0):
    gates, power, atr, d_open, w_open, m_open = [], [], [], [], [], []
    for k in range(universe.n):
        methods, p_k, atr_k, d_k, w_k, m_k = _one_symbol(universe, k, params, open_hour)
        gates.append(methods)
        power.append(p_k)
        atr.append(atr_k)
        d_open.append(d_k)
        w_open.append(w_k)
        m_open.append(m_k)
    names = ("last_type", "last_start", "last_str", "pend_dir", "last_sig")
    stacked = {
        key: np.stack([np.stack([getattr(g, f) for g in methods], axis=1) for methods in gates], 1)
        for key, f in zip(("dirs", "starts", "strs", "pends", "sigs"), names)
    }
    cols = (np.column_stack(x) for x in (power, atr, d_open, w_open, m_open))
    return stacked, *cols


def _own_gate(types, pcts, params: PanelParams, confirmed: np.ndarray) -> np.ndarray:
    machine = OwnDivergence(types.shape[1], **params.own)
    ok = np.zeros(types.shape, dtype=bool)
    for i in range(types.shape[0]):
        if confirmed[i]:
            machine.update(types[i], pcts[i])
        ok[i] = machine.gate(types[i], pcts[i], params.own_mode)[0]
    return ok


def _extra_scores(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    base, types, close, atr, own_ok, day: DayInputs | None, params: PanelParams
):
    """Hedef mesafesi puanı ve Devisso kolaylık puanı (DEPV2 f_tgtAll)."""
    dr_pct = np.full(base.shape, np.nan)
    if day is None:
        return base, dr_pct
    atr_p = atr_percent(atr, close)
    tgt = np.zeros(base.shape)
    if params.use_target:
        for k in range(base.shape[1]):
            dist = target_distance(
                day.levels[:, k, :], close[:, k], atr[:, k], types[:, k] == 1.0, params.tgt_mode
            )
            tgt[:, k] = target_bonus(
                dist, atr_p[:, k], params.tgt_w, params.tgt_cap, params.rank_mode
            )
    dr_pct = combine_percentile(day.percentile, cross_percentile(day.ratio), params.dr_src)
    bonus = (
        ease_bonus(dr_pct, params.dr_rank_w, atr_p, params.rank_mode)
        if params.dr_rank
        else np.zeros(base.shape)
    )
    return combine_score(base, tgt, bonus, own_ok), dr_pct


def _shadow_last(selected: dict) -> dict | None:
    """Gölge kasanın son bar durumu (Yöntem × Emtia tablosu için): (sembol, 5) diziler."""
    if not len(selected["book_side"]):
        return None
    last = {key: selected[f"book_{key}"][-1] for key in ("side", "count", "wins", "net")}
    last.update(entry=selected["last_entry"], notional=selected["last_notional"])
    last["auto"] = selected["s0_auto"][-1]
    return last


def build_panel(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    universe: Universe,
    params: PanelParams = PanelParams(),
    confirm_last: bool = False,
    start: int = 0,
    signal_start: int | None = None,
    day: DayInputs | None = None,
) -> SignalPanel:
    """`start`: ana hattın (seçici, gölge kasa, sıralama, ayrışma, motor) başladığı bar.

    `signal_start`: sembol sinyallerinin (A1–A8, request.security bağlamı) başladığı bar; en çok
    `start` kadar olabilir. Orijinalde ikisi farklı olabilir: grafik az bar yüklediyse ana hat
    grafiğin başından, sembol bağlamları kendi son `calc_bars_count` barından başlar.
    """
    signal_start = start if signal_start is None else min(signal_start, start)
    gates, power, atr, d_open, w_open, m_open = _symbol_arrays(
        universe.slice_from(signal_start), params.symbol, params.daily_open_hour
    )
    skip = start - signal_start
    gates = {key: value[skip:] for key, value in gates.items()}
    power, atr, d_open, w_open, m_open = (x[skip:] for x in (power, atr, d_open, w_open, m_open))
    universe = universe.slice_from(start)
    if day is not None:
        day = day.slice_from(start)
        w_open, m_open = day.weekly_open, day.monthly_open
    confirmed = np.ones(universe.bars, dtype=bool)
    confirmed[-1] = confirm_last
    selected = run_with_shadow(
        gates, universe.close, atr, (params.selector,), params.shadow, track_last=confirm_last
    )
    types, starts = selected["s0_sig_type"], selected["s0_start"]
    pcts = signal_pct(universe.close, starts)
    base = base_score(types, pcts, atr_percent(atr, universe.close), params.rank_mode)
    own_ok = _own_gate(types, pcts, params, confirmed)
    scores, dr_pct = _extra_scores(base, types, universe.close, atr, own_ok, day, params)
    rank = rank_all(scores, types, own_ok, params.n_top, params.gap_min, ties="pine")
    divergence = run_divergence(
        universe.close, types, rank.top, rank.idx_long, rank.idx_short,
        params.divergence, track_last=confirm_last,
    )  # fmt: skip
    return SignalPanel(
        universe, atr, d_open, w_open, m_open, power, types, starts, selected["s0_strength"],
        selected["s0_method"], pcts, base, scores, own_ok, rank, divergence, confirmed, dr_pct,
        None if day is None else day.ratio, _shadow_last(selected),
    )  # fmt: skip
