"""
Adım yöneticisi: her kapanmış sinyal barında bütün sistemi bir adım ilerletir.

Toplu hesap (`panel.build_panel` + `system.run_system`) bütün geçmişi bir kerede işler. Burada aynı
işlemler bar bar, AYNI SIRAYLA yapılır:

  1. sembol gösterge zinciri (`streaming.SymbolStream`) ve günlük açılış
  2. günlük bağlam (`daily_stream.DailyContext`)
  3. seçici + gölge kasa (`selector`, `shadow`)
  4. sinyal yüzdesi, temel puan, kendi-hareket kapısı
  5. hedef mesafesi ve Devisso puanı, sıralama (havuz maskesiyle), ayrışma takibi
  6. işlem motoru (`engine.TradeEngine.step`), istatistikler ve olaylar

Henüz işlem görmeyen ya da veri boşluğu olan sembollerde (kapanış `NaN`) gösterge zinciri ilerlemez
ve son değer taşınır; toplu hesaptaki `panel._expand` ile aynıdır. Eşdeğerlik `tests/test_stepper.py`
ile altın referansta ve rastgele evrenlerde sınanır.
"""

# pylint: disable=too-many-instance-attributes,too-many-locals
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alamet.daily_stream import DailyContext
from alamet.divergence import DivergenceTracker
from alamet.engine import BarInputs, TradeEngine, TradeParams
from alamet.events import EventLog
from alamet.own_divergence import OwnDivergence
from alamet.panel import PanelParams
from alamet.ranking import rank_all
from alamet.score import (
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
from alamet.selector import SignalSelector
from alamet.shadow import ShadowBook, _method_dirs
from alamet.stats import live_rows, stats_vector
from alamet.streaming import DailyOpen, SymbolStep, SymbolStream
from alamet.system import SECONDS_PER_DAY, log_divergence
from alamet.validate import validate_params

NAN = math.nan
LEVEL_COLUMNS = 18
BLANK_METHOD = (NAN, NAN, NAN, NAN, NAN)
BLANK_STEP = SymbolStep((BLANK_METHOD,) * 4, NAN, NAN)


@dataclass
class StepRecord:
    """Bir barın sonucu (toplu hesabın bar başına çıktılarıyla aynı büyüklükler)."""

    time_s: int
    sides: np.ndarray
    entries: np.ndarray
    notionals: np.ndarray
    ks: np.ndarray
    closed_count: int
    types: np.ndarray
    starts: np.ndarray
    strengths: np.ndarray
    methods: np.ndarray
    scores: np.ndarray
    own_ok: np.ndarray
    count: np.ndarray
    eligible: np.ndarray
    t_rank: np.ndarray
    stats: np.ndarray | None = None


class StepSystem:
    """`n_symbols` sembollü sistemin bar bar ilerleyen hali. Her çağrı bir kapanmış bardır."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        n_symbols: int,
        panel_params: PanelParams = PanelParams(),
        trade_params: TradeParams = TradeParams(),
        daily_mode: str = "history",
        first_day: int | None = None,
        collect_stats: bool = False,
        collect_events: bool = False,
    ) -> None:
        validate_params(panel_params, trade_params)
        if daily_mode not in ("history", "live"):
            raise ValueError("daily_mode 'history' ya da 'live' olmalı")
        self.n, self.pp, self.tp = n_symbols, panel_params, trade_params
        self.daily_mode, self.collect_stats = daily_mode, collect_stats
        tf = panel_params.symbol.signal_tf_seconds
        self.streams = [SymbolStream(panel_params.symbol) for _ in range(n_symbols)]
        self.d_open = [DailyOpen(panel_params.daily_open_hour) for _ in range(n_symbols)]
        self.last: list[SymbolStep] = [BLANK_STEP] * n_symbols
        self.last_d_open = np.full(n_symbols, NAN)
        self.daily = DailyContext(
            n_symbols, tf, panel_params.dr_rsi_len, panel_params.dr_look, first_day=first_day
        )
        self.selector = SignalSelector(n_symbols, panel_params.selector)
        self.book = ShadowBook(n_symbols, panel_params.shadow)
        self.own = OwnDivergence(n_symbols, **panel_params.own)
        self.tracker = DivergenceTracker(n_symbols, panel_params.divergence)
        self.engine = TradeEngine(n_symbols, trade_params)
        self.events = EventLog() if collect_events else None
        if self.events is not None:
            self.engine.events = self.events
        self.index = 0
        self.prev_time: int | None = None
        self.known, self.watch, self.closed_total = 0, [], 0
        self.bar_times: dict[int, int] = {}

    def _signal_rows(self, time_s, open_, high, low, close, volume):
        """Gösterge zinciri: geçerli satırlarda ilerler, diğerlerinde son değer taşınır."""
        for k in range(self.n):
            if math.isnan(close[k]):
                continue
            self.last[k] = self.streams[k].update(
                open_[k], high[k], low[k], close[k], volume[k], time_s
            )
            self.last_d_open[k] = self.d_open[k].update(time_s, open_[k])
        steps = self.last
        dirs, starts, strs, pends, sigs = (
            np.array([[step.methods[m][c] for m in range(4)] for step in steps]) for c in range(5)
        )
        atr = np.array([step.atr for step in steps])
        return (dirs, starts, strs, pends, sigs), atr

    def _day_inputs(self) -> np.ndarray:
        return self.daily.history() if self.daily_mode == "history" else self.daily.live()

    def _scores(self, types, pcts, close, atr, own_ok, pack):
        pp = self.pp
        atr_p = atr_percent(atr, close)
        base = base_score(types, pcts, atr_p, pp.rank_mode)
        levels, ratio, percentile = pack[:, :LEVEL_COLUMNS], pack[:, LEVEL_COLUMNS], pack[:, 19]
        tgt = np.zeros(base.shape)
        if pp.use_target:
            dist = target_distance(levels, close, atr, types == 1.0, pp.tgt_mode)
            tgt = target_bonus(dist, atr_p, pp.tgt_w, pp.tgt_cap, pp.rank_mode)
        dr_pct = combine_percentile(
            percentile[None, :], cross_percentile(ratio[None, :]), pp.dr_src
        )[0]
        bonus = (
            ease_bonus(dr_pct, pp.dr_rank_w, atr_p, pp.rank_mode)
            if pp.dr_rank
            else np.zeros(base.shape)
        )
        return combine_score(base, tgt, bonus, own_ok), dr_pct, ratio

    def step(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, time_s: int, open_, high, low, close, volume
    ) -> StepRecord:
        """Bir kapanmış bar; her dizi sembol başına bir değer (veri yoksa `NaN`)."""
        pp, engine = self.pp, self.engine
        open_, high, low, close, volume = (
            np.asarray(a, dtype=float) for a in (open_, high, low, close, volume)
        )
        time_s = int(time_s)
        if self.prev_time is not None and time_s <= self.prev_time:
            raise ValueError(
                f"Bar zamanı ilerlemeli: {time_s} <= {self.prev_time} (tekrar ya da eski bar)"
            )
        self.daily.on_bar(time_s, open_, high, low, close)
        pack = self._day_inputs()
        (dirs, starts_in, strs, pends, sigs), atr = self._signal_rows(
            time_s, open_, high, low, close, volume
        )
        out = self.selector.step(
            dirs, starts_in, strs, pends, sigs, True, self.book.count[:, :4], self.book.net_d[:, :4]
        )
        self.book.step(
            self.index,
            close,
            atr,
            np.concatenate([_method_dirs(dirs), self.selector.cb_dir[:, None].astype(int)], axis=1),
        )
        types, starts = out.sig_type, out.start
        pcts = signal_pct(close, starts)
        self.own.update(types, pcts)
        own_ok = self.own.gate(types, pcts, pp.own_mode)[0]
        scores, dr_pct, ratio = self._scores(types, pcts, close, atr, own_ok, pack)
        allowed = None
        pool = engine.pool
        if self.tp.pool.on:
            allowed = pool.protect_top1(pool.mask(), scores, types, own_ok)[None, :]
        rank = rank_all(
            scores[None, :], types[None, :], own_ok[None, :], pp.n_top, pp.gap_min, allowed, "pine"
        )
        div = self.tracker.step(
            close, types, rank.top[0], rank.idx_long[0], rank.idx_short[0], True
        )
        if self.events is not None:
            log_divergence(self.events, self.index, div.events)
        weekly, monthly = pack[:, 0], pack[:, 6]
        changed = self._day_changed(time_s)
        inputs = BarInputs(
            open_, high, low, close, atr, types, starts, self._power(), scores, div.eligible,
            div.t_rank, rank.idx_long[0], rank.idx_short[0], self.last_d_open.copy(), weekly,
            monthly, changed, dr_pct, div.count, out.strength, out.method, ratio,
        )  # fmt: skip
        engine.step(self.index, inputs, True)
        record = self._record(time_s, out, own_ok, scores, div, rank, close, types, starts)
        self.bar_times[self.index] = time_s
        self.prev_time = time_s
        self.index += 1
        return record

    def _power(self) -> np.ndarray:
        return np.array([step.final_pct for step in self.last])

    def _day_changed(self, time_s: int) -> bool:
        if self.prev_time is None:
            return False
        hour_utc = self.pp.daily_open_hour
        if hour_utc == 0:
            return time_s // SECONDS_PER_DAY != self.prev_time // SECONDS_PER_DAY
        hours = (time_s % SECONDS_PER_DAY) // 3600
        prev_hours = (self.prev_time % SECONDS_PER_DAY) // 3600
        return hours == hour_utc and prev_hours != hour_utc

    def _record(self, time_s, out, own_ok, scores, div, rank, close, types, starts) -> StepRecord:
        s, engine = self.engine.s, self.engine
        trades = engine.trades
        for trade in trades[self.known :]:
            trade.open_time = time_s
            self.watch.append(trade)
        self.known = len(trades)
        for trade in [t for t in self.watch if t.close_bar >= 0]:
            trade.close_time = time_s
            self.watch.remove(trade)
            self.closed_total += 1
        stats = None
        if self.collect_stats:
            rows = live_rows(rank.top[0], div.active, div.count, div.cur_move, scores, types, close)
            stats = stats_vector(
                engine, rows, engine.open_direction(close, 1), engine.open_direction(close, -1)
            )
        return StepRecord(
            time_s, np.where(s.active, s.side, 0), np.where(s.active, s.entry, np.nan),
            np.where(s.active, s.notional, 0.0), s.ks[:3].copy(), self.closed_total, types,
            starts, out.strength, out.method, scores, own_ok, div.count, div.eligible, div.t_rank,
            stats,
        )  # fmt: skip

    def drain(self) -> tuple[list, list]:
        """Kapanmış işlemleri ve biriken olayları verip bellekten siler (açık işlemler kalır).

        Kalıcı depoya yazılan veri bellekte tutulmaz; böylece uzun çalışmada bellek ve durum
        kaydının boyutu işlem sayısıyla büyümez.
        """
        closed = [t for t in self.engine.trades if t.close_bar >= 0]
        self.engine.trades = [t for t in self.engine.trades if t.close_bar < 0]
        self.known = len(self.engine.trades)
        events = self.events.drain() if self.events is not None else []
        return closed, events

    def undrain(self, closed: list, events: list, bar_times: dict[int, int]) -> None:
        """`drain` geri alınır (kalıcı yazma başarısız olduysa hiçbir şey kaybolmasın)."""
        self.engine.trades = sorted([*closed, *self.engine.trades], key=lambda t: t.trade_id)
        self.known = len(self.engine.trades)
        if self.events is not None:
            self.events._items = [*events, *self.events._items]  # pylint: disable=protected-access
        self.bar_times = {**bar_times, **self.bar_times}
