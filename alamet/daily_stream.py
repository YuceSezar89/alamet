"""
Günlük bağlamın adım adım (artımlı) karşılığı: `daily.daily_pack` + `map_to_bars` + `map_live`.

Toplu hesap bütün günlük geçmişi bir kerede işler; canlı bot ise her kapanmış sinyal barında bir
adım ilerler. Bir gün, son barı (23:45) kapanınca tamamlanır; o an her sembolün 20 sütunluk günlük
paketi (haftalık/aylık/yıllık seviyeler + Devisso oranı ve yüzdeliği) bir satır ilerler.

İki eşleme sunulur:
  - `history()` : TradingView geçmiş davranışı (doğrulanmış): ara barlar bir önceki günün paketini,
    günün son barı kendi gününün paketini görür.
  - `live()`    : gerçek zamanlı davranış: gelişen günün kısmi mumu seviyelere katılır, Devisso
    sütunları yalnızca kapanmış günlere bağlıdır.

Toplu hesapla farkı yalnızca şudur: toplu `map_live` bir günün (ileride) tam olup olmadığını bilir;
canlıda bu bilinemez, o yüzden eksik barlı bir günün ara barları da değer alır.
"""

# pylint: disable=too-many-instance-attributes
from __future__ import annotations

import math

import numpy as np

from alamet.daily import LEVEL_KINDS, PACK_SIZE
from alamet.devisso import LOOK, MIN_N, RSI_LEN, _rsi_value
from alamet.levels import period_ids
from alamet.timeframe import SignalTf

SECONDS_PER_DAY = 86400
NAN = math.nan
LEVEL_COLUMNS = 18
KINDS = tuple(kind for kind, _ in LEVEL_KINDS)


def period_id(day_ts: int, kind: str) -> int:
    return int(period_ids(np.array([day_ts], dtype=np.int64), kind)[0])


class _Level:
    """`levels.period_levels` döngüsünün durumu (bir periyot türü için)."""

    def __init__(self) -> None:
        self.c_o = self.c_h = self.c_l = self.p_h = self.p_l = self.p_c = NAN
        self.prev_id: int | None = None


class SymbolDaily:
    """Bir sembolün kapanmış günlerinden 20 sütunluk paket (`daily._symbol_pack` adım adım)."""

    def __init__(self, rsi_len: int = RSI_LEN, look: int = LOOK, min_n: int = MIN_N) -> None:
        self.rsi_len, self.look, self.min_n = rsi_len, look, min_n
        self.started = False
        self.index = -1
        self.levels = {kind: _Level() for kind in KINDS}
        self.prev_close = NAN
        self.total = NAN
        self.rsi = NAN
        self._gains: list[float] = []
        self._losses: list[float] = []
        self._up = self._down = NAN
        self._current = NAN
        self.ratios: list[float] = []

    def _delta(self, close: float) -> float:
        before = self.prev_close
        if self.index == 0 or math.isnan(before) or before == 0.0:
            return 0.0
        if close > before:
            return (close - before) / before * 100.0
        if close < before:
            return -(before - close) / close * 100.0
        return 0.0

    def _advance_rsi(self, change: float) -> float:
        length = self.rsi_len
        gain, loss = float(np.maximum(change, 0.0)), float(-np.minimum(change, 0.0))
        if self.index <= length:
            self._gains.append(gain)
            self._losses.append(loss)
            if self.index < length:
                return NAN
            self._up = float(np.array(self._gains).mean())
            self._down = float(np.array(self._losses).mean())
        else:
            self._up = (gain + (length - 1) * self._up) / length
            self._down = (loss + (length - 1) * self._down) / length
        return _rsi_value(self._up, self._down)

    def percentile_at(self, t: int) -> float:
        """`devisso.ratio_percentile` satırı: `t`. gün için yüzdelik (en küçük = 0)."""
        if t < 1 or t - 1 >= len(self.ratios):
            return NAN
        base = self.ratios[t - 1]
        if math.isnan(base):
            return NAN
        window = np.array(self.ratios[max(0, t - self.look - 1) : max(0, t - 1)])
        window = window[~np.isnan(window)]
        if len(window) < self.min_n:
            return NAN
        below = (window < base).sum() + 0.5 * (window == base).sum()
        return float(below / len(window) * 100.0)

    def close_day(self, day_ts: int, open_: float, high: float, low: float, close: float):
        """Kapanan günün paketi (20 değer). Sembolün ilk geçerli gününden önce hep `NaN`."""
        if not self.started:
            if math.isnan(close):
                return np.full(PACK_SIZE, NAN)
            self.started = True
        self.index += 1
        i = self.index
        row: list[float] = []
        for kind in KINDS:
            lv, ident = self.levels[kind], period_id(day_ts, kind)
            if (i > 0 and ident != lv.prev_id) or math.isnan(lv.c_o):
                if not math.isnan(lv.c_o):
                    lv.p_h, lv.p_l, lv.p_c = lv.c_h, lv.c_l, self.prev_close
                lv.c_o, lv.c_h, lv.c_l = open_, high, low
            else:
                lv.c_h = max(lv.c_h, high)
                lv.c_l = min(lv.c_l, low)
            lv.prev_id = ident
            row += [lv.c_o, lv.c_h, lv.c_l, lv.p_h, lv.p_l, lv.p_c]
        old_total, old_rsi = self.total, self.rsi
        self.total = self._delta(close) if i == 0 else self.total + self._delta(close)
        self.rsi = NAN if i == 0 else self._advance_rsi(self.total - old_total)
        d_tot = NAN if i == 0 else self.total - old_total
        d_rsi = NAN if i == 0 else self.rsi - old_rsi
        if not math.isnan(d_rsi) and not math.isnan(d_tot) and d_tot != 0.0:
            self._current = abs(d_rsi / d_tot)
        self.ratios.append(self._current)
        prev_ratio = self.ratios[i - 1] if i >= 1 else NAN
        self.prev_close = close
        return np.array(row + [prev_ratio, self.percentile_at(i)])

    def next_devisso(self) -> tuple[float, float]:
        """Henüz kapanmamış sıradaki günün Devisso sütunları (dünkü oran, yüzdelik)."""
        if not self.ratios:
            return NAN, NAN
        return self.ratios[-1], self.percentile_at(len(self.ratios))


class DailyContext:
    """Bütün sembollerin günlük bağlamı; her kapanmış sinyal barında `on_bar` çağrılır."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        n_symbols: int,
        tf_seconds: int = 900,
        rsi_len: int = RSI_LEN,
        look: int = LOOK,
        min_n: int = MIN_N,
        first_day: int | None = None,
    ) -> None:
        self.n = n_symbols
        self.tf = SignalTf(tf_seconds)
        self.first_day = first_day
        self.symbols = [SymbolDaily(rsi_len, look, min_n) for _ in range(n_symbols)]
        self.day: int | None = None
        self.count = 0
        self.whole = False
        self.time_s = 0
        self.d_open = self.d_high = self.d_low = np.full(n_symbols, NAN)
        self.last_pack = np.full((n_symbols, PACK_SIZE), NAN)
        self.last_close = np.full(n_symbols, NAN)
        self.last_day: int | None = None
        self.next_dr = np.full((n_symbols, 2), NAN)
        self.y_pack, self.y_close = self.last_pack, self.last_close
        self.y_day, self.y_dr = self.last_day, self.next_dr
        self._period_new: dict[str, bool] = {}

    def on_bar(
        self, time_s: int, open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> None:
        """Bir kapanmış bar (her dizi sembol başına bir değer). Günün son barında gün kapanır."""
        day = int(time_s) // SECONDS_PER_DAY * SECONDS_PER_DAY
        self.time_s = int(time_s)
        if day != self.day:
            self.y_pack, self.y_close = self.last_pack, self.last_close
            self.y_day, self.y_dr = self.last_day, self.next_dr
            self.day, self.count = day, 0
            self.whole = int(time_s) % SECONDS_PER_DAY == 0
            self.d_open, self.d_high, self.d_low = open_.copy(), high.copy(), low.copy()
            self._period_new = {
                kind: period_id(day, kind) != period_id(day - SECONDS_PER_DAY, kind)
                for kind in KINDS
            }
        else:
            self.d_high = np.maximum(self.d_high, high)
            self.d_low = np.minimum(self.d_low, low)
        self.count += 1
        if self.is_last_bar(time_s) and self.count == self.tf.bars_per_day and self.whole:
            self._close_day(day, close)

    def is_last_bar(self, time_s: int) -> bool:
        return int(time_s) % SECONDS_PER_DAY == self.tf.last_bar_offset

    def _close_day(self, day: int, close: np.ndarray) -> None:
        if self.first_day is not None and day < self.first_day:
            return
        rows = [
            sym.close_day(day, self.d_open[k], self.d_high[k], self.d_low[k], close[k])
            for k, sym in enumerate(self.symbols)
        ]
        self.last_pack = np.array(rows)
        self.last_close = np.array(close, dtype=float)
        self.last_day = day
        self.next_dr = np.array([sym.next_devisso() for sym in self.symbols])

    def history(self) -> np.ndarray:
        """TradingView geçmiş eşlemesi: (sembol, 20). Son bar kendi gününü, diğerleri dünü görür."""
        if self.is_last_bar(self.time_s):
            pack, found = self.last_pack, self.last_day == self.day
        else:
            pack, found = self.y_pack, self.y_day == self.day - SECONDS_PER_DAY
        return pack.copy() if found else np.full((self.n, PACK_SIZE), NAN)

    def live(self) -> np.ndarray:
        """Gerçek zamanlı eşleme: gelişen günün kısmi mumu seviyelere katılır."""
        out = np.full((self.n, PACK_SIZE), NAN)
        if self.y_day != self.day - SECONDS_PER_DAY or not self.whole:
            return out
        before = self.y_pack
        for kind, c0 in LEVEL_KINDS:
            new = self._period_new[kind]
            out[:, c0] = self.d_open if new else before[:, c0]
            out[:, c0 + 1] = self.d_high if new else np.maximum(before[:, c0 + 1], self.d_high)
            out[:, c0 + 2] = self.d_low if new else np.minimum(before[:, c0 + 2], self.d_low)
            out[:, c0 + 3] = before[:, c0 + 1] if new else before[:, c0 + 3]
            out[:, c0 + 4] = before[:, c0 + 2] if new else before[:, c0 + 4]
            out[:, c0 + 5] = self.y_close if new else before[:, c0 + 5]
        out[:, LEVEL_COLUMNS:] = self.y_dr
        return out
