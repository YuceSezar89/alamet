"""
Günlük paket (DEPV2 f_dayPack): haftalık/aylık/yıllık seviyeler ve Devisso oranı, 15 dk iniş.

Kaynak Pine: pine/DEPV2 debug kopyası (f_per 617-638, f_dayPack 639-666, mDay 739).
Paket sütunları (20): 0-5 haftalık [open, high, low, önceki high, low, close], 6-11 aylık,
12-17 yıllık, 18 Devisso oranı (dünkü, drB), 19 oranın son 60 gündeki yüzdeliği (drPr).

Gecikme: orijinal `request.security(sembol, "D", f_dayPack())` çağrısında `lookahead` yoktur.
TradingView'ın varsayılanıyla geçmiş 15 dk barlar, o günün değil BİR ÖNCEKİ günün kapanmış paketini
görür; yalnızca günün son 15 dk barı (23:45) o günün kendi paketini görür ("history" modu).
Üretim davranışı ("live": son bar gelişen günün paketini görür) 15 dk verisinden hesaplanır ve
henüz taşınmadı.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alamet.devisso import LOOK, RSI_LEN, devisso_daily
from alamet.levels import period_ids, period_levels
from alamet.timeframe import SignalTf
from alamet.universe import SYMBOLS, Universe

SECONDS_PER_DAY = 86400
LAST_BAR_OFFSET = SECONDS_PER_DAY - 900
BARS_PER_DAY = 96
PACK_SIZE = 20
LEVEL_COLUMNS = 18
DR_RATIO = 18
DR_PERCENTILE = 19
HISTORY = "history"
LIVE = "live"


@dataclass(frozen=True)
class DailyUniverse:
    symbols: tuple[str, ...]
    time: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    @property
    def n(self) -> int:
        return len(self.symbols)


def load_daily(path_1: str, path_2: str, symbols: tuple[str, ...] = SYMBOLS) -> DailyUniverse:
    first = pd.read_csv(path_1).set_index("time")
    second = pd.read_csv(path_2).set_index("time")
    common = first.index.intersection(second.index)
    first, second = first.loc[common], second.loc[common]
    fields = {key: [] for key in ("open", "high", "low", "close")}
    for number in range(1, len(symbols) + 1):
        if number == 1:
            columns, frame = ("open", "high", "low", "close"), first
        else:
            columns = tuple(f"{letter}{number:02d}" for letter in "OHLC")
            frame = first if number <= 12 else second
        for key, column in zip(fields, columns):
            fields[key].append(frame[column].to_numpy(float))
    stacked = {key: np.column_stack(values) for key, values in fields.items()}
    return DailyUniverse(symbols, common.to_numpy(np.int64), **stacked)


def daily_from_bars(universe: Universe, tf_seconds: int = 900) -> DailyUniverse:
    """Sinyal TF'si evreninden UTC günlük mumlar; yalnızca günün bütün barları bulunan
    (kapanmış) günler alınır."""
    per_day = SignalTf(tf_seconds).bars_per_day
    time = np.asarray(universe.time, dtype=np.int64)
    days = time // SECONDS_PER_DAY
    first, counts = np.unique(days, return_index=True, return_counts=True)[1:]
    full = (counts == per_day) & (time[first] % SECONDS_PER_DAY == 0)
    starts = first[full]
    return DailyUniverse(
        universe.symbols,
        days[starts] * SECONDS_PER_DAY,
        universe.open[starts],
        np.maximum.reduceat(universe.high, first)[full],
        np.minimum.reduceat(universe.low, first)[full],
        universe.close[starts + per_day - 1],
    )


def window_daily(
    daily: DailyUniverse,
    window_start_time: int,
    last_time: int | None = None,
    calc_bars: int = 4000,
    tf_seconds: int = 900,
) -> DailyUniverse:
    """calc_bars_count penceresi: günlük bağlam son `calc_bars // günlük bar sayısı` günlük bardan
    oluşur (15 dk'da 4000 // 96 = 41 gün; 5 dk'da 13 gün).

    `last_time` (grafiğin son barı) verilirse TradingView kuralı uygulanır: son günü içeren
    o kadar gün. Verilmezse pencere içindeki ilk tam günden başlar (yalnızca pencere sabah
    erken başlıyorsa TradingView ile aynıdır).
    """
    per_day = SignalTf(tf_seconds).bars_per_day
    if last_time is None:
        first_day = -(-int(window_start_time) // SECONDS_PER_DAY) * SECONDS_PER_DAY
    else:
        first_day = (
            int(last_time) // SECONDS_PER_DAY - max(calc_bars // per_day - 1, 0)
        ) * SECONDS_PER_DAY
    keep = daily.time >= first_day
    return DailyUniverse(
        daily.symbols, daily.time[keep], daily.open[keep], daily.high[keep],
        daily.low[keep], daily.close[keep],
    )  # fmt: skip


def _symbol_pack(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    time: np.ndarray, o, h, l, c, rsi_len: int = RSI_LEN, look: int = LOOK
) -> np.ndarray:
    out = np.full((len(time), PACK_SIZE), np.nan)
    valid = ~np.isnan(c)
    if not valid.any():
        return out
    first = int(np.argmax(valid))
    t, o, h, l, c = time[first:], o[first:], h[first:], l[first:], c[first:]
    columns = []
    for kind in ("W", "M", "Y"):
        lv = period_levels(o, h, l, c, period_ids(t, kind))
        columns += [lv.open, lv.high, lv.low, lv.prev_high, lv.prev_low, lv.prev_close]
    dr = devisso_daily(c, rsi_len, look)
    columns += [dr.prev_ratio, dr.percentile]
    out[first:] = np.column_stack(columns)
    return out


def daily_pack(daily: DailyUniverse, rsi_len: int = RSI_LEN, look: int = LOOK) -> np.ndarray:
    """(gün, sembol, 20) paket; her gün o günün KAPANMIŞ değerleridir."""
    return np.stack(
        [
            _symbol_pack(
                daily.time, daily.open[:, k], daily.high[:, k], daily.low[:, k],
                daily.close[:, k], rsi_len, look,
            )
            for k in range(daily.n)
        ],
        axis=1,
    )  # fmt: skip


def map_to_bars(
    pack: np.ndarray,
    daily_time: np.ndarray,
    bar_time: np.ndarray,
    mode: str = HISTORY,
    tf_seconds: int = 900,
):
    """Günlük paketi sinyal TF'sinin barlarına indirir (bar, sembol, 20)."""
    if mode == LIVE:
        raise NotImplementedError("live modu için map_live(pack, daily, universe) kullanılır")
    if mode != HISTORY:
        raise ValueError(f"bilinmeyen mod: {mode}")
    bar_time = np.asarray(bar_time, dtype=np.int64)
    day_start = (bar_time // SECONDS_PER_DAY) * SECONDS_PER_DAY
    is_last = (bar_time - day_start) == SignalTf(tf_seconds).last_bar_offset
    wanted = np.where(is_last, day_start, day_start - SECONDS_PER_DAY)
    index = np.searchsorted(daily_time, wanted)
    index = np.clip(index, 0, len(daily_time) - 1)
    found = daily_time[index] == wanted
    out = np.full((len(bar_time),) + pack.shape[1:], np.nan)
    out[found] = pack[index[found]]
    return out


LEVEL_KINDS = (("W", 0), ("M", 6), ("Y", 12))


def _partial_day(universe: Universe) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Her bar için günün o bara kadarki high/low'u, açılışı ve günün tam başlayıp başlamadığı."""
    day = np.asarray(universe.time, dtype=np.int64) // SECONDS_PER_DAY * SECONDS_PER_DAY
    _, start, counts = np.unique(day, return_index=True, return_counts=True)
    high, low = np.empty_like(universe.high), np.empty_like(universe.low)
    day_open = np.empty_like(universe.open)
    starts_at_midnight = np.zeros(len(day), dtype=bool)
    for first, count in zip(start, counts):
        part = slice(first, first + count)
        high[part] = np.maximum.accumulate(universe.high[part], axis=0)
        low[part] = np.minimum.accumulate(universe.low[part], axis=0)
        day_open[part] = universe.open[first]
        starts_at_midnight[part] = universe.time[first] % SECONDS_PER_DAY == 0
    return high, low, day_open, starts_at_midnight


def _row_of(daily_time: np.ndarray, wanted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = np.clip(np.searchsorted(daily_time, wanted), 0, len(daily_time) - 1)
    return index, daily_time[index] == wanted


def _with_today(daily: DailyUniverse, today: int) -> DailyUniverse:
    """Bugün kapanmamış olduğundan günlüğe yer tutucu gün ekler (yalnızca Devisso oranı için)."""
    if daily.time[-1] >= today:
        return daily
    filler = daily.close[-1:]
    return DailyUniverse(
        daily.symbols, np.append(daily.time, today), np.vstack([daily.open, filler]),
        np.vstack([daily.high, filler]), np.vstack([daily.low, filler]),
        np.vstack([daily.close, filler]),
    )  # fmt: skip


def map_live(  # pylint: disable=too-many-locals
    pack: np.ndarray, daily: DailyUniverse, universe: Universe, rsi_len: int = RSI_LEN,
    look: int = LOOK,
) -> np.ndarray:  # fmt: skip
    """Canlı davranış: 15 dk bar, o günün KISMİ mumunu görür (orijinal `request.security`
    gerçek zamanlı barı). Hafta/ay/yıl high-low'u kısmi günle birleşir, açılış ve önceki-periyot
    sütunları periyot değişiminde güncellenir; Devisso sütunları yalnızca kapanmış günlere bağlıdır
    (drB = dünkü oran). Günün son barında sonuç `map_to_bars` HISTORY paketiyle aynıdır.

    `pack`/`daily` kapanmış günlerden üretilmiş olmalıdır (içinde bugün yoktur).
    """
    bar_time = np.asarray(universe.time, dtype=np.int64)
    day = bar_time // SECONDS_PER_DAY * SECONDS_PER_DAY
    high, low, day_open, whole = _partial_day(universe)
    extended = _with_today(daily, int(day[-1]))
    ratio_pack = pack if extended is daily else daily_pack(extended, rsi_len, look)
    ratio_rows, ratio_ok = _row_of(extended.time, day)
    prev, prev_ok = _row_of(daily.time, day - SECONDS_PER_DAY)
    out = np.full((len(bar_time),) + pack.shape[1:], np.nan)
    before, close_prev = pack[prev], daily.close[prev]
    for kind, c0 in LEVEL_KINDS:
        new = (period_ids(day, kind) != period_ids(day - SECONDS_PER_DAY, kind))[:, None]
        out[:, :, c0] = np.where(new, day_open, before[:, :, c0])
        out[:, :, c0 + 1] = np.where(new, high, np.maximum(before[:, :, c0 + 1], high))
        out[:, :, c0 + 2] = np.where(new, low, np.minimum(before[:, :, c0 + 2], low))
        out[:, :, c0 + 3] = np.where(new, before[:, :, c0 + 1], before[:, :, c0 + 3])
        out[:, :, c0 + 4] = np.where(new, before[:, :, c0 + 2], before[:, :, c0 + 4])
        out[:, :, c0 + 5] = np.where(new, close_prev, before[:, :, c0 + 5])
    out[:, :, DR_RATIO:] = ratio_pack[ratio_rows][:, :, DR_RATIO:]
    out[~(prev_ok & ratio_ok & whole)] = np.nan
    return out


@dataclass(frozen=True)
class DayInputs:
    """15 dk barlara inmiş günlük paketin panelin kullandığı dilimleri (bar, sembol[, 18])."""

    levels: np.ndarray
    ratio: np.ndarray
    percentile: np.ndarray
    weekly_open: np.ndarray
    monthly_open: np.ndarray

    def slice_from(self, start: int) -> "DayInputs":
        return DayInputs(
            self.levels[start:], self.ratio[start:], self.percentile[start:],
            self.weekly_open[start:], self.monthly_open[start:],
        )  # fmt: skip


def day_inputs(mapped: np.ndarray) -> DayInputs:
    return DayInputs(
        mapped[..., :LEVEL_COLUMNS],
        mapped[..., DR_RATIO],
        mapped[..., DR_PERCENTILE],
        mapped[..., 0],
        mapped[..., 6],
    )
