"""
20 sembollü evren: TradingView dışa aktarımlarından (DATA_1.pine, DATA_2.pine) OHLCV matrisleri.

Dosyalar aynı grafikten (BTCUSDT.P, 15 dk) alındığı için zaman ekseni ortaktır; iki dosya
`time` üzerinden birleştirilir, yalnızca ortak aralık tutulur. Sembol 1 grafiğin kendi
open/high/low/close sütunları ve VOL01'dir, 2..20 için O/H/L/C/V + iki haneli sıra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SYMBOLS = (
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "TRX",
    "DOT", "LTC", "BCH", "ATOM", "NEAR", "APT", "ARB", "OP", "SUI", "INJ",
)  # fmt: skip


@dataclass(frozen=True)
class Universe:
    symbols: tuple[str, ...]
    time: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def slice_from(self, start: int) -> "Universe":
        return Universe(
            self.symbols, self.time[start:], self.open[start:], self.high[start:],
            self.low[start:], self.close[start:], self.volume[start:],
        )  # fmt: skip

    def slice_to(self, stop: int) -> "Universe":
        return Universe(
            self.symbols, self.time[:stop], self.open[:stop], self.high[:stop],
            self.low[:stop], self.close[:stop], self.volume[:stop],
        )  # fmt: skip

    @property
    def bars(self) -> int:
        return len(self.time)

    @property
    def n(self) -> int:
        return len(self.symbols)

    @property
    def bar_seconds(self) -> int | None:
        """Bar aralığı (sn): zaman adımlarının medyanı; tek barda bilinmez."""
        if len(self.time) < 2:
            return None
        return int(np.median(np.diff(np.asarray(self.time, dtype=np.int64))))


def load_universe(path_1: str, path_2: str, symbols: tuple[str, ...] = SYMBOLS) -> Universe:
    first = pd.read_csv(path_1).set_index("time")
    second = pd.read_csv(path_2).set_index("time")
    common = first.index.intersection(second.index)
    first, second = first.loc[common], second.loc[common]
    fields = {key: [] for key in ("open", "high", "low", "close", "volume")}
    for number in range(1, len(symbols) + 1):
        if number == 1:
            columns = ("open", "high", "low", "close", "VOL01")
            frame = first
        else:
            columns = tuple(f"{letter}{number:02d}" for letter in "OHLCV")
            frame = first if number <= 12 else second
        for key, column in zip(fields, columns):
            fields[key].append(frame[column].to_numpy(float))
    stacked = {key: np.column_stack(values) for key, values in fields.items()}
    return Universe(symbols, common.to_numpy(np.int64), **stacked)


def resample_universe(universe: Universe, seconds: int) -> Universe:
    """İnce barları `seconds` aralıklı barlara birleştirir (açılış ilk, kapanış son, high/low uç,
    hacim toplam). Yalnızca bütün alt barları bulunan (kapanmış) kovalar alınır; kovalar UTC
    gün başına hizalıdır (`timeframe.change` ile aynı)."""
    base = universe.bar_seconds
    if base is None or seconds % base != 0 or seconds < base:
        raise ValueError(f"{seconds} sn, {base} sn barların tam katı olmalı")
    per = seconds // base
    time = np.asarray(universe.time, dtype=np.int64)
    bucket = time // seconds
    first, counts = np.unique(bucket, return_index=True, return_counts=True)[1:]
    full = (counts == per) & (time[first] % seconds == 0)
    starts = first[full]
    return Universe(
        universe.symbols,
        bucket[starts] * seconds,
        universe.open[starts],
        np.maximum.reduceat(universe.high, first)[full],
        np.minimum.reduceat(universe.low, first)[full],
        universe.close[starts + per - 1],
        np.add.reduceat(universe.volume, first)[full],
    )
