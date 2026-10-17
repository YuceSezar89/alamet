"""
Canlı mum birleştirici: kapanmış 1 dk mumlardan sinyal TF'sinin mumunu kurar.

Kural (Baybora'daki "resample" ile aynı): bir sinyal TF mumu, içindeki bütün 1 dk mumlar geldiyse
tamamdır. Bütün semboller tamamsa hemen, değilse bekleme süresi (`grace_ms`) dolunca kapatılır;
eksik semboller `missing` listesinde döner ve REST ile onarılır. Mumlar yalnızca sırayla çıkar
(bir önceki mum çözülmeden sonraki verilmez); hiç veri gelmeyen mumlar da eksik olarak çıkar.

Saf koddur (ağ, saat, dosya yok): zaman `ready(now_ms)` ile dışarıdan verilir. Kovalar UTC'ye
hizalıdır (`zaman damgası // TF`), Pine'daki `timeframe.change` ile aynı.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from alamet.timeframe import SignalTf

MINUTE_MS = 60_000
DEFAULT_GRACE_MS = 15_000


@dataclass(frozen=True)
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float


@dataclass(frozen=True)
class ClosedBucket:
    """Çözülmüş sinyal TF mumu: tam olan semboller ve onarım bekleyenler."""

    start_ms: int
    candles: dict[str, Candle]
    missing: tuple[str, ...]


@dataclass
class AggregatorStats:
    accepted: int = 0
    duplicates: int = 0
    late: int = 0
    unknown: int = 0
    repaired_buckets: int = 0
    missing_candles: int = 0


def parse_kline_message(raw: str | bytes) -> tuple[str, Candle, bool] | None:
    """Birleşik akış mesajından (sembol, 1 dk mum, kapandı mı); kline değilse `None`."""
    message = json.loads(raw)
    data = message.get("data", message)
    if data.get("e") != "kline":
        return None
    kline = data["k"]
    candle = Candle(
        int(kline["t"]), float(kline["o"]), float(kline["h"]), float(kline["l"]),
        float(kline["c"]), float(kline["v"]), float(kline["q"]),
    )  # fmt: skip
    return str(data["s"]), candle, bool(kline["x"])


def merge_minutes(bucket_start: int, minutes: Sequence[Candle]) -> Candle:
    """Sıralı 1 dk mumlardan tek mum: açılış ilk, kapanış son, uçlar, hacim toplamı."""
    return Candle(
        bucket_start,
        minutes[0].open,
        max(m.high for m in minutes),
        min(m.low for m in minutes),
        minutes[-1].close,
        sum(m.volume for m in minutes),
        sum(m.quote_volume for m in minutes),
    )


@dataclass
class BarAggregator:  # pylint: disable=too-many-instance-attributes
    symbols: tuple[str, ...]
    tf: SignalTf
    grace_ms: int = DEFAULT_GRACE_MS
    stats: AggregatorStats = field(default_factory=AggregatorStats)
    _minutes: dict[int, dict[str, dict[int, Candle]]] = field(default_factory=dict)
    _next: int | None = None

    def __post_init__(self) -> None:
        if self.tf.seconds % 60:
            raise ValueError("Sinyal TF'si dakikanın katı olmalı")
        self.per = self.tf.seconds // 60
        self._known = set(self.symbols)

    def bucket_of(self, open_time_ms: int) -> int:
        return open_time_ms // self.tf.millis * self.tf.millis

    @property
    def next_bucket(self) -> int | None:
        return self._next

    def start_at(self, bucket_start_ms: int) -> None:
        """Birleştirmenin başlayacağı ilk sinyal mumu (öncesi geçmişten/REST'ten gelir)."""
        if bucket_start_ms % self.tf.millis:
            raise ValueError("Başlangıç mumu sinyal TF'sine hizalı olmalı")
        self._next = bucket_start_ms

    def add(self, symbol: str, candle: Candle) -> bool:
        """Kapanmış 1 dk mum ekler; bilinmeyen sembol ve çıkmış muma geç veri reddedilir."""
        if symbol not in self._known:
            self.stats.unknown += 1
            return False
        bucket = self.bucket_of(candle.open_time)
        if self._next is not None and bucket < self._next:
            self.stats.late += 1
            return False
        minutes = self._minutes.setdefault(bucket, {}).setdefault(symbol, {})
        if candle.open_time in minutes:
            self.stats.duplicates += 1
        else:
            self.stats.accepted += 1
        minutes[candle.open_time] = candle
        if self._next is None:
            self._next = bucket
        return True

    def ready(self, now_ms: int) -> list[ClosedBucket]:
        """Çözülmüş mumlar, sırayla: hepsi tamamsa hemen, değilse bekleme süresi dolunca."""
        out: list[ClosedBucket] = []
        while self._next is not None:
            start = self._next
            data = self._minutes.get(start, {})
            complete = [s for s in self.symbols if len(data.get(s, ())) == self.per]
            end = start + self.tf.millis
            if len(complete) < len(self.symbols) and now_ms < end + self.grace_ms:
                break
            candles = {
                s: merge_minutes(start, [data[s][k] for k in sorted(data[s])]) for s in complete
            }
            missing = tuple(s for s in self.symbols if s not in candles)
            if missing:
                self.stats.repaired_buckets += 1
                self.stats.missing_candles += len(missing)
            out.append(ClosedBucket(start, candles, missing))
            self._minutes.pop(start, None)
            self._next = end
        return out
