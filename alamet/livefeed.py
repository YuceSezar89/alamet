"""
Canlı veri akışı (Baybora deseni): WebSocket önce, geçmiş arkada, sürekli onarım.

Akış:
  1. `prepare`: önbellekteki son mumdan şimdiye kadar eksik geçmişi REST ile tamamla, birleştiriciyi
     bir sonraki sinyal mumundan başlat (öncesi artık geçmiştir, WebSocket'e bağlı değildir).
  2. WebSocket: sembol başına `kline_1m`; yalnızca kapanmış dakikalar birleştiriciye girer.
  3. `process`: tam ya da süresi dolmuş sinyal mumları sırayla çözülür; eksik semboller REST ile
     onarılır; önbelleğe yazılır; `on_bar` çağrılır.
  4. `reconcile`: son mumları Binance'in resmî mumlarıyla karşılaştırır; farklıysa düzeltir ve
     `on_correction` ile bildirir (kopma, geç işlem verisi gibi sessiz bozulmaları yakalar).

Ağ ve saat dışarıdan değiştirilebilir (`fetch_candles`, `start_stream`, `now_ms`), bu yüzden
mantığın tamamı ağsız test edilir. Bot yalnızca KAPANMIŞ mumları görür.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alamet.aggregator import BarAggregator, Candle, parse_kline_message
from alamet.binance_data import (
    COLUMNS,
    PageFetcher,
    default_root,
    load_cache,
    save_cache,
    update_symbol,
)
from alamet.exchange import default_client
from alamet.market import bases, load_snapshot
from alamet.stream import StreamManager
from alamet.timeframe import SignalTf, from_interval

REPAIR_ATTEMPTS = 5
REPAIR_DELAY_S = 2.0
SETTLE_MS = 30_000
VOLUME_TOLERANCE = 1e-9

CandleFetcher = Callable[[str, int, int], Awaitable[list[Candle]]]
StreamStarter = Callable[
    [list[str], Callable[[str], None]], Awaitable[Callable[[], Awaitable[None]]]
]


@dataclass(frozen=True)
class BarSet:
    """Çözülmüş bir sinyal mumu: sembol (taban ad) → mum. `missing` onarılamayanlar,
    `repaired` REST ile tamamlananlar."""

    start_ms: int
    candles: dict[str, Candle]
    repaired: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class Correction:
    base: str
    start_ms: int
    old: Candle | None
    new: Candle


def candles_from_frame(frame: pd.DataFrame) -> list[Candle]:
    return [
        Candle(int(r.open_time), r.open, r.high, r.low, r.close, r.volume, r.quote_volume)
        for r in frame[COLUMNS].itertuples()
    ]


def same_candle(a: Candle, b: Candle) -> bool:
    """OHLC birebir, hacimler göreli 1e-9 içinde."""
    if (a.open, a.high, a.low, a.close) != (b.open, b.high, b.low, b.close):
        return False
    return all(
        abs(x - y) <= VOLUME_TOLERANCE * max(abs(y), 1.0)
        for x, y in ((a.volume, b.volume), (a.quote_volume, b.quote_volume))
    )


def store_candle(root: Path, base: str, candle: Candle) -> None:
    """Mumu önbelleğe yazar (aynı zaman damgası varsa değiştirir), sıralı tutar."""
    cache = load_cache(root, base)
    row = pd.DataFrame([candle.__dict__])[COLUMNS]
    merged = pd.concat([cache[cache["open_time"] != candle.open_time], row], ignore_index=True)
    save_cache(root, base, merged.sort_values("open_time").reset_index(drop=True))


async def rest_candles(symbol: str, interval: str, start_ms: int, limit: int) -> list[Candle]:
    """Binance REST → kapanmış Candle listesi."""
    return candles_from_frame(await default_client().klines(symbol, interval, limit, start_ms))


async def websocket_starter(
    streams: list[str], on_message: Callable[[str], None]
) -> Callable[[], Awaitable[None]]:
    """WebSocket akışlarını açar; durdurma fonksiyonu döner."""
    manager = StreamManager(on_message)
    await manager.start(streams)
    return manager.stop


class LiveFeed:  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        root: Path,
        symbols: tuple[str, ...],
        tf: SignalTf,
        on_bar: Callable[[BarSet], Awaitable[None] | None],
        on_correction: Callable[[Correction], Awaitable[None] | None] | None = None,
        fetch_candles: CandleFetcher = rest_candles,
        start_stream: StreamStarter = websocket_starter,
        now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
        grace_ms: int = 15_000,
        repair_attempts: int = REPAIR_ATTEMPTS,
        repair_delay_s: float = REPAIR_DELAY_S,
        fetch_page: PageFetcher | None = None,
        settle_ms: int = SETTLE_MS,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.root, self.symbols, self.tf = root, symbols, tf
        self.on_bar, self.on_correction = on_bar, on_correction
        self.fetch_candles, self.start_stream, self.now_ms = fetch_candles, start_stream, now_ms
        self.repair_attempts, self.repair_delay_s = repair_attempts, repair_delay_s
        self.fetch_page = fetch_page
        self.settle_ms = settle_ms
        self.log = log or (lambda _text: None)
        self._reconciled = 0
        self.aggregator = BarAggregator(symbols, tf, grace_ms)
        self._stop: Callable[[], Awaitable[None]] | None = None

    def _symbol(self, base: str) -> str:
        return f"{base}USDT"

    def on_message(self, raw: str) -> None:
        parsed = parse_kline_message(raw)
        if parsed is None or not parsed[2]:
            return
        symbol, candle, _ = parsed
        self.aggregator.add(symbol.removesuffix("USDT"), candle)

    async def prepare(self) -> int:
        """Geçmişi şimdiye kadar tamamla ve birleştiriciyi sıradaki sinyal mumundan başlat."""
        last = []
        for number, base in enumerate(self.symbols, 1):
            if number % 25 == 0:
                self.log(f"  geçmiş tamamlanıyor {number}/{len(self.symbols)}")
            cache = load_cache(self.root, base)
            if cache.empty:
                raise ValueError(f"{base}: önbellek boş; önce alamet.bulk_data ile geçmişi indir")
            await update_symbol(self.root, base, 0, self.fetch_page, tf=self.tf)
            last.append(int(load_cache(self.root, base)["open_time"].iloc[-1]))
        start = max(last) + self.tf.millis
        self.aggregator.start_at(start)
        return start

    async def start(self) -> None:
        streams = [f"{self._symbol(b).lower()}@kline_1m" for b in self.symbols]
        self._stop = await self.start_stream(streams, self.on_message)

    async def stop(self) -> None:
        if self._stop is not None:
            await self._stop()
            self._stop = None

    async def _repair(self, bucket_start: int, missing: tuple[str, ...]) -> dict[str, Candle]:
        found: dict[str, Candle] = {}
        for attempt in range(self.repair_attempts):
            todo = [b for b in missing if b not in found]
            if not todo:
                break
            if attempt:
                await asyncio.sleep(self.repair_delay_s)
            results = await asyncio.gather(
                *(
                    self.fetch_candles(self._symbol(b), self.tf.interval, bucket_start, 1)
                    for b in todo
                )
            )
            for base, candles in zip(todo, results):
                hit = [c for c in candles if c.open_time == bucket_start]
                if hit:
                    found[base] = hit[0]
        return found

    async def process(self, now_ms: int | None = None) -> list[BarSet]:
        """Çözülmüş sinyal mumlarını sırayla işler: onar, yaz, bildir."""
        now = self.now_ms() if now_ms is None else now_ms
        out = []
        for closed in self.aggregator.ready(now):
            candles = dict(closed.candles)
            repaired: dict[str, Candle] = {}
            if closed.missing:
                repaired = await self._repair(closed.start_ms, closed.missing)
                candles.update(repaired)
            missing = tuple(b for b in self.symbols if b not in candles)
            for base, candle in candles.items():
                store_candle(self.root, base, candle)
            ordered = {b: candles[b] for b in self.symbols if b in candles}
            bar_set = BarSet(closed.start_ms, ordered, tuple(repaired), missing)
            result = self.on_bar(bar_set)
            if asyncio.iscoroutine(result):
                await result
            out.append(bar_set)
        return out

    async def reconcile(self, buckets: int = 3, now_ms: int | None = None) -> list[Correction]:
        """Son `buckets` mumu Binance'in resmî mumlarıyla karşılaştırır; farkı düzeltir ve bildirir.

        Yalnızca kapanışı `settle_ms` kadar geçmiş mumlara bakılır: kapanıştan hemen sonra resmî
        mum henüz eksik olabilir (gölge çalışmada +1 sn'de hacim ve kapanış geride kaldı) ve iyi
        olan birleştirilmiş mumun üzerine yazılırdı.
        """
        now = self.now_ms() if now_ms is None else now_ms
        last_closed = (now - self.settle_ms) // self.tf.millis * self.tf.millis - self.tf.millis
        first = last_closed - (buckets - 1) * self.tf.millis
        results = await asyncio.gather(
            *(
                self.fetch_candles(self._symbol(b), self.tf.interval, first, buckets)
                for b in self.symbols
            )
        )
        corrections = []
        for base, official in zip(self.symbols, results):
            cache = load_cache(self.root, base).set_index("open_time")
            for candle in official:
                if not first <= candle.open_time <= last_closed:
                    continue
                old = None
                if candle.open_time in cache.index:
                    row = cache.loc[candle.open_time]
                    old = Candle(candle.open_time, *(float(row[c]) for c in COLUMNS[1:]))
                    if same_candle(old, candle):
                        continue
                store_candle(self.root, base, candle)
                correction = Correction(base, candle.open_time, old, candle)
                corrections.append(correction)
                if self.on_correction is not None:
                    result = self.on_correction(correction)
                    if asyncio.iscoroutine(result):
                        await result
        return corrections

    async def run(self, poll_s: float = 0.5) -> None:
        """Hazırla, WebSocket'i aç, sonsuza kadar işle. Her mum `settle_ms` sonra bir kez uzlaştırılır."""
        await self.prepare()
        await self.start()
        await self.loop(poll_s)

    async def loop(self, poll_s: float = 0.5) -> None:
        """WebSocket açıkken sonsuz döngü: mumları işle, kapanıştan `settle_ms` sonra uzlaştır."""
        try:
            while True:
                await asyncio.sleep(poll_s)
                await self.process()
                now = self.now_ms()
                target = (now - self.settle_ms) // self.tf.millis * self.tf.millis - self.tf.millis
                if target > self._reconciled:
                    self._reconciled = target
                    await self.reconcile(now_ms=now)
        finally:
            await self.stop()


def _console_bar(bar: BarSet) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M", time.gmtime(bar.start_ms / 1000))
    print(
        f"{stamp} UTC  mum {len(bar.candles)}  onarılan {len(bar.repaired)}  "
        f"eksik {len(bar.missing)} {list(bar.missing)[:5] if bar.missing else ''}",
        flush=True,
    )


def _console_correction(correction: Correction) -> None:
    kind = "eksikti" if correction.old is None else "farklıydı"
    print(f"   düzeltme: {correction.base} {correction.start_ms} {kind}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Canlı mum akışı (gölge çalışma: yalnızca izler)")
    parser.add_argument("--tf", default="15m")
    parser.add_argument("--root", default="")
    parser.add_argument("--universe", required=True)
    args = parser.parse_args(argv)
    tf = from_interval(args.tf)
    root = Path(args.root) if args.root else default_root(tf)
    symbols = bases(load_snapshot(Path(args.universe)))
    feed = LiveFeed(root, symbols, tf, _console_bar, _console_correction)
    asyncio.run(feed.run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
