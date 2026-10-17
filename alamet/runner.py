"""
Canlı yürütücü: `LiveFeed` (kapanmış mumlar) → `StepSystem` (sistem) → `Store` (kalıcı depo).

Açılış: geçmiş mum önbelleği tamamlanır; depoda kayıtlı durum varsa yüklenir ve kaçırılan barlar
oynatılır, yoksa çapadan baştan kurulur (ilk kurulum uzun sürer, ara kayıtlarla). Geçmiş oynatma
sırasında yeni mumlar kapanmış olabilir; WebSocket yetişene kadar tekrar eşitlenir ve ancak sonra
açılır. Canlıda her kapanmış sinyal mumu için: adım → dinleyiciler (emir, bildirim) → depoya tek
işlemde yazma. Adım sırasında hata olursa yürütücü DURUR; son kayıt tutarlı kalır.

Bu sürümde emir gönderilmez (gölge çalışma): dinleyiciler olayları görür, hiçbir şey göndermez.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np

from alamet.binance_data import default_root, load_universe_binance
from alamet.engine import TradeParams
from alamet.exchange import default_client
from alamet.livefeed import BarSet, Correction, LiveFeed
from alamet.market import bases, load_snapshot
from alamet.panel import PanelParams
from alamet.stages import with_signal_tf
from alamet.stepper import StepRecord, StepSystem
from alamet.store import Store, fingerprint, replay
from alamet.timeframe import SignalTf, from_interval

PROFILES = {"orijinal": "S17"}
DEFAULT_ANCHOR = "2023-07-01"
FIELDS = ("open", "high", "low", "close", "volume")

Sink = Callable[[list, StepRecord, StepSystem], Awaitable[None] | None]


def profile_params(name: str):
    """Ayar kümesi: profil adı ('orijinal' = Pine varsayılanlarının tamamı) ya da basamak (S0–S50)."""
    from alamet.stages import stage_params  # pylint: disable=import-outside-toplevel

    return stage_params(PROFILES.get(name, name))


def bar_arrays(bar: BarSet, symbols: tuple[str, ...]) -> list[np.ndarray]:
    """Mumu sembol sırasına göre dizilere çevirir; eksik sembol `NaN`."""
    out = {field: np.full(len(symbols), np.nan) for field in FIELDS}
    for k, base in enumerate(symbols):
        candle = bar.candles.get(base)
        if candle is not None:
            for field in FIELDS:
                out[field][k] = getattr(candle, field)
    return [out[field] for field in FIELDS]


class Runner:  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        root: Path,
        symbols: tuple[str, ...],
        tf: SignalTf,
        db_path: Path,
        panel: PanelParams,
        trade: TradeParams,
        anchor_ms: int,
        daily_mode: str = "live",
        sinks: tuple[Sink, ...] = (),
        log: Callable[[str], None] = print,
        feed_options: dict | None = None,
        replay_every: int = 2000,
    ) -> None:
        self.root, self.symbols, self.tf, self.anchor_ms = root, symbols, tf, anchor_ms
        self.panel, self.trade = with_signal_tf(panel, tf.seconds), trade
        self.daily_mode, self.sinks, self.log = daily_mode, sinks, log
        self.replay_every = replay_every
        self.config = fingerprint(
            self.panel, self.trade, symbols, anchor_ms, daily_mode, first_day=None
        )
        self.store = Store(db_path)
        default_client().log = log
        self.system: StepSystem | None = None
        self.corrections = 0
        self.feed = LiveFeed(
            root, symbols, tf, self.on_bar, self.on_correction, log=log, **(feed_options or {})
        )

    # ------------------------------------------------------------- açılış
    def _new_system(self) -> StepSystem:
        return StepSystem(
            len(self.symbols), self.panel, self.trade, self.daily_mode, None, False, True
        )

    def restore_or_create(self) -> StepSystem:
        system, warnings = self.store.restore(self.config)
        for warning in warnings:
            self.log(f"UYARI: {warning}")
        if system is None:
            self.log("Kayıtlı durum yok: çapadan kuruluyor")
            return self._new_system()
        self.log(f"Durum yüklendi: bar {system.index}, son bar {system.prev_time}")
        return system

    def sync(self, system: StepSystem) -> int:
        """Önbellekteki yeni barları (sistemin son barından sonrası) sırayla oynatır."""
        start_ms = self.anchor_ms
        if system.prev_time is not None:
            start_ms = max(start_ms, (system.prev_time + 1) * 1000)
        try:
            universe = load_universe_binance(
                self.root, self.symbols, start_ms, None, True, self.tf, union=True
            )
        except ValueError as exc:
            if "ortak zaman aralığı boş" in str(exc):
                return 0
            raise
        if universe.bars == 0:
            return 0
        started = time.time()

        def progress(done: int, total: int) -> None:
            self.log(f"  oynatma {done}/{total} ({time.time() - started:.0f} sn)")

        replay(system, universe, self.store, list(self.symbols), self.config, self.replay_every,
               0, progress)  # fmt: skip
        return universe.bars

    async def prepare(self) -> StepSystem:
        """Geçmişi tamamla, durumu kur/yükle ve WebSocket açılmadan önce yetişene kadar eşitle."""
        system = self.restore_or_create()
        while True:
            self.log("Geçmiş mumlar tamamlanıyor (REST)")
            await self.feed.prepare()
            replayed = self.sync(system)
            self.log(f"Eşitleme: {replayed} yeni bar işlendi")
            if replayed == 0:
                break
        self.system = system
        return system

    # ------------------------------------------------------------- canlı
    async def on_bar(self, bar: BarSet) -> None:
        system = self.system
        if system is None:
            raise RuntimeError("Sistem hazır değil")
        time_s = bar.start_ms // 1000
        if system.prev_time is not None and time_s <= system.prev_time:
            self.log(f"UYARI: eski/tekrar bar atlandı ({time_s})")
            return
        record = system.step(time_s, *bar_arrays(bar, self.symbols))
        events = system.events.ordered() if system.events is not None else []
        for sink in self.sinks:
            result = sink(events, record, system)
            if asyncio.iscoroutine(result):
                await result
        self.store.commit(system, list(self.symbols), self.config)
        stamp = time.strftime("%Y-%m-%d %H:%M", time.gmtime(time_s))
        self.log(
            f"{stamp} UTC  sembol {len(bar.candles)}/{len(self.symbols)}  "
            f"onarılan {len(bar.repaired)}  eksik {len(bar.missing)}  olay {len(events)}  "
            f"açık {int((record.sides != 0).sum())}"
        )

    def on_correction(self, correction: Correction) -> None:
        self.corrections += 1
        self.store.note(
            "VeriDüzeltmesi",
            correction.start_ms // 1000,
            correction.base,
            {"eski": correction.old.__dict__ if correction.old else None,
             "yeni": correction.new.__dict__},
        )  # fmt: skip

    async def run(self, poll_s: float = 0.5) -> None:
        """Hazırla, WebSocket'i aç, sonsuza kadar işle. Hata olursa durur."""
        try:
            await self.prepare()
            await self.feed.start()
            await self.feed.loop(poll_s)
        finally:
            await self.feed.stop()
            self.store.close()


async def run_until_signal(runner: Runner) -> None:
    """SIGINT/SIGTERM gelince döngüyü iptal eder; `finally` WebSocket'i kapatır, depoyu bırakır."""
    import signal  # pylint: disable=import-outside-toplevel

    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(runner.run())
    for name in ("SIGINT", "SIGTERM"):
        loop.add_signal_handler(getattr(signal, name), task.cancel)
    try:
        await task
    except asyncio.CancelledError:
        runner.log("Durduruldu (sinyal): son kayıt tutarlı.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alamet canlı yürütücü (gölge çalışma: emir yok)")
    parser.add_argument("--universe", required=True)
    parser.add_argument("--tf", default="15m")
    parser.add_argument("--root", default="")
    parser.add_argument("--db", default="data/alamet.sqlite")
    parser.add_argument("--profile", default="orijinal", help="orijinal ya da S0..S50")
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR, help="çapa tarihi (UTC)")
    parser.add_argument("--daily-mode", choices=("live", "history"), default="live")
    args = parser.parse_args(argv)
    from alamet.binance_data import to_ms  # pylint: disable=import-outside-toplevel

    tf = from_interval(args.tf)
    panel, trade = profile_params(args.profile)
    runner = Runner(
        Path(args.root) if args.root else default_root(tf),
        bases(load_snapshot(Path(args.universe))),
        tf,
        Path(args.db),
        panel,
        trade,
        to_ms(args.anchor),
        args.daily_mode,
    )
    asyncio.run(run_until_signal(runner))
    return 0


if __name__ == "__main__":
    sys.exit(main())
