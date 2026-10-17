"""
Binance WebSocket akış yöneticisi: stream listesini bağlantılara böler, kopunca yeniden bağlanır.

Her bağlantı kendi asyncio görevidir (thread yok). Bağlantı kurulunca stream'lere SUBSCRIBE edilir;
kopma ya da hata olursa 1 sn'den başlayıp 30 sn'ye çıkan üstel beklemeyle yeniden bağlanır ve
yeniden abone olur. Kopma süresince kaçan mumlar burada telafi edilmez: birleştirici eksik
dakikayı görür, canlı akış REST ile onarır.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable

DEFAULT_BASE = "wss://fstream.binance.com/market"
MAX_STREAMS = 200
PING_INTERVAL_S = 20.0
PING_TIMEOUT_S = 30.0
STAGGER_S = 0.25
FIRST_DELAY_S = 1.0
MAX_DELAY_S = 30.0


def default_connect(url: str, **kwargs):
    import websockets  # pylint: disable=import-outside-toplevel

    return websockets.connect(url, **kwargs)


class Connection:  # pylint: disable=too-many-instance-attributes
    """Tek WebSocket bağlantısı: abone ol, mesajları ilet, kopunca yeniden bağlan."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        number: int,
        streams: list[str],
        url: str,
        on_message: Callable[[str], None],
        connect=default_connect,
        sleep=asyncio.sleep,
    ) -> None:
        self.number, self.streams, self.url = number, streams, url
        self.on_message, self.connect, self.sleep = on_message, connect, sleep
        self.connected = False
        self.last_message_time = 0.0
        self.reconnects = 0
        self._stop = False
        self._task: asyncio.Task | None = None

    def start(self) -> asyncio.Task:
        self._task = asyncio.create_task(self._run(), name=f"alamet_ws_{self.number}")
        return self._task

    async def _run(self) -> None:
        delay = FIRST_DELAY_S
        while not self._stop:
            try:
                async with self.connect(
                    self.url, ping_interval=PING_INTERVAL_S, ping_timeout=PING_TIMEOUT_S
                ) as socket:
                    await socket.send(
                        json.dumps(
                            {"method": "SUBSCRIBE", "params": self.streams, "id": self.number}
                        )
                    )
                    self.connected = True
                    delay = FIRST_DELAY_S
                    async for raw in socket:
                        self.last_message_time = time.time()
                        try:
                            self.on_message(raw)
                        except Exception:  # pylint: disable=broad-exception-caught
                            continue
            except asyncio.CancelledError:
                raise
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            self.connected = False
            if self._stop:
                break
            self.reconnects += 1
            await self.sleep(delay)
            delay = min(delay * 2, MAX_DELAY_S)

    async def stop(self) -> None:
        self._stop = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # pylint: disable=broad-exception-caught
                pass
        self.connected = False


class StreamManager:
    def __init__(
        self,
        on_message: Callable[[str], None],
        base_url: str = DEFAULT_BASE,
        max_streams: int = MAX_STREAMS,
        connect=default_connect,
        sleep=asyncio.sleep,
    ) -> None:
        self.on_message, self.base_url, self.max_streams = on_message, base_url, max_streams
        self.connect, self.sleep = connect, sleep
        self.connections: list[Connection] = []

    async def start(self, streams: list[str]) -> list[Connection]:
        for number, begin in enumerate(range(0, len(streams), self.max_streams), 1):
            connection = Connection(
                number, streams[begin : begin + self.max_streams], f"{self.base_url}/stream",
                self.on_message, self.connect, self.sleep,
            )  # fmt: skip
            connection.start()
            self.connections.append(connection)
            await self.sleep(STAGGER_S)
        return self.connections

    async def stop(self) -> None:
        for connection in self.connections:
            await connection.stop()
        self.connections.clear()

    @property
    def healthy(self) -> bool:
        return bool(self.connections) and all(c.connected for c in self.connections)
