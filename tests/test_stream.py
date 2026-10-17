"""WebSocket akış yöneticisi: abonelik, bölme, yeniden bağlanma, hata yalıtımı, durdurma."""

import asyncio
import json

from alamet.stream import Connection, StreamManager


class FakeSocket:
    def __init__(self, messages, hang=True):
        self.messages, self.hang, self.sent = list(messages), hang, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.messages:
            return self.messages.pop(0)
        if self.hang:
            await asyncio.Event().wait()
        raise ConnectionError("koptu")


def connector(*plan):
    """plan: her bağlanma denemesi için bir FakeSocket ya da fırlatılacak hata."""
    attempts = []
    queue = list(plan)

    def connect(url, **kwargs):
        attempts.append((url, kwargs))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return connect, attempts


async def settle(predicate, limit=200):
    for _ in range(limit):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("beklenen durum oluşmadı")


def test_connection_subscribes_and_forwards_messages():
    async def scenario():
        socket = FakeSocket(["a", "b"])
        connect, attempts = connector(socket)
        got = []
        connection = Connection(1, ["x@kline_1m"], "wss://h/stream", got.append, connect)
        connection.start()
        await settle(lambda: got == ["a", "b"])
        await connection.stop()
        return socket, attempts, got, connection

    socket, attempts, got, connection = asyncio.run(scenario())
    assert socket.sent == [{"method": "SUBSCRIBE", "params": ["x@kline_1m"], "id": 1}]
    assert attempts[0][0] == "wss://h/stream" and not connection.connected


def test_connection_reconnects_with_growing_delays_and_resubscribes():
    async def scenario():
        second = FakeSocket(["ok"])
        connect, attempts = connector(OSError("a"), OSError("b"), second)
        delays, got = [], []

        async def sleep(seconds):
            delays.append(seconds)
            await asyncio.sleep(0)

        connection = Connection(2, ["s"], "u", got.append, connect, sleep)
        connection.start()
        await settle(lambda: got == ["ok"])
        await connection.stop()
        return delays, second, connection

    delays, second, connection = asyncio.run(scenario())
    assert delays[:2] == [1.0, 2.0] and second.sent[0]["method"] == "SUBSCRIBE"
    assert connection.reconnects == 2


def test_a_dropped_socket_triggers_a_new_connection():
    async def scenario():
        first, second = FakeSocket(["one"], hang=False), FakeSocket(["two"])
        connect, attempts = connector(first, second)
        got = []

        async def sleep(_seconds):
            await asyncio.sleep(0)

        connection = Connection(1, ["s"], "u", got.append, connect, sleep)
        connection.start()
        await settle(lambda: got == ["one", "two"])
        await connection.stop()
        return len(attempts)

    assert asyncio.run(scenario()) == 2


def test_a_failing_handler_does_not_kill_the_connection():
    async def scenario():
        connect, _ = connector(FakeSocket(["bad", "good"]))
        got = []

        def handler(raw):
            if raw == "bad":
                raise ValueError("kötü mesaj")
            got.append(raw)

        connection = Connection(1, ["s"], "u", handler, connect)
        connection.start()
        await settle(lambda: got == ["good"])
        await connection.stop()
        return got

    assert asyncio.run(scenario()) == ["good"]


def test_manager_splits_streams_across_connections_and_reports_health():
    async def scenario():
        sockets = [FakeSocket([]), FakeSocket([]), FakeSocket([])]
        connect, attempts = connector(*sockets)

        async def sleep(_seconds):
            await asyncio.sleep(0)

        manager = StreamManager(lambda raw: None, "wss://h/market", 2, connect, sleep)
        streams = [f"s{i}@kline_1m" for i in range(5)]
        await manager.start(streams)
        await settle(lambda: manager.healthy)
        sizes = [len(c.streams) for c in manager.connections]
        await manager.stop()
        return sizes, [s.sent[0]["params"] for s in sockets], attempts, manager.healthy

    sizes, params, attempts, healthy = asyncio.run(scenario())
    assert sizes == [2, 2, 1] and params[2] == ["s4@kline_1m"] and not healthy
    assert attempts[0][0] == "wss://h/market/stream"
