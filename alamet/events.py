"""
Yapılandırılmış olaylar: DEPV2'nin 7 `alert()` çağrısının sayısal karşılığı.

Kaynak Pine: AÇ (2045), KAPAT (1769 `f_close`), AYRIŞMA (1847), ÇİFT AYRIŞMA (1359 `f_upd`),
SERT DD ACİL ÇIKIŞ (1918), GÜNLÜK KAYIP LİMİTİ (1926), KASA KÂR KİLİDİ (1943).

Olaylar karar mantığına girmez; motor `events` boşken (varsayılan) hiçbir şey üretmez ve hesaplamaz.
Değerler ham sayıdır (yuvarlama ve metin `render` içindedir). Bar içi sıra Pine'daki sırayı izler:
açık işlemleri yönet → ayrışma → korumalar → girişler (`phase`); aynı evrede ekleme sırası (`seq`).

`render` Pine metinlerini yeniden üretir. Sayı biçimleri (`#.##`, `math.round`) Pine
belgelerinden ve elle yazılmış örneklerden türetildi; TradingView alarm günlüğüyle henüz
karşılaştırılmadı.
"""

# pylint: disable=too-many-instance-attributes
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

PHASE_EXIT, PHASE_DIVERGENCE, PHASE_PROTECT, PHASE_ENTRY = 0, 1, 2, 3

METHOD_NAMES = ("TREND", "MOMENTUM", "VOLATİLİTE", "HACİM")
STATUS_TEXT = {1: "TP/KİLİT", 2: "SIRALAMA", 4: "STOP/ZAMAN/DD"}
STATUS_SIGNAL_TEXT = "TERS SİNYAL"


@dataclass(frozen=True, kw_only=True)
class Event:
    bar: int
    phase: int
    seq: int = 0


@dataclass(frozen=True, kw_only=True)
class OpenEvent(Event):
    """AÇ: giriş ve büyüklük zincirinin bütün parçaları."""

    symbol: int
    side: int
    trade_id: int
    method: int
    strength: int
    rank: int
    div_n: int
    entry: float
    take: float
    notional: float
    risk_pct: float
    multiplier: float
    dyn_mult: float
    kelly_mult: float
    dr_mult: float
    rank_mult: float
    dyn_on: bool
    dyn_score: float
    dd_factor: float
    streak_factor: float
    below_start_cap: bool
    kelly_on: bool
    kelly_mean: float
    kelly_effective_n: float
    kelly_exploring: bool
    dr_on: bool
    dr_ratio: float
    dr_percentile: float


@dataclass(frozen=True, kw_only=True)
class CloseEvent(Event):
    """KAPAT: `status` Pine `st` (1 TP/KİLİT · 2 SIRALAMA · 3 TERS SİNYAL · 4 STOP/ZAMAN/DD),
    `reason` motorun ayrıntılı sebebi (0-8)."""

    symbol: int
    side: int
    trade_id: int
    status: int
    reason: int
    entry: float
    exit: float
    r: float
    pnl_pct: float
    usd: float
    hold: int


@dataclass(frozen=True, kw_only=True)
class DivergenceEvent(Event):
    """AYRIŞMA: ikinci ve sonraki ayrışmanın başlangıcı (`direction` 1 LONG, -1 SHORT)."""

    symbol: int
    number: int
    direction: int


@dataclass(frozen=True, kw_only=True)
class DoubleDivergenceEvent(Event):
    """ÇİFT AYRIŞMA: `move` yön işaretiyle çarpılmış hareket %."""

    symbol: int
    number: int
    move: float


@dataclass(frozen=True, kw_only=True)
class HardDrawdownEvent(Event):
    drawdown_pct: float
    closed: int
    wait_bars: int


@dataclass(frozen=True, kw_only=True)
class DayLossEvent(Event):
    loss_pct: float
    closed: int
    closes_trades: bool


@dataclass(frozen=True, kw_only=True)
class EquityLockEvent(Event):
    peak_profit: float
    closed: int
    wait_bars: int


class EventLog:
    """Olay deposu; `ordered()` Pine'daki bar içi sırayı verir."""

    def __init__(self) -> None:
        self._items: list[Event] = []
        self._next = 0
        self.times: np.ndarray | None = None

    def add(self, kind: type[Event], bar: int, phase: int, **fields) -> Event:
        event = kind(bar=bar, phase=phase, seq=self._next, **fields)
        self._next += 1
        self._items.append(event)
        return event

    def drain(self) -> list[Event]:
        """Biriken olayları sırayla verir ve listeyi boşaltır (sıra sayacı sıfırlanmaz)."""
        items = self.ordered()
        self._items = []
        return items

    def ordered(self) -> list[Event]:
        return sorted(self._items, key=lambda e: (e.bar, e.phase, e.seq))

    def of(self, kind: type[Event]) -> list[Event]:
        return [e for e in self.ordered() if isinstance(e, kind)]

    def time_of(self, event: Event) -> int:
        if self.times is None:
            raise ValueError("EventLog.times tanımlı değil")
        return int(self.times[event.bar])

    def __len__(self) -> int:
        return len(self._items)


def _round(x: float, digits: int) -> float:
    scale = 10.0**digits
    return math.floor(x * scale + 0.5) / scale


def _num(x: float) -> str:
    """Pine `str.tostring(x)` varsayılanı: en çok 10 ondalık, sondaki sıfırlar atılır."""
    text = f"{x:.10f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _fixed(x: float, digits: int) -> str:
    """Pine `str.tostring(x, "#.##")`: en çok `digits` ondalık, sondaki sıfırlar atılır."""
    text = f"{_round(x, digits):.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _signed(x: float, text: str) -> str:
    return ("+" if x >= 0 else "") + text


def fmt_move(x: float) -> str:
    """Pine `f_mv`."""
    return "-" if math.isnan(x) else _signed(x, _num(_round(x, 2))) + "%"


def fmt_price(x: float) -> str:
    """Pine `f_px`."""
    if math.isnan(x):
        return "-"
    digits = 2 if x >= 100 else (4 if x >= 1 else 8)
    return _fixed(x, digits)


def fmt_r(x: float) -> str:
    """Pine `f_rr`."""
    return "-" if math.isnan(x) else _signed(x, _fixed(x, 2)) + "R"


def strength_text(code: int) -> str:
    """Pine `f_stN`: 0-2 güç, +3 ikinci sinyal."""
    base = {2: "🚀 Çok güçlü", 1: "⚡ Strong"}.get(code % 3, "● Normal")
    return base + (" ②" if code >= 3 else "")


def _open_reason(e: OpenEvent) -> str:
    if e.dyn_on:
        text = ("DD kısıtı " if e.dd_factor < 1.0 else "") + (
            "seri kısıtı " if e.streak_factor < 1.0 else ""
        )
        text += "skor " + ("+" if e.dyn_score >= 0 else "") + _num(_round(e.dyn_score, 2)) + "R"
    else:
        text = "normal"
    if e.kelly_on:
        if e.kelly_exploring:
            why = "keşif (n≈" + _num(_round(e.kelly_effective_n, 1)) + ")"
        else:
            sign = "+" if e.kelly_mean >= 0 else ""
            why = (
                "E[R] " + sign + _num(_round(e.kelly_mean, 2)) + "R · n≈"
                + _num(_round(e.kelly_effective_n, 1))
            )  # fmt: skip
        text += " · MOTOR x" + _num(_round(e.kelly_mult, 2)) + " " + why
    if e.dr_on:
        ratio = 0.0 if math.isnan(e.dr_ratio) else e.dr_ratio
        pct = "-" if math.isnan(e.dr_percentile) else _num(_round(e.dr_percentile, 0))
        text += (
            " · DR x" + _num(_round(e.dr_mult, 2)) + " (ratio " + _num(_round(ratio, 3))
            + " · yüzdelik " + pct + ")"
        )  # fmt: skip
    return text


def render(event: Event, names: Sequence[str], mode: str = "") -> str:
    """Olayı Pine `alert()` metnine çevirir. `names`: `f_nm` karşılığı (ör. "BTCUSDT")."""
    if isinstance(event, OpenEvent):
        return (
            f"AÇ | {names[event.symbol]} | {'LONG' if event.side == 1 else 'SHORT'} "
            f"({mode} · {METHOD_NAMES[event.method]}) {strength_text(event.strength)} | "
            f"Hedef #{event.rank} | Giriş {fmt_price(event.entry)} | TP {fmt_price(event.take)} | "
            f"Büyüklük {_num(_round(event.notional, 2))} USDT | "
            f"1R risk %{_num(_round(event.risk_pct, 2))} kasa | "
            f"Dinamik x{_num(_round(event.multiplier, 2))} ({_open_reason(event)})"
        )
    if isinstance(event, CloseEvent):
        status = STATUS_TEXT.get(event.status, STATUS_SIGNAL_TEXT)
        return (
            f"KAPAT | {names[event.symbol]} | {'LONG' if event.side == 1 else 'SHORT'} | {status} | "
            f"R {fmt_r(event.r)} | PNL {fmt_move(event.pnl_pct)} | "
            f"K/Z {_num(_round(event.usd, 2))} USDT"
        )
    if isinstance(event, DivergenceEvent):
        arrow = "▲L" if event.direction == 1 else "▼S"
        return f"{names[event.symbol]} · {event.number}. AYRIŞMA {arrow}"
    if isinstance(event, DoubleDivergenceEvent):
        return (
            f"{names[event.symbol]} · {event.number}. AYRIŞMA ★ ÇİFT (kendi içinde de) "
            f"{fmt_move(event.move)}"
        )
    if isinstance(event, HardDrawdownEvent):
        return (
            f"⛔ SERT DD ACİL ÇIKIŞ | zirveden -{_num(_round(event.drawdown_pct, 1))}% | "
            f"{event.closed} işlem kapatıldı | {event.wait_bars} SS barı bekleme"
        )
    if isinstance(event, DayLossEvent):
        tail = f" | {event.closed} işlem kapatıldı" if event.closes_trades else ""
        return (
            f"⛔ GÜNLÜK KAYIP LİMİTİ | -{_num(_round(event.loss_pct, 1))}% | "
            f"gün sonuna kadar yeni işlem yok{tail}"
        )
    if isinstance(event, EquityLockEvent):
        return (
            f"🔒 KASA KÂR KİLİDİ | zirve kâr {_num(_round(event.peak_profit, 1))} USDT | "
            f"{event.closed} işlem kapatıldı | {event.wait_bars} SS barı bekleme"
        )
    raise TypeError(f"bilinmeyen olay: {type(event).__name__}")
