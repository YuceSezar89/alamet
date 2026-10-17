"""
İstatistik ve gösterim verisi (DEPV2 f_acc, f_close istatistikleri, lg* işlem günlüğü, f_live,
f_openDir). Hiçbiri işlem kararına girmez; bot arayüzü ve denetim içindir.

Kaynak Pine: 1217-1320 (durum dizileri), 1617-1626 (`f_acc`), 1694-1770 (`f_close`), 2047-2095
(`f_live`), 1414-1428 (`f_openDir`), işlem günlüğü yazımı f_blockB (≈ 1996-2012).

Sebep kovaları (56): 0-8 çıkış sebebi · 9-10 yön · 11-14 yöntem · 15-17 sinyal gücü · 18-19 hedef
sırası · 20-23 ayrışma no · 24-26 tutma süresi · 27-30 işlem seyri · 31-32 giriş tipi · 33-34 günlük
open yönü · 35 komisyon kaynaklı kayıp · 36-55 emtia.
"""

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-many-locals
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alamet.ranking import SENTINEL, sort_descending

REASON_BUCKETS = 9
FEE_LOSS_BUCKET = 35
SYMBOL_BASE = 36
DEFAULT_SYMBOLS = 20


def cause_buckets(n_symbols: int) -> int:
    """Sebep kovası sayısı: 36 sabit kova + sembol başına bir kova (20 sembolde 56)."""
    return SYMBOL_BASE + n_symbols


CAUSE_BUCKETS = cause_buckets(DEFAULT_SYMBOLS)
DYN_BUCKETS = 12
LOG_SIZE = 8
HOLD_SHORT = 3
HOLD_LONG = 12


@dataclass(frozen=True)
class StatsParams:
    hold_short: int = HOLD_SHORT
    hold_long: int = HOLD_LONG
    log_size: int = LOG_SIZE


class TradeLog:
    """İşlem günlüğü halkası (lg*): son `size` işlem; kapanışta çıkış fiyatı ve durumu yazılır."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.head = 0
        self.symbol = np.full(size, -1, dtype=int)
        self.status = np.zeros(size, dtype=int)
        self.id = np.zeros(size, dtype=int)
        self.side = np.ones(size, dtype=int)
        self.entry, self.risk, self.take, self.exit, self.signal_pct, self.notional = (
            np.full(size, np.nan) for _ in range(6)
        )

    def open(
        self, trade_id, symbol, side, entry, risk, take, signal_pct, notional
    ) -> int:  # pylint: disable=too-many-arguments,too-many-positional-arguments
        slot = self.head % self.size
        self.head += 1
        self.symbol[slot], self.status[slot], self.id[slot], self.side[slot] = (
            symbol, 0, trade_id, side,
        )  # fmt: skip
        self.entry[slot], self.risk[slot], self.take[slot] = entry, risk, take
        self.exit[slot], self.signal_pct[slot], self.notional[slot] = np.nan, signal_pct, notional
        return slot

    def close(self, slot: int, trade_id: int, price: float, status: int) -> None:
        if slot >= 0 and self.id[slot] == trade_id:
            self.exit[slot], self.status[slot] = price, status


class TradeStats:  # pylint: disable=too-many-instance-attributes
    def __init__(self, n_symbols: int, params: StatsParams = StatsParams()) -> None:
        self.p = params
        self.count = np.zeros(2, dtype=int)
        self.wins = np.zeros(2, dtype=int)
        self.sum_r = np.zeros(2)
        self.sum_pnl = np.zeros(2)
        self.sum_usd = np.zeros(2)
        self.fee_loss = np.zeros(2)
        self.symbol_count = np.zeros(n_symbols, dtype=int)
        self.symbol_net = np.zeros(n_symbols)
        size = cause_buckets(n_symbols)
        self.cause_wins = np.zeros(size, dtype=int)
        self.cause_losses = np.zeros(size, dtype=int)
        self.cause_win_usd = np.zeros(size)
        self.cause_loss_usd = np.zeros(size)
        self.cause_r = np.zeros(size)
        self.cause_mfe = np.zeros(size)
        self.cause_hold = np.zeros(size)
        self.id_seq = 0
        self.log = TradeLog(params.log_size)

    def _add(self, bucket: int, usd: float, r: float, mfe: float, hold: int) -> None:
        if usd > 0:
            self.cause_wins[bucket] += 1
            self.cause_win_usd[bucket] += usd
        else:
            self.cause_losses[bucket] += 1
            self.cause_loss_usd[bucket] -= usd
        self.cause_r[bucket] += r
        self.cause_mfe[bucket] += mfe
        self.cause_hold[bucket] += hold

    def hold_bucket(self, hold: int, base: int) -> int:
        """Tutma süresi kovası (`base` = 24 sebep kovaları, 7 dinamik kasa)."""
        if hold <= self.p.hold_short:
            return base
        return base + 1 if hold <= self.p.hold_long else base + 2

    def record(
        self, trade, usd: float, pnl: float, reason: int, fee_loss: bool
    ) -> None:  # pylint: disable=too-many-locals
        """`f_close` istatistikleri: yön toplamları, sembol toplamları ve 10-11 sebep kovası."""
        di = 0 if trade.side == 1 else 1
        r, mfe, hold, symbol = trade.r, trade.mfe, trade.hold, trade.symbol
        self.count[di] += 1
        self.sum_r[di] += r
        self.sum_pnl[di] += pnl
        self.sum_usd[di] += usd
        if usd > 0:
            self.wins[di] += 1
        elif fee_loss:
            self.fee_loss[0] += 1.0
            self.fee_loss[1] -= usd
        self.symbol_count[symbol] += 1
        self.symbol_net[symbol] += usd
        self._add(max(0, min(REASON_BUCKETS - 1, reason)), usd, r, mfe, hold)
        div = trade.div_n
        div_bucket = 20 if div <= 2 else (21 if div == 3 else (22 if div <= 5 else 23))
        if usd > 0:
            course = 27 if (mfe > 0 and r >= mfe * 0.7) else 28
        else:
            course = 29 if mfe < 0.25 else 30
        buckets = (
            9 + di, 11 + trade.method, 15 + trade.strength % 3, 18 if trade.rank == 1 else 19,
            div_bucket, self.hold_bucket(hold, 24), course, 32 if trade.strength >= 3 else 31,
            34 if trade.op_side == -1 else 33, SYMBOL_BASE + symbol,
        )  # fmt: skip
        for bucket in buckets:
            self._add(bucket, usd, r, mfe, hold)
        if fee_loss:
            self._add(FEE_LOSS_BUCKET, usd, r, mfe, hold)


def live_rows(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    top: np.ndarray, active: np.ndarray, count: np.ndarray, cur_move: np.ndarray,
    scores: np.ndarray, types: np.ndarray, close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """`f_live`: (hedefteki ayrışma sayısı, canlı hareket %, LONG sırası, SHORT sırası)."""
    n = len(top)
    n_live = count.astype(int).copy()
    move = np.full(n, np.nan)
    for k in range(n):
        if top[k] and not math.isnan(close[k]):
            if active[k]:
                move[k] = cur_move[k]
            else:
                n_live[k] += 1
                move[k] = 0.0
    keys = {1.0: np.full(n, SENTINEL), -1.0: np.full(n, SENTINEL)}
    for k in range(n):
        if not (math.isnan(scores[k]) or math.isnan(types[k])) and types[k] in keys:
            keys[types[k]][k] = scores[k] + (1e6 if n_live[k] > 0 else 0.0)
    long_order = sort_descending(keys[1.0][None, :], ties="pine")[0]
    short_order = sort_descending(keys[-1.0][None, :], ties="pine")[0]
    return n_live, move, long_order, short_order


def stats_layout(
    log_size: int = LOG_SIZE, n_symbols: int = DEFAULT_SYMBOLS
) -> list[tuple[str, int]]:
    """`stats_vector` düzeni: (ad, uzunluk) çiftleri; doğrulama çıktısında değerleri adlandırır."""
    cause = ("lsW", "lsL", "lsWu", "lsLu", "lsR", "lsM", "lsH")
    return [
        ("ks", 10), ("tCnt", 2), ("tWin", 2), ("tSumR", 2), ("tSumPnl", 2), ("tSumUsd", 2),
        ("xc", 2), ("symN", n_symbols), ("symNet", n_symbols),
        *[(name, cause_buckets(n_symbols)) for name in cause], ("bNe", DYN_BUCKETS), ("bSr", DYN_BUCKETS),
        ("lgHead", 1), *[(name, log_size) for name in (
            "lgSym", "lgSt", "lgId", "lgDir", "lgEnt", "lgRu", "lgTp", "lgExit", "lgDv", "lgNot",
        )],
        ("nLive", n_symbols), ("lvMv", n_symbols), ("tIdxL", n_symbols), ("tIdxS", n_symbols),
        ("openLong", 4), ("openShort", 4),
    ]  # fmt: skip


def vector_names(log_size: int = LOG_SIZE, n_symbols: int = DEFAULT_SYMBOLS) -> list[str]:
    return [f"{name}[{i}]" for name, size in stats_layout(log_size, n_symbols) for i in range(size)]


def log_vector(log: TradeLog) -> np.ndarray:
    return np.concatenate(
        [
            [float(log.head)], log.symbol, log.status, log.id, log.side, log.entry, log.risk,
            log.take, log.exit, log.signal_pct, log.notional,
        ]  # fmt: skip
    ).astype(float)


def stats_vector(engine, live: tuple, open_long: np.ndarray, open_short: np.ndarray) -> np.ndarray:
    """Tüm istatistik dizilerinin tek, sabit sıralı vektörü (TV doğrulamasıyla aynı düzen)."""
    s, st = engine.s, engine.stats
    parts = [
        s.ks, st.count, st.wins, st.sum_r, st.sum_pnl, st.sum_usd, st.fee_loss,
        st.symbol_count, st.symbol_net, st.cause_wins, st.cause_losses, st.cause_win_usd,
        st.cause_loss_usd, st.cause_r, st.cause_mfe, st.cause_hold, s.bne, s.bsr,
        log_vector(st.log), *live, open_long, open_short,
    ]  # fmt: skip
    return np.concatenate([np.asarray(p, dtype=float).ravel() for p in parts])
