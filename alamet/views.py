"""
Görünüm tabloları: Pine'daki 5 tablonun sayısal karşılığı (DataFrame ve sözlük).

Kaynak Pine: İŞLEMLER (`f_tradeTbl` 2100), KASA (`f_drawKs` 2167), KAYIP/KAZANÇ SEBEP ANALİZİ
(`f_drawLs` 2214-2312), KASA MOTORU (`f_drawEng` 2321), YÖNTEM × EMTİA (`f_drawMt` 2376).

Tablolar `barstate.islast` anındaki, yani koşunun son barındaki durumu gösterir. Değerler ham
sayıdır: metin, yuvarlama, renk ve 🔥/🏆 gibi simgeler arayüzün işidir. Pine'da "-" yazılan boş
değerler `NaN`'dır. Kaynak, TradingView'de doğrulanmış istatistik vektörünü üreten aynı motor
durumudur (`stats.py`); tablolar yalnızca bunlardan türetilir.
"""

# pylint: disable=too-many-locals,too-many-instance-attributes
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alamet.engine import TradeEngine, TradeParams
from alamet.events import METHOD_NAMES
from alamet.kelly import DIRECTION_BASE, ease_base
from alamet.primitives import ge, le
from alamet.shadow import ACCOUNTS, ShadowParams
from alamet.shadow import trade_usd as shadow_usd
from alamet.stats import SYMBOL_BASE
from alamet.universe import SYMBOLS

EXIT_LABELS = (
    "TP hedefi", "Kâr kilidi (işlem içi geri verme)", "Ters SİNYAL (seçili mod, TF onaylı)",
    "SIRALAMA değişimi", "Felaket STOP", "ZAMAN stopu", "SERT DD acil çıkış",
    "Günlük limit çıkışı", "Kasa kâr kilidi (NS19)",
)  # fmt: skip
DIRECTION_LABELS = ("▲ LONG", "▼ SHORT")
STRENGTH_LABELS = ("● Normal güç", "⚡ Strong güç", "🚀 Çok güçlü")
RANK_LABELS = ("Hedef #1", "Hedef #2+")
DIVERGENCE_LABELS = ("Ayrışma no ≤ 2", "Ayrışma no 3", "Ayrışma no 4–5", "Ayrışma no 6+")
COURSE_LABELS = (
    "Kazanç: zirveye yakın çıkış", "Kazanç: zirveden geri verdi", "Kayıp: hiç lehine gitmedi",
    "Kayıp: lehine gitti sonra zarar",
)  # fmt: skip
ENTRY_LABELS = ("Giriş: direkt TF onaylı", "Giriş: ② 2. sinyal (onay sonradan)")
OPEN_LABELS = ("Günlük open: DOĞRU taraf", "Günlük open: YANLIŞ taraf")
FEE_LABELS = ("Fiyat lehine ama komisyonla zarar",)
EASE_LABELS = (
    "DR ÇOK KOLAY (yüzdelik ≤ 25)", "DR KOLAY (25–50)", "DR ORTA (50–75)", "DR ZOR (> 75)",
)  # fmt: skip
ACCOUNT_LABELS = (*METHOD_NAMES, "SEÇİLİ MOD")
NAN = math.nan


@dataclass(frozen=True)
class ViewContext:
    """Tabloların ihtiyaç duyduğu her şey: motor, son bar kapanışları, parametreler, gölge kasa."""

    engine: TradeEngine
    close: np.ndarray
    names: Sequence[str]
    trade: TradeParams
    shadow: dict | None = None
    shadow_params: ShadowParams = ShadowParams()
    enabled: tuple[bool, ...] = (True, True, True, True)
    auto_mode: bool = False
    fallback_method: int = 3

    @property
    def n_symbols(self) -> int:
        return self.engine.n


def view_context(result, panel_params, trade_params: TradeParams, names=None) -> ViewContext:
    """`run_system` sonucundan görünüm bağlamı (motor durumu ve son bar kapanışları)."""
    if result.engine is None:
        raise ValueError("SystemResult.engine yok")
    selector = panel_params.selector
    return ViewContext(
        result.engine,
        np.asarray(result.panel.universe.close[-1], dtype=float),
        list(names) if names is not None else [f"{s}USDT" for s in SYMBOLS],
        trade_params,
        result.panel.shadow_last,
        panel_params.shadow,
        tuple(selector.enabled),
        selector.mode == 6,
        selector.fallback_method,
    )


def _profit_factor(win: float, loss: float) -> float:
    """Pine `f_pf`: kayıp yoksa kazanç varsa sonsuz, yoksa NaN."""
    if loss > 0:
        return win / loss
    return math.inf if win > 0 else NAN


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else NAN


# ---------------------------------------------------------------- 1. İŞLEMLER


@dataclass(frozen=True)
class TradesView:
    rows: pd.DataFrame
    totals: dict[str, float]
    header: dict[str, float | bool]


def _open_totals(ctx: ViewContext) -> dict[str, float]:
    long_ = ctx.engine.open_direction(ctx.close, 1)
    short = ctx.engine.open_direction(ctx.close, -1)
    return {
        "open_long": long_[0], "open_short": short[0], "open_count": long_[0] + short[0],
        "open_usd_long": long_[1], "open_usd_short": short[1], "open_usd": long_[1] + short[1],
        "open_r": long_[2] + short[2], "open_pnl_pct": long_[3] + short[3],
    }  # fmt: skip


def trades_view(ctx: ViewContext) -> TradesView:
    """Son `log_size` işlem, yeniden eskiye. Açık işlemde `price` anlık kapanıştır."""
    engine, p = ctx.engine, ctx.trade
    log, stats = engine.stats.log, engine.stats
    rows = []
    for back in range(log.size):
        seq = log.head - 1 - back
        if seq < 0:
            continue
        slot = seq % log.size
        symbol, status, side = int(log.symbol[slot]), int(log.status[slot]), int(log.side[slot])
        entry, risk = log.entry[slot], log.risk[slot]
        price = ctx.close[symbol] if status == 0 else log.exit[slot]
        known = not math.isnan(price)
        rows.append(
            {
                "trade_id": int(log.id[slot]), "symbol": symbol, "side": side, "entry": entry,
                "take": log.take[slot], "price": price, "status": status,
                "r": engine.trade_r(side, entry, risk, price) if known else NAN,
                "pnl_pct": engine.trade_pnl(side, entry, price) if known else NAN,
                "usd": engine.trade_usd(log.notional[slot], side, entry, price) if known else NAN,
                "signal_pct": log.signal_pct[slot] * side,
            }
        )  # fmt: skip
    columns = ["trade_id", "symbol", "side", "entry", "take", "price", "status", "r", "pnl_pct",
               "usd", "signal_pct"]  # fmt: skip
    frame = pd.DataFrame(rows, columns=columns)
    closed_usd = float(stats.sum_usd.sum())
    totals = {
        "closed_count": float(stats.count.sum()),
        "closed_r": float(stats.sum_r.sum()),
        "closed_pnl_pct": float(stats.sum_pnl.sum()),
        "closed_usd": closed_usd,
        "closed_usd_long": float(stats.sum_usd[0]),
        "closed_usd_short": float(stats.sum_usd[1]),
        **_open_totals(ctx),
    }
    totals["total_usd"] = closed_usd + totals["open_usd"]
    totals["total_pct"] = totals["total_usd"] / p.start_cap * 100.0
    header = {"use_tp": p.use_tp, "tp_r": p.tp_r, "use_stop": p.use_stop, "stop_r": p.stop_r,
              "r_unit": p.r_unit}  # fmt: skip
    return TradesView(frame, totals, header)


# ------------------------------------------------------------------- 2. KASA


def cash_view(ctx: ViewContext) -> dict[str, float]:
    engine, p = ctx.engine, ctx.trade
    s, stats = engine.s, engine.stats
    open_ = _open_totals(ctx)
    realised = float(s.ks[0])
    open_usd = open_["open_usd"]
    equity = realised + open_usd
    net = equity - p.start_cap
    trades = int(stats.count.sum())
    wins = int(stats.wins.sum())
    gross_profit, gross_loss = float(s.ks[3]), float(s.ks[4])
    peak = max(float(s.ks[1]), equity)
    drawdown = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
    return {
        "start_cap": p.start_cap, "equity": equity, "net_usd": net,
        "net_pct": net / p.start_cap * 100.0, "realised": realised, "open_usd": open_usd,
        "peak": peak, "drawdown_now": drawdown, "drawdown_max": max(float(s.ks[2]), drawdown),
        "wins": wins, "losses": trades - wins, "trades": trades,
        "win_pct": wins / trades * 100.0 if trades > 0 else NAN,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else NAN,
        "long_usd": float(stats.sum_usd[0]) + open_["open_usd_long"],
        "short_usd": float(stats.sum_usd[1]) + open_["open_usd_short"],
        "best_usd": float(s.ks[6]), "worst_usd": float(s.ks[7]),
        "max_loss_streak": float(s.ks[9]),
    }  # fmt: skip


# --------------------------------------------------------- 3. SEBEP ANALİZİ


def hold_labels(hold_short: int, hold_long: int) -> tuple[str, str, str]:
    return (
        f"Tutma ≤ {hold_short} bar", f"Tutma {hold_short + 1}–{hold_long} bar",
        f"Tutma > {hold_long} bar",
    )  # fmt: skip


def reason_sections(ctx: ViewContext) -> list[tuple[int, str, Sequence[str], int]]:
    """(bölüm no, başlık, etiketler, ilk kova)."""
    p = ctx.trade
    return [
        (1, "ÇIKIŞ SEBEBİ", EXIT_LABELS, 0),
        (2, "YÖN", DIRECTION_LABELS, 9),
        (3, "GİRİŞİ AÇAN YÖNTEM", METHOD_NAMES, 11),
        (4, "SİNYAL GÜCÜ", STRENGTH_LABELS, 15),
        (5, "HEDEF SIRASI", RANK_LABELS, 18),
        (6, "AYRIŞMA NO", DIVERGENCE_LABELS, 20),
        (7, "TUTMA SÜRESİ", hold_labels(p.hold_short, p.hold_long), 24),
        (8, "İŞLEM SEYRİ", COURSE_LABELS, 27),
        (9, "GİRİŞ TİPİ", ENTRY_LABELS, 31),
        (10, "GÜNLÜK OPEN YÖNÜ", OPEN_LABELS, 33),
        (11, "KOMİSYON KAYNAKLI KAYIP", FEE_LABELS, 35),
        (12, "EMTİA", [f"{k + 1}-{name}" for k, name in enumerate(ctx.names)], SYMBOL_BASE),
    ]


@dataclass(frozen=True)
class ReasonView:
    summary: dict[str, float]
    rows: pd.DataFrame


def reason_view(ctx: ViewContext) -> ReasonView:
    """Kayıp/kazanç sebep analizi. `is_top_loss` / `is_top_win` bölümün 🔥 / 🏆 satırıdır."""
    engine = ctx.engine
    stats, s = engine.stats, engine.s
    gross_profit, gross_loss = float(s.ks[3]), float(s.ks[4])
    trades = int(stats.count.sum())
    wins = int(stats.wins.sum())
    losses = trades - wins
    net = gross_profit - gross_loss
    avg_win = gross_profit / wins if wins > 0 else NAN
    avg_loss = gross_loss / losses if losses > 0 else NAN
    summary = {
        "closed": trades, "gross_profit": gross_profit, "wins": wins, "gross_loss": gross_loss,
        "losses": losses, "net_usd": net,
        "win_rate": wins * 100.0 / trades if trades > 0 else NAN,
        "profit_factor": _profit_factor(gross_profit, gross_loss),
        "avg_win": avg_win, "avg_loss": avg_loss,
        "payoff": avg_win / avg_loss if avg_loss > 0 else NAN,
        "expectancy": net / trades if trades > 0 else NAN,
        "cost": float(s.ks[5]), "fee_loss_count": float(stats.fee_loss[0]),
        "fee_loss_usd": float(stats.fee_loss[1]),
    }  # fmt: skip
    rows = []
    for number, title, labels, base in reason_sections(ctx):
        top_loss = top_win = -1
        max_loss = max_win = 0.0
        for i in range(len(labels)):
            if stats.cause_loss_usd[base + i] > max_loss:
                max_loss, top_loss = stats.cause_loss_usd[base + i], i
            if stats.cause_win_usd[base + i] > max_win:
                max_win, top_win = stats.cause_win_usd[base + i], i
        for i, label in enumerate(labels):
            b = base + i
            won, lost = int(stats.cause_wins[b]), int(stats.cause_losses[b])
            n = won + lost
            win_usd, loss_usd = float(stats.cause_win_usd[b]), float(stats.cause_loss_usd[b])
            rows.append(
                {
                    "section": number, "title": title, "bucket": b, "label": label, "trades": n,
                    "share_pct": n * 100.0 / trades if trades > 0 else NAN,
                    "wins": won, "win_usd": win_usd,
                    "win_usd_share": win_usd / gross_profit * 100.0
                    if win_usd > 0 and gross_profit > 0 else NAN,
                    "losses": lost, "loss_usd": loss_usd,
                    "loss_usd_share": loss_usd / gross_loss * 100.0
                    if loss_usd > 0 and gross_loss > 0 else NAN,
                    "net_usd": win_usd - loss_usd if n > 0 else NAN,
                    "win_pct": won * 100.0 / n if n > 0 else NAN,
                    "avg_r": stats.cause_r[b] / n if n > 0 else NAN,
                    "avg_mfe": stats.cause_mfe[b] / n if n > 0 else NAN,
                    "avg_hold": stats.cause_hold[b] / n if n > 0 else NAN,
                    "is_top_loss": i == top_loss, "is_top_win": i == top_win,
                }
            )  # fmt: skip
    return ReasonView(summary, pd.DataFrame(rows))


# ---------------------------------------------------------- 4. KASA MOTORU


def engine_sections(ctx: ViewContext) -> list[tuple[str, Sequence[str], int]]:
    return [
        ("YÖN", DIRECTION_LABELS, DIRECTION_BASE),
        ("YÖNTEM", METHOD_NAMES, 11),
        ("SİNYAL GÜCÜ", STRENGTH_LABELS, 15),
        ("HEDEF SIRASI", RANK_LABELS, 18),
        ("AYRIŞMA NO", DIVERGENCE_LABELS, 20),
        ("GİRİŞ TİPİ", ENTRY_LABELS, 31),
        ("GÜNLÜK OPEN YÖNÜ", OPEN_LABELS, 33),
        ("DEVİSSO KOLAYLIK DİLİMİ", EASE_LABELS, ease_base(ctx.n_symbols)),
        ("EMTİA", [f"{k + 1}-{name}" for k, name in enumerate(ctx.names)], SYMBOL_BASE),
    ]


@dataclass(frozen=True)
class EngineView:
    header: dict[str, float | bool]
    rows: pd.DataFrame


def engine_view(ctx: ViewContext) -> EngineView:
    """Kova başına n, ham R, küçültülmüş E[R] ve bu kova tek başına olsaydı Kelly çarpanı."""
    engine, p = ctx.engine, ctx.trade
    kelly, kp = engine.kelly, p.kelly
    sd = kelly.deviation()
    equity = float(engine.s.ks[0]) + engine.unrealised(ctx.close)
    header = {
        "on": kp.on, "kelly": kp.kelly, "sd": sd,
        "learned": float(kelly.n[DIRECTION_BASE] + kelly.n[DIRECTION_BASE + 1]),
        "equity": equity, "risk_pct": p.risk_pct, "cap_on": p.cap_on, "cap_pct": p.cap_pct,
        "below_start": equity < p.start_cap and p.lock_on, "neg_mult": p.neg_mult,
    }  # fmt: skip
    rows = []
    for title, labels, base in engine_sections(ctx):
        for i, label in enumerate(labels):
            b = base + i
            n, total = float(kelly.n[b]), float(kelly.total[b])
            mu = total / (n + kp.prior)
            safe = mu - kp.z * sd / math.sqrt(max(n, 1.0))
            raw_kelly = kp.kelly * (safe / (sd * sd)) / (p.risk_pct / 100.0) if safe > 0 else 0.0
            known = n > 0
            rows.append(
                {
                    "section": title, "bucket": b, "label": label, "n": n,
                    "raw_r": total / n if known else NAN, "expected_r": mu if known else NAN,
                    "kelly_x": min(kp.max_mult, max(kp.min_mult, raw_kelly)) if known else NAN,
                    "blocked": bool(known and le(mu, -kp.block) and ge(n, kp.min_n)),
                    "boosted": bool(known and raw_kelly > 1.0),
                }
            )  # fmt: skip
    return EngineView(header, pd.DataFrame(rows))


# ------------------------------------------------------ 5. YÖNTEM × EMTİA


@dataclass(frozen=True)
class MethodSymbolView:
    cells: pd.DataFrame
    best: pd.DataFrame
    engine: pd.DataFrame
    totals: pd.DataFrame
    real_total: float


def method_symbol_view(ctx: ViewContext) -> MethodSymbolView:
    """Gölge kasa: her emtia × (4 yöntem + seçili mod), net K/Z açık sanal pozisyon dahil."""
    if ctx.shadow is None:
        raise ValueError("Gölge kasa durumu yok (SignalPanel.shadow_last)")
    sh, n = ctx.shadow, ctx.n_symbols
    cells, best_rows, engine_rows = [], [], []
    tot_net, tot_count, tot_wins = (
        np.zeros(ACCOUNTS),
        np.zeros(ACCOUNTS, int),
        np.zeros(ACCOUNTS, int),
    )
    real_total = 0.0
    for k in range(n):
        price = ctx.close[k]
        net = np.zeros(ACCOUNTS)
        best_value, best_method = -1e10, -1
        for a in range(ACCOUNTS):
            side = int(sh["side"][k, a])
            net[a] = sh["net"][k, a]
            if side != 0 and not math.isnan(price):
                net[a] += shadow_usd(
                    ctx.shadow_params, sh["notional"][k, a], side, sh["entry"][k, a], price
                )
            count = int(sh["count"][k, a])
            enabled = a == 4 or ctx.enabled[a]
            if a < 4 and enabled and (count > 0 or side != 0) and net[a] > best_value:
                best_value, best_method = net[a], a
        for a in range(ACCOUNTS):
            count, wins, side = int(sh["count"][k, a]), int(sh["wins"][k, a]), int(sh["side"][k, a])
            cells.append(
                {
                    "symbol": k, "name": ctx.names[k], "account": a, "label": ACCOUNT_LABELS[a],
                    "net_usd": net[a], "trades": count, "wins": wins,
                    "win_pct": wins * 100.0 / count if count > 0 else NAN, "open_side": side,
                    "has_data": count > 0 or side != 0, "enabled": a == 4 or ctx.enabled[a],
                    "is_best": a == best_method,
                }
            )  # fmt: skip
            tot_net[a] += net[a]
            tot_count[a] += count
            tot_wins[a] += wins
        auto = int(sh["auto"][k]) if ctx.auto_mode else None
        best_rows.append(
            {
                "symbol": k, "name": ctx.names[k], "best_method": best_method,
                "best_usd": best_value if best_method >= 0 else NAN,
                "auto_method": (auto if auto is not None and auto >= 0 else ctx.fallback_method)
                if ctx.auto_mode else -1,
                "auto_is_fallback": bool(ctx.auto_mode and auto is not None and auto < 0),
            }
        )  # fmt: skip
        engine = ctx.engine
        open_ = bool(engine.s.active[k]) and not math.isnan(price)
        real = float(engine.stats.symbol_net[k])
        if open_:
            real += engine.trade_usd(engine.s.notional[k], int(engine.s.side[k]),
                                     engine.s.entry[k], price)  # fmt: skip
        real_total += real
        engine_rows.append(
            {"symbol": k, "name": ctx.names[k], "net_usd": real,
             "trades": int(engine.stats.symbol_count[k]), "open": open_,
             "has_data": int(engine.stats.symbol_count[k]) > 0 or open_}  # fmt: skip
        )
    best_total, best_account = -1e10, -1
    for a in range(4):
        if ctx.enabled[a] and tot_net[a] > best_total and tot_count[a] > 0:
            best_total, best_account = tot_net[a], a
    totals = pd.DataFrame(
        {
            "account": range(ACCOUNTS), "label": ACCOUNT_LABELS, "net_usd": tot_net,
            "trades": tot_count, "wins": tot_wins,
            "win_pct": [w * 100.0 / c if c > 0 else NAN for w, c in zip(tot_wins, tot_count)],
            "is_best": [a == best_account for a in range(ACCOUNTS)],
        }
    )  # fmt: skip
    return MethodSymbolView(
        pd.DataFrame(cells), pd.DataFrame(best_rows), pd.DataFrame(engine_rows), totals, real_total
    )
