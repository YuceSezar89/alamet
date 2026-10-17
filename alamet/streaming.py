"""
Sembol gösterge zincirinin adım adım (artımlı) karşılığı: A1–A8, `symbol_signals` ile birebir.

Toplu hesap (`symbol.symbol_signals`) bütün seriyi bir kerede işler; canlı bot ise her kapanmış
mumda yalnızca bir adım ilerlemelidir. Burada her gösterge, toplu sürümle AYNI sayısal işlemleri
yapan küçük bir durum makinesidir:
  - pencere işlemleri (SMA, toplam, standart sapma, fiyat bölgesi payı) son pencereye toplu
    fonksiyonların kendisini uygular, bu yüzden sonuç bit düzeyinde aynıdır;
  - özyinelemeli işlemler (RMA/ATR, HA, eşik ve hacim durumları) aynı formülleri aynı sırayla
    uygular;
  - onay makinesi `gate.ConfirmGate` zaten adım adımdır ve olduğu gibi kullanılır.

Eşdeğerlik `tests/test_streaming.py` ile gerçek piyasa verisinde ve rastgele verilerde sınanır.
"""

# pylint: disable=too-many-instance-attributes
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from alamet.gate import ConfirmGate
from alamet.primitives import pine_stdev, pine_sum
from alamet.symbol import SymbolParams
from alamet.volume import MIN_SHARE

NAN = math.nan


def true_range(high: float, low: float, prev_close: float | None) -> float:
    """`pine_tr`: ilk bar high - low, sonrası üç adayın en büyüğü (NaN yayılır)."""
    if prev_close is None:
        return high - low
    return float(
        np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    )


class Rma:
    """`pine_rma`: ilk değer ilk `length` verinin ortalaması, sonra (x + (n-1)·önceki) / n."""

    def __init__(self, length: int) -> None:
        self.length = length
        self._first: list[float] = []
        self.value = NAN
        self._ready = False

    def update(self, x: float) -> float:
        if self._ready:
            self.value = (x + (self.length - 1) * self.value) / self.length
        else:
            self._first.append(x)
            if len(self._first) == self.length:
                self.value = float(np.array(self._first).mean())
                self._ready = True
        return self.value


class Window:
    """Son `length` değer; dolunca batch fonksiyonları son pencereye uygulanır."""

    def __init__(self, length: int) -> None:
        self.length = length
        self._items: deque[float] = deque(maxlen=length)

    def push(self, x: float) -> np.ndarray | None:
        self._items.append(x)
        return np.array(self._items) if len(self._items) == self.length else None


def window_mean(window: np.ndarray | None, length: int) -> float:
    """`pine_sma`: pencere dolmadıysa NaN."""
    if window is None:
        return NAN
    return float(np.lib.stride_tricks.sliding_window_view(window, length).mean(axis=1)[-1])


def window_sum(window: np.ndarray | None, length: int) -> float:
    return NAN if window is None else float(pine_sum(window, length)[-1])


def window_std(window: np.ndarray | None, length: int) -> float:
    return NAN if window is None else float(pine_stdev(window, length)[-1])


class Atr:
    def __init__(self, length: int) -> None:
        self.rma = Rma(length)
        self.prev_close: float | None = None
        self.tr = NAN

    def update(self, high: float, low: float, close: float) -> float:
        self.tr = true_range(high, low, self.prev_close)
        self.prev_close = close
        return self.rma.update(self.tr)


class Threshold:
    """`threshold_state` / `threshold_strength` (TREND ve MOMENTUM)."""

    def __init__(self, thr: float) -> None:
        self.thr = thr
        self.state = 0

    def update(self, z: float) -> tuple[int, int]:
        if z >= self.thr:
            self.state = 1
        elif z <= -self.thr:
            self.state = -1
        elif (self.state == 1 and z < 0) or (self.state == -1 and z > 0):
            self.state = 0
        a = abs(z)
        strength = 2 if a >= self.thr * 3 else (1 if a >= self.thr * 2 else 0)
        return self.state, strength


class Trend:
    def __init__(self, p: SymbolParams) -> None:
        self.p, self.atr = p, Atr(p.atr_len)
        self.window, self.threshold = Window(p.td_len), Threshold(p.td_thr)

    def update(self, high: float, low: float, close: float) -> tuple[int, int]:
        atr = self.atr.update(high, low, close)
        ma = window_mean(self.window.push((high + low + close) / 3.0), self.p.td_len)
        z = (close - ma) / atr if atr > 0 else 0.0
        return self.threshold.update(z)


class Momentum:
    def __init__(self, p: SymbolParams) -> None:
        self.p = p
        self.closes: deque[float] = deque(maxlen=p.mo_len + 1)
        self.window = Window(p.mo_z_len)
        self.threshold = Threshold(p.mo_thr)
        self.prev_close: float | None = None

    def update(self, close: float) -> tuple[int, int]:
        lr1 = NAN if self.prev_close is None else float(np.log(close / self.prev_close))
        self.prev_close = close
        self.closes.append(close)
        lr_n = NAN
        if len(self.closes) == self.p.mo_len + 1:
            lr_n = float(np.log(close / self.closes[0]))
        sd = window_std(self.window.push(lr1), self.p.mo_z_len)
        z = lr_n / (sd * float(np.sqrt(self.p.mo_len))) if sd > 0 else 0.0
        return self.threshold.update(z)


class Volatility:
    def __init__(self, p: SymbolParams) -> None:
        self.p = p
        self.slow, self.fast = Atr(p.va_slow), Atr(p.va_fast)
        self.prev_slow = NAN
        self.state = 0

    def update(self, open_: float, high: float, low: float, close: float) -> tuple[int, int]:
        p = self.p
        atr_slow = self.slow.update(high, low, close)
        atr_fast = self.fast.update(high, low, close)
        tr = self.slow.tr
        x = tr / self.prev_slow if np.nan_to_num(self.prev_slow) > 0 else 1.0
        ratio = atr_fast / atr_slow if atr_slow > 0 else 1.0
        self.prev_slow = atr_slow
        if x >= p.va_thr and close > open_:
            self.state = 1
        elif x >= p.va_thr and close < open_:
            self.state = -1
        elif ratio < p.va_calm:
            self.state = 0
        strength = 2 if x >= p.va_thr * 1.5 else (1 if x >= p.va_thr * 1.2 else 0)
        return self.state, strength


class Volume:
    def __init__(self, p: SymbolParams) -> None:
        self.p = p
        self.atr = Atr(p.atr_len)
        self.w_volume, self.w_delta = Window(p.hv_len), Window(p.hv_len)
        self.w_hlc3 = deque([NAN] * (p.hv_len - 1), maxlen=p.hv_len)
        self.w_pad_volume = deque([0.0] * (p.hv_len - 1), maxlen=p.hv_len)
        self.state = 0

    def _share(self, hlc3: float, close: float, volume: float, atr: float) -> float:
        self.w_hlc3.append(hlc3)
        self.w_pad_volume.append(volume)
        hlc3_w = np.array(self.w_hlc3)[None, :]
        vol_w = np.array(self.w_pad_volume)[None, :]
        with np.errstate(invalid="ignore"):
            inside = np.abs(hlc3_w - close) <= self.p.hv_band * atr
        sum_all = vol_w.sum(axis=1)[0]
        sum_in = (vol_w * inside).sum(axis=1)[0]
        return float(sum_in / sum_all) if sum_all > 0 else 0.0

    def update(self, high: float, low: float, close: float, volume: float) -> tuple[int, int]:
        p = self.p
        atr = self.atr.update(high, low, close)
        rng = high - low
        vol_window = self.w_volume.push(volume)
        vol_sma = window_mean(vol_window, p.hv_len)
        rel_v = volume / vol_sma if vol_sma > 0 else 1.0
        delta = volume * ((close - low) - (high - close)) / rng if rng > 0 else 0.0
        sum_v = window_sum(vol_window, p.hv_len)
        sum_d = window_sum(self.w_delta.push(delta), p.hv_len)
        flow = sum_d / sum_v if sum_v > 0 else 0.0
        share = self._share((high + low + close) / 3.0, close, volume, atr)
        ratio = rel_v / max(share, MIN_SHARE)
        up = ratio >= p.hv_thr and flow >= p.hv_flow
        down = ratio >= p.hv_thr and flow <= -p.hv_flow
        if up:
            self.state = 1
        elif down:
            self.state = -1
        elif (self.state == 1 and flow < 0) or (self.state == -1 and flow > 0):
            self.state = 0
        strength = 2 if ratio >= p.hv_thr * 1.5 else (1 if ratio >= p.hv_thr * 1.2 else 0)
        return self.state, strength


class HaScore:
    """`heikin.ha_score`: HTF mumu sinyal barlarından birleştirilir, HA gücü -100..+100."""

    def __init__(self) -> None:
        self.c_o = self.c_h = self.c_l = self.p_o = self.p_c = NAN
        self.prev_bucket: int | None = None
        self.prev_close = NAN

    def update(self, open_: float, high: float, low: float, close: float, bucket: int) -> float:
        new_bucket = self.prev_bucket is not None and bucket != self.prev_bucket
        if new_bucket or math.isnan(self.c_o):
            if not math.isnan(self.c_o):
                last_c = (self.c_o + self.c_h + self.c_l + self.prev_close) / 4.0
                if math.isnan(self.p_o):
                    last_o = (self.c_o + self.prev_close) / 2.0
                else:
                    last_o = (self.p_o + self.p_c) / 2.0
                self.p_o, self.p_c = last_o, last_c
            self.c_o, self.c_h, self.c_l = open_, high, low
        else:
            self.c_h = max(self.c_h, high)
            self.c_l = min(self.c_l, low)
        ha_c = (self.c_o + self.c_h + self.c_l + close) / 4.0
        ha_o = (self.c_o + close) / 2.0 if math.isnan(self.p_o) else (self.p_o + self.p_c) / 2.0
        rg = max(self.c_h, ha_o, ha_c) - min(self.c_l, ha_o, ha_c)
        self.prev_bucket, self.prev_close = bucket, close
        return (ha_c - ha_o) / rg * 100.0 if rg != 0 else 0.0


def mtf_confirm_step(
    scores: list[float], weights: tuple[float, ...], need: int, tf_on: bool
) -> tuple[float, bool, bool]:
    """`mtf.mtf_confirm`: (ağırlıklı güç, uzun onayı, kısa onayı)."""
    weighted = scores[0] * weights[0]
    total = weights[0]
    for score, weight in zip(scores[1:], weights[1:]):
        weighted = weighted + score * weight
        total = total + weight
    n_bull = sum(int(s > 0) for s in scores)
    n_bear = sum(int(s < 0) for s in scores)
    if tf_on:
        return weighted / total, n_bull >= need, n_bear >= need
    return weighted / total, True, True


@dataclass(frozen=True)
class SymbolStep:
    """Bir barın sonucu: 4 yöntemin kapı çıktısı (tip, başlangıç, güç, bekleyen, bar), HA onay
    gücü ve ATR."""

    methods: tuple[tuple[float, float, float, float, float], ...]
    final_pct: float
    atr: float


class SymbolStream:
    """Tek sembolün A1–A8 zinciri; `update` her kapanmış barda bir kez çağrılır."""

    def __init__(self, params: SymbolParams = SymbolParams()) -> None:
        self.p = params
        self.bar = 0
        self.ha = [HaScore() for _ in params.ha_multipliers]
        self.trend, self.momentum = Trend(params), Momentum(params)
        self.volatility, self.volume = Volatility(params), Volume(params)
        self.atr = Atr(params.atr_len)
        self.gates = [ConfirmGate(params.cool, params.pend_max, params.conf_bars) for _ in range(4)]

    def update(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, open_: float, high: float, low: float, close: float, volume: float, time_s: int
    ) -> SymbolStep:
        p = self.p
        scores = [
            ha.update(open_, high, low, close, int(time_s) // int(p.signal_tf_seconds * mult))
            for ha, mult in zip(self.ha, p.ha_multipliers)
        ]
        final_pct, ok_long, ok_short = mtf_confirm_step(scores, p.ha_weights, p.tf_need, p.tf_on)
        raw = (
            self.trend.update(high, low, close),
            self.momentum.update(close),
            self.volatility.update(open_, high, low, close),
            self.volume.update(high, low, close, volume),
        )
        methods = tuple(
            gate.step(
                self.bar, state == 1, state == -1, ok_long, ok_short, strength, strength, high, low
            )
            for gate, (state, strength) in zip(self.gates, raw)
        )
        atr = self.atr.update(high, low, close)
        self.bar += 1
        return SymbolStep(methods, final_pct, atr)
