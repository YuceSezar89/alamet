"""
A8 — Ortak onay makinesi f_gate: ham durum → BEKLEYEN → TF onayı → TETİK (direkt veya 2. sinyal).

Kaynak Pine: pine/A8_gate.pine (DEPV2 f_gate 411-470). Her yöntem kendi örneğini kullanır.
`bar` Pine'daki bar_index'tir (verinin ilk satırı 0). Güç kodu: 0-2 temel güç, +3 = 2. sinyal
(TF onayı sinyal barından sonra geldi).

ConfirmGate.step tek bir barı işler (canlı akış için), run_gate bütün diziyi çalıştırır.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VP_COOL = 10
VP_PEND_MAX = 12
VP_CONF_BARS = 1
NO_SIGNAL_BAR = -100000.0


@dataclass(frozen=True)
class GateResult:
    last_type: np.ndarray
    last_start: np.ndarray
    last_str: np.ndarray
    pend_dir: np.ndarray
    last_sig: np.ndarray


class ConfirmGate:
    def __init__(
        self,
        cool: int = VP_COOL,
        pend_max: int = VP_PEND_MAX,
        conf_bars: int = VP_CONF_BARS,
    ) -> None:
        self.cool = cool
        self.pend_max = pend_max
        self.conf_bars = conf_bars
        self.conf_dir = 0
        self.last_bar = 0
        self.pend_dir = 0
        self.pend_bar = 0
        self.pend_str = 0
        self.conf_cnt = 0
        self.last_sig = NO_SIGNAL_BAR
        self.last_start = float("nan")
        self.last_type = 0.0
        self.last_str = 0

    def _expire_pending(self, bar: int, raw_long: bool, raw_short: bool) -> None:
        if self.pend_dir == 1 and (not raw_long or bar - self.pend_bar > self.pend_max):
            self.pend_dir = 0
            self.conf_cnt = 0
        if self.pend_dir == -1 and (not raw_short or bar - self.pend_bar > self.pend_max):
            self.pend_dir = 0
            self.conf_cnt = 0

    def _open_pending(self, bar: int, direction: int, strength: int) -> None:
        self.pend_dir = direction
        self.pend_bar = bar
        self.pend_str = strength
        self.conf_cnt = 0
        self.last_bar = bar

    def _confirm(self, bar: int, tf_ok: bool, direction: int) -> int | None:
        self.conf_cnt = self.conf_cnt + 1 if tf_ok else 0
        if self.conf_cnt < self.conf_bars:
            return None
        strength = self.pend_str + (3 if bar > self.pend_bar else 0)
        self.conf_dir = direction
        self.pend_dir = 0
        self.conf_cnt = 0
        return strength

    def step(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        bar: int,
        raw_long: bool,
        raw_short: bool,
        tf_long: bool,
        tf_short: bool,
        str_long: int,
        str_short: int,
        high: float,
        low: float,
    ) -> tuple[float, float, float, float, float]:
        self._expire_pending(bar, raw_long, raw_short)
        cool_ok = (bar - self.last_bar) >= self.cool
        if raw_long and self.conf_dir != 1 and self.pend_dir != 1 and cool_ok:
            self._open_pending(bar, 1, str_long)
        if raw_short and self.conf_dir != -1 and self.pend_dir != -1 and cool_ok:
            self._open_pending(bar, -1, str_short)
        signal_long = signal_short = None
        if self.pend_dir == 1:
            signal_long = self._confirm(bar, tf_long, 1)
        elif self.pend_dir == -1:
            signal_short = self._confirm(bar, tf_short, -1)
        if signal_long is not None:
            self.last_start, self.last_type = low, 1.0
            self.last_str, self.last_sig = signal_long, float(bar)
        if signal_short is not None:
            self.last_start, self.last_type = high, -1.0
            self.last_str, self.last_sig = signal_short, float(bar)
        return (
            self.last_type,
            self.last_start,
            float(self.last_str),
            float(self.pend_dir),
            self.last_sig,
        )


def run_gate(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    raw_long: np.ndarray,
    raw_short: np.ndarray,
    tf_long: np.ndarray,
    tf_short: np.ndarray,
    strength: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    cool: int = VP_COOL,
    pend_max: int = VP_PEND_MAX,
    conf_bars: int = VP_CONF_BARS,
) -> GateResult:
    n = len(raw_long)
    gate = ConfirmGate(cool, pend_max, conf_bars)
    out = np.empty((n, 5))
    for i in range(n):
        out[i] = gate.step(
            i,
            bool(raw_long[i]),
            bool(raw_short[i]),
            bool(tf_long[i]),
            bool(tf_short[i]),
            int(strength[i]),
            int(strength[i]),
            float(high[i]),
            float(low[i]),
        )
    return GateResult(
        last_type=out[:, 0],
        last_start=out[:, 1],
        last_str=out[:, 2],
        pend_dir=out[:, 3],
        last_sig=out[:, 4],
    )
