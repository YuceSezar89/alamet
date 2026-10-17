"""A5 — HACİM yöntemi: bağımsız döngü referansı, durum makinesi ve Pine na/nz davranışı."""

import math

import numpy as np

from alamet.primitives import pine_atr, pine_sum
from alamet.volume import volume_signal, volume_state, volume_strength


def _bars(n, seed):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    open_ = close - rng.normal(0, 0.5, n)
    spread = rng.uniform(0.2, 2.0, n)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(50.0, 500.0, n)
    return high, low, close, volume


def _naive(high, low, close, volume, length, band):
    n = len(close)
    atr = pine_atr(high, low, close, 10)
    hlc3 = [(high[i] + low[i] + close[i]) / 3.0 for i in range(n)]
    delta = [
        (
            volume[i] * ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            if high[i] - low[i] > 0
            else 0.0
        )
        for i in range(n)
    ]
    rel, flow, share, ratio = [], [], [], []
    for t in range(n):
        if t >= length - 1:
            win = range(t - length + 1, t + 1)
            sma = sum(volume[j] for j in win) / length
            sum_v = sum(volume[j] for j in win)
            sum_d = sum(delta[j] for j in win)
            rel.append(volume[t] / sma if sma > 0 else 1.0)
            flow.append(sum_d / sum_v if sum_v > 0 else 0.0)
        else:
            rel.append(1.0)
            flow.append(0.0)
        s_in = s_all = 0.0
        for i in range(length):
            j = t - i
            vi = volume[j] if j >= 0 else 0.0
            s_all += vi
            if j >= 0 and not math.isnan(atr[t]) and abs(hlc3[j] - close[t]) <= band * atr[t]:
                s_in += vi
        share.append(s_in / s_all if s_all > 0 else 0.0)
        ratio.append(rel[-1] / max(share[-1], 0.1))
    return rel, flow, share, ratio


def test_sum_window():
    out = pine_sum(np.array([1.0, 2.0, 3.0, 4.0]), 3)
    assert np.isnan(out[:2]).all()
    np.testing.assert_allclose(out[2:], [6.0, 9.0])


def test_matches_naive_loop_including_partial_windows():
    high, low, close, volume = _bars(150, 21)
    res = volume_signal(high, low, close, volume)
    rel, flow, share, ratio = _naive(high, low, close, volume, 12, 0.5)
    np.testing.assert_allclose(res.rel_v, rel, atol=1e-9)
    np.testing.assert_allclose(res.flow, flow, atol=1e-9)
    np.testing.assert_allclose(res.share, share, atol=1e-9)
    np.testing.assert_allclose(res.ratio, ratio, atol=1e-9)


def test_warmup_defaults():
    high, low, close, volume = _bars(60, 4)
    res = volume_signal(high, low, close, volume)
    assert (res.rel_v[:11] == 1.0).all()
    assert (res.flow[:11] == 0.0).all()
    assert (res.share[:9] == 0.0).all()
    assert np.allclose(res.ratio[:9], 1.0 / 0.1)


def test_state_machine_sequence():
    up = np.array([True, False, False, False, False])
    down = np.array([False, False, False, True, False])
    flow = np.array([0.2, 0.1, -0.1, -0.2, 0.05])
    assert volume_state(up, down, flow).tolist() == [1, 1, 0, -1, 0]


def test_state_ignores_nan_flow():
    up = np.array([True, False])
    down = np.array([False, False])
    assert volume_state(up, down, np.array([0.2, np.nan])).tolist() == [1, 1]


def test_strength_thresholds():
    ratio = np.array([1.0, 2.5, 3.0, 3.5, np.nan])
    assert volume_strength(ratio, 2.0).tolist() == [0, 1, 2, 2, 0]


def test_zero_volume_never_triggers():
    n = 40
    base = np.linspace(100.0, 110.0, n)
    res = volume_signal(base + 1.0, base - 1.0, base, np.zeros(n))
    assert (res.state == 0).all()
    assert (res.flow == 0.0).all()


def test_buying_pressure_spike_gives_long_state():
    n = 60
    base = np.full(n, 100.0)
    high, low, close = base + 1.0, base - 1.0, base.copy()
    volume = np.full(n, 100.0)
    close[50], high[50], low[50], volume[50] = 100.9, 101.0, 99.0, 1500.0
    res = volume_signal(high, low, close, volume)
    assert res.state[50] == 1
    assert res.strength[50] == 2
