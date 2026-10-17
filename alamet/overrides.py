"""
Komut satırından ayar geçersiz kılma: `panel.n_top=3`, `trade.quality.max_ext=6`, `trade.use_tp=true`.

Yol, `PanelParams` / `TradeParams` içindeki kayıt alanlarını izler (iç içe kayıtlar nokta ile,
`panel.own.mult` gibi sözlük anahtarları da). Değer, mevcut değerin türüne çevrilir. Motor düzeyindeki
basit alanlar (`trade.use_tp`, `trade.risk_pct`...) `stage_params`'a iletilir; gölge kasa bunlarla
yeniden türetilir. İç içe alanlar (`trade.quality.*`, `trade.kelly.*`, `trade.pool.*`) sonradan uygulanır.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

from alamet.panel import PanelParams
from alamet.stages import stage_params, with_signal_tf


def coerce(raw: str, current):
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "evet", "açık", "on")
    if isinstance(current, int):
        return int(float(raw))
    if isinstance(current, float):
        return float(raw)
    if current is None:
        text = raw.strip().lower()
        if text in ("true", "false"):
            return text == "true"
        try:
            return int(text) if text.lstrip("-").isdigit() else float(text)
        except ValueError:
            return raw
    return raw


def _set(obj, parts: list[str], raw: str):
    head, rest = parts[0], parts[1:]
    if isinstance(obj, dict):
        current = obj.get(head)
        value = coerce(raw, current) if not rest else _set(current, rest, raw)
        return {**obj, head: value}
    if not is_dataclass(obj) or head not in {f.name for f in fields(obj)}:
        raise KeyError(f"bilinmeyen alan: {head}")
    current = getattr(obj, head)
    value = coerce(raw, current) if not rest else _set(current, rest, raw)
    return replace(obj, **{head: value})


def params_with(stage: str, sets: list[str], tf_seconds: int | None = None):
    """(panel, trade): basamak parametreleri + `--set` geçersiz kılmaları."""
    simple: dict[str, str] = {}
    later: list[tuple[str, list[str], str]] = []
    for item in sets:
        path, _, raw = item.partition("=")
        root, *parts = path.split(".")
        if root not in ("panel", "trade") or not parts:
            raise ValueError(f"geçersiz --set: {item} (panel.* ya da trade.* olmalı)")
        if root == "trade" and len(parts) == 1:
            simple[parts[0]] = raw
        else:
            later.append((root, parts, raw))
    panel, trade = stage_params(stage)
    if simple:
        typed = {name: coerce(raw, getattr(trade, name)) for name, raw in simple.items()}
        panel, trade = stage_params(stage, **typed)
    for root, parts, raw in later:
        if root == "panel":
            panel = _set(panel, parts, raw)
        else:
            trade = _set(trade, parts, raw)
    if tf_seconds is not None:
        panel = with_signal_tf(panel, tf_seconds)
    return panel, trade


def panel_only(sets: list[str]) -> PanelParams:
    """Yalnız panel geçersiz kılmalarıyla varsayılan panel (sinyal basamakları için)."""
    panel = PanelParams()
    for item in sets:
        path, _, raw = item.partition("=")
        root, *parts = path.split(".")
        if root == "panel" and parts:
            panel = _set(panel, parts, raw)
    return panel
