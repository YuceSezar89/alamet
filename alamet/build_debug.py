"""
Orijinal DEPV2'nin debug kopyalarını üretir (kural kodu değişmez):
  - girdi varsayılanları basamağa göre ayarlanır (stages.py),
  - 20 sembol BINANCE:…USDT.P olur, calc_bars_count verilen değere ayarlanır (bellek sınırı),
  - özgün 20 plot silinir (64 plot sınırı), yerine doğrulama plotları eklenir.

Kullanım: python -m alamet.build_debug /yol/DEPV2.txt pine/debug [calc_bars]
`DEBUG_ALL.pine`: tüm S basamakları tek dosyada, `stg` listesiyle seçilir (CSV'de `STG`).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from alamet.stages import SIGNAL_STAGES, STAGES
from alamet.universe import SYMBOLS

N = 20


def literal(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return repr(float(value)) if isinstance(value, float) else str(value)


def set_default(text: str, name: str, value) -> str:
    pattern = re.compile(rf"^({re.escape(name)}\s*=\s*input\.\w+\()([^,]+)(,)", re.MULTILINE)
    new, count = pattern.subn(lambda m: m.group(1) + literal(value) + m.group(3), text)
    if count != 1:
        raise ValueError(f"{name}: {count} eşleşme")
    return new


def base_text(source: str, title: str, calc_bars: int) -> str:
    text = source.replace("calc_bars_count = 4000", f"calc_bars_count = {calc_bars}")
    text = text.replace(
        'indicator("Devis\'So Otomatik Bot Eğitim PaneliV2"', f'indicator("{title}"', 1
    )
    for number, symbol in enumerate(SYMBOLS, start=1):
        pattern = re.compile(rf'^(t{number}\s*=\s*input\.symbol\()"[^"]*"', re.MULTILINE)
        text, count = pattern.subn(lambda m, s=symbol: f'{m.group(1)}"BINANCE:{s}USDT.P"', text)
        if count != 1:
            raise ValueError(f"t{number}: {count} eşleşme")
    return "\n".join(line for line in text.split("\n") if not line.startswith("plot("))


def indexed(expression: str, title: str) -> list[str]:
    return [
        f'plot({expression.format(k=k)}, "{title}{k + 1:02d}")' for k in range(N)
    ]  # fmt: skip


def trade_plots() -> str:
    lines = indexed("trAct.get({k}) ? 1.0 * trDir.get({k}) : 0.0", "SD")
    lines += indexed("trAct.get({k}) ? trEnt.get({k}) : na", "EN")
    lines += indexed("trAct.get({k}) ? trNot.get({k}) : 0.0", "NO")
    lines += ['plot(ks.get(0), "KS0")', 'plot(ks.get(1), "KS1")', 'plot(ks.get(2), "KS2")']
    lines += ['plot(1.0 * (tCnt.get(0) + tCnt.get(1)), "CL")']
    return "\n".join(lines)


def signal_plots() -> str:
    lines = indexed("types.get({k})", "TY")
    lines += indexed("scR.get({k})", "SC")
    lines += indexed("1.0 * tRank.get({k})", "TR")
    count = [
        "int dbgTop = 0",
        "int dbgEl = 0",
        "int dbgOk = 0",
        "for k = 0 to 19",
        "    dbgTop += topArr.get(k) ? 1 : 0",
        "    dbgEl += eligA.get(k) ? 1 : 0",
        "    dbgOk += ownOkA.get(k) ? 1 : 0",
    ]
    tail = ['plot(1.0 * dbgTop, "NTOP")', 'plot(1.0 * dbgEl, "NEL")', 'plot(1.0 * dbgOk, "NOK")']
    return "\n".join(count + lines + tail)


def day_plots() -> str:
    lines = [
        f'plot(mDay.get({k}, {c}), "MD{k + 1:02d}_{c:02d}")' for k in range(3) for c in range(20)
    ]
    lines += [f'plot(drXs.get({k}), "XS{k + 1:02d}")' for k in range(3)]
    return "\n".join(lines)


def build(source: str, name: str, overrides: dict, plots: str, calc_bars: int) -> str:
    text = base_text(source, f"DEPV2 DEBUG {name}", calc_bars)
    for key, value in overrides.items():
        text = set_default(text, key, value)
    return text.rstrip("\n") + "\n\n// ---- DEBUG PLOTLARI ----\n" + plots + "\n"


def _statement_end(lines: list[str], start: int) -> int:
    """`start` satırında başlayan ifadenin (girintili devam satırları dahil) sonraki satırı."""
    end = start + 1
    while end < len(lines) and lines[end].startswith((" ", "\t")):
        end += 1
    return end


DISPLAY_MARKER = "CANLI SIRA VERİSİ"
TABLE_MARKER = "İŞLEMLER TABLOSU"
STATS_PER_SLICE = 62


def _without_display(lines: list[str], marker: str = DISPLAY_MARKER) -> list[str]:
    """Canlı etiket ve tablo çizimlerini (`f_peak` çağrısından sonrası) atar: karara girmez,
    Pine'ın derlenmiş belirteç sınırı (100.256) için gerekli."""
    at = next((i for i, line in enumerate(lines) if marker in line), None)
    if at is None:
        return lines
    return lines[: max(at - 1, 0)]


def _ranges(indices: list[int]) -> str:
    """Basamak indekslerini `stgIdx` aralık koşullarına sıkıştırır."""
    parts, start, prev = [], indices[0], indices[0]
    for index in [*indices[1:], None]:
        if index is not None and index == prev + 1:
            prev = index
            continue
        parts.append(
            f"stgIdx == {start}" if start == prev else f"stgIdx >= {start} and stgIdx <= {prev}"
        )
        if index is not None:
            start = prev = index
    return " or ".join(parts)


def _cases(per_stage: list[str]) -> tuple[list[tuple[str, str]], str]:
    """Basamak başına değer metninden (koşul, değer) çiftleri ve en yaygın değer (son kol)."""
    counts: dict[str, int] = {}
    for text in per_stage:
        counts[text] = counts.get(text, 0) + 1
    fallback = max(counts, key=lambda text: counts[text])
    cases = []
    for text in dict.fromkeys(per_stage):
        if text != fallback:
            cases.append((_ranges([i for i, value in enumerate(per_stage) if value == text]), text))
    return cases, fallback


def _effective(name: str, per_stage: list[str]) -> str:
    """Girdi `name_in`; basamağa göre etkin değer `name` (basamaklar STAGES sırasındadır)."""
    cases, expr = _cases(per_stage)
    for cond, text in reversed(cases):
        expr = f"({cond}) ? {text} : ({expr})"
    return f"{name} = {expr}"


def stats_code() -> str:
    """İstatistik dizilerini `stats.stats_vector` ile aynı sırada tek vektöre dizer ve çizer."""
    lines = [
        "f_pushI(array<float> v, array<int> a) =>",
        "    for i = 0 to a.size() - 1",
        "        v.push(1.0 * a.get(i))",
        "    0",
        "f_pushF(array<float> v, array<float> a, int n) =>",
        "    for i = 0 to n - 1",
        "        v.push(a.get(i))",
        "    0",
        'statSlice = input.int(0, "İstatistik dilimi", minval = 0, maxval = 10)',
        "array<float> sv = array.new<float>()",
        "f_pushF(sv, ks, 10)",
        "f_pushI(sv, tCnt)",
        "f_pushI(sv, tWin)",
        "f_pushF(sv, tSumR, 2)",
        "f_pushF(sv, tSumPnl, 2)",
        "f_pushF(sv, tSumUsd, 2)",
        "f_pushF(sv, xc, 2)",
        "f_pushI(sv, symN)",
        "f_pushF(sv, symNet, 20)",
        "f_pushI(sv, lsW)",
        "f_pushI(sv, lsL)",
        "f_pushF(sv, lsWu, 56)",
        "f_pushF(sv, lsLu, 56)",
        "f_pushF(sv, lsR, 56)",
        "f_pushF(sv, lsM, 56)",
        "f_pushF(sv, lsH, 56)",
        "f_pushF(sv, bNe, 12)",
        "f_pushF(sv, bSr, 12)",
        "f_pushI(sv, lgHead)",
        "f_pushI(sv, lgSym)",
        "f_pushI(sv, lgSt)",
        "f_pushI(sv, lgId)",
        "f_pushI(sv, lgDir)",
        "f_pushF(sv, lgEnt, logN)",
        "f_pushF(sv, lgRu, logN)",
        "f_pushF(sv, lgTp, logN)",
        "f_pushF(sv, lgExit, logN)",
        "f_pushF(sv, lgDv, logN)",
        "f_pushF(sv, lgNot, logN)",
        "f_pushI(sv, nLive)",
        "f_pushF(sv, lvMv, 20)",
        "f_pushI(sv, tIdxL)",
        "f_pushI(sv, tIdxS)",
        "[oLn, oLu, oLr, oLp] = f_openDir(1)",
        "[oSn, oSu, oSr, oSp] = f_openDir(-1)",
        "sv.push(1.0 * oLn)",
        "sv.push(oLu)",
        "sv.push(oLr)",
        "sv.push(oLp)",
        "sv.push(1.0 * oSn)",
        "sv.push(oSu)",
        "sv.push(oSr)",
        "sv.push(oSp)",
    ]
    for i in range(STATS_PER_SLICE):
        index = f"statSlice * {STATS_PER_SLICE} + {i}"
        lines.append(f'plot({index} < sv.size() ? sv.get({index}) : na, "V{i:02d}")')
    lines += ['plot(1.0 * stgIdx, "STG")', 'plot(1.0 * statSlice, "SLC")']
    return "\n".join(lines)


def build_master(  # pylint: disable=too-many-locals
    source: str, calc_bars: int = 4000, stats: bool = False
) -> str:
    """Tüm S basamakları tek dosyada: `stg` listesinden seçilenin girdileri etkin olur."""
    text = base_text(source, "DEPV2 DEBUG ALL", calc_bars)
    stage_names = list(STAGES)
    overrides: dict[str, list[str]] = {}
    for key in sorted({key for stage in STAGES.values() for key in stage.pine}):
        overrides[key] = [
            literal(stage.pine[key]) if key in stage.pine else f"{key}_in"
            for stage in STAGES.values()
        ]
    lines = _without_display(text.split("\n"), TABLE_MARKER if stats else DISPLAY_MARKER)
    inserts: list[tuple[int, str]] = []
    for name, per_stage in overrides.items():
        pattern = re.compile(rf"^{re.escape(name)}(\s*=\s*input\.\w+\()")
        found = [i for i, line in enumerate(lines) if pattern.match(line)]
        if len(found) != 1:
            raise ValueError(f"{name}: {len(found)} eşleşme")
        i = found[0]
        lines[i] = pattern.sub(rf"{name}_in\1", lines[i], count=1)
        inserts.append((_statement_end(lines, i), _effective(name, per_stage)))
    first = next(i for i, line in enumerate(lines) if re.match(r"^t1\s*=\s*input\.symbol", line))
    options = ", ".join(f'"{name}"' for name in stage_names)
    index = str(len(stage_names) - 1)
    for k in range(len(stage_names) - 2, -1, -1):
        index = f'stg == "{stage_names[k]}" ? {k} : ({index})'
    header = [
        f'string stg = input.string("S9", "Doğrulama basamağı", options = [{options}])',
        f"int stgIdx = {index}",
    ]
    inserts.append((first, "\n".join(header)))
    for position, line in sorted(inserts, key=lambda item: item[0], reverse=True):
        lines.insert(position, line)
    if stats:
        body = "\n".join(lines).rstrip("\n") + "\n\n// ---- İSTATİSTİK VEKTÖRÜ ----\n"
        return body + stats_code() + "\n"
    plots = trade_plots().replace('plot(ks.get(1), "KS1")', 'plot(1.0 * stgIdx, "STG")')
    return "\n".join(lines).rstrip("\n") + "\n\n// ---- DEBUG PLOTLARI ----\n" + plots + "\n"


def main(source_path: str, out_dir: str, calc_bars: int = 4000) -> None:
    source = Path(source_path).read_text(encoding="utf-8")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, (pine, _, _) in SIGNAL_STAGES.items():
        (out / f"DEBUG_{name}.pine").write_text(
            build(source, name, pine, signal_plots(), calc_bars)
        )
    (out / "DEBUG_DAY.pine").write_text(
        build(source, "DAY", SIGNAL_STAGES["SIG"][0], day_plots(), calc_bars)
    )
    for name, stage in STAGES.items():
        (out / f"DEBUG_{name}.pine").write_text(
            build(source, name, stage.pine, trade_plots(), calc_bars)
        )
    (out / "DEBUG_ALL.pine").write_text(build_master(source, calc_bars))
    (out / "DEBUG_STATS.pine").write_text(build_master(source, calc_bars, stats=True))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4000)
