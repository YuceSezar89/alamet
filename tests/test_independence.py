"""alamet yeni, bağımsız bir proje: eski sistemin hiçbir modülünü içe aktaramaz."""

import ast
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "alamet"
THIRD_PARTY = {"numpy", "pandas", "aiohttp", "websockets", "pyarrow", "yarl"}
OLD_SYSTEM = {
    "binance_client", "config", "database", "dashboard", "desktop", "indicators",
    "live_data_manager", "signals", "utils", "backtest", "research", "run_services",
    "signal_service",
}  # fmt: skip


def imported_roots(path: Path) -> set[str]:
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_imports_only_itself_the_standard_library_and_declared_packages():
    allowed = set(sys.stdlib_module_names) | THIRD_PARTY | {"alamet", "__future__"}
    offenders = {}
    for path in sorted(PACKAGE.glob("*.py")):
        extra = imported_roots(path) - allowed
        if extra:
            offenders[path.name] = sorted(extra)
    assert not offenders, offenders


def test_no_module_name_of_the_old_system_is_imported_anywhere_in_alamet():
    found = {path.name: sorted(imported_roots(path) & OLD_SYSTEM) for path in PACKAGE.glob("*.py")}
    assert not {name: roots for name, roots in found.items() if roots}


def test_tests_do_not_import_the_old_system_either():
    tests = Path(__file__).resolve().parent
    offenders = {}
    for path in sorted(tests.glob("test_*.py")):
        hits = imported_roots(path) & OLD_SYSTEM
        if hits:
            offenders[path.name] = sorted(hits)
    assert not offenders, offenders
