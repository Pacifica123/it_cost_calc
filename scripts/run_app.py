from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

_CATALOG_PARSER_SWITCH = "--catalog-parser"
_PLAYWRIGHT_INSTALL_SWITCH = "--playwright-install"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IT Cost Calc desktop UI.")
    parser.add_argument(
        "--smoke-check",
        action="store_true",
        help="Create Qt objects without showing the window.",
    )
    return parser.parse_args(argv)


def _run_catalog_parser(argv: list[str]) -> int:
    from tools.catalog_parser.cli import main as catalog_main

    return int(catalog_main(argv))


def _install_playwright_browser(engine: str) -> int:
    if engine not in {"firefox", "chromium"}:
        print(f"Unsupported Playwright engine: {engine}", file=sys.stderr)
        return 2
    try:
        from playwright._impl._driver import compute_driver_executable, get_driver_env
    except ModuleNotFoundError:
        print("Playwright is not bundled with this release.", file=sys.stderr)
        return 2

    driver_executable, driver_cli = compute_driver_executable()
    completed = subprocess.run(
        [driver_executable, driver_cli, "install", engine],
        env=get_driver_env(),
        check=False,
    )
    return int(completed.returncode)


def _dispatch_internal_command(argv: list[str]) -> int | None:
    if argv and argv[0] == _CATALOG_PARSER_SWITCH:
        return _run_catalog_parser(argv[1:])
    if argv and argv[0] == _PLAYWRIGHT_INSTALL_SWITCH:
        if len(argv) != 2:
            print("Usage: ITCostCalc --playwright-install <firefox|chromium>", file=sys.stderr)
            return 2
        return _install_playwright_browser(argv[1])
    return None


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    internal_result = _dispatch_internal_command(raw_argv)
    if internal_result is not None:
        return internal_result

    args = parse_args(raw_argv)
    from bootstrap import main as app_main, smoke_check_qt
    from ui_qt.app import QtDependencyError

    try:
        if args.smoke_check:
            return smoke_check_qt()
        return app_main()
    except QtDependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
