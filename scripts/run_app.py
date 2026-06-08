from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IT Cost Calc desktop UI.")
    parser.add_argument(
        "--smoke-check",
        action="store_true",
        help="Create Qt objects without showing the window.",
    )
    parser.add_argument(
        "--legacy-tk",
        action="store_true",
        help="Run the archived Tkinter UI fallback.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    from bootstrap import main as app_main, main_legacy_tk, smoke_check_qt
    from ui_qt.app import QtDependencyError

    try:
        if args.legacy_tk:
            return main_legacy_tk()
        if args.smoke_check:
            return smoke_check_qt()
        return app_main()
    except QtDependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
