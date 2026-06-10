from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from collections.abc import Iterable

MIN_PYTHON = (3, 11)
RUNTIME_IMPORTS: tuple[tuple[str, str], ...] = (
    ("PySide6", "PySide6"),
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
    ("bs4", "beautifulsoup4"),
    ("lxml", "lxml"),
    ("requests", "requests"),
    ("tqdm", "tqdm"),
)
BUILD_IMPORTS: tuple[tuple[str, str], ...] = (("PyInstaller", "pyinstaller"),)


def _version_text() -> str:
    version = sys.version_info
    return f"{version.major}.{version.minor}.{version.micro}"


def _format_packages(packages: Iterable[str]) -> str:
    return ", ".join(sorted(packages))


def check_python_version() -> bool:
    current = sys.version_info[:3]
    required = ".".join(str(part) for part in MIN_PYTHON)
    if current < (*MIN_PYTHON, 0):
        print(
            f"ERROR: нужен Python >= {required}. Обнаружен Python {_version_text()} "
            f"({platform.python_implementation()})."
        )
        print("Для разработки установите Python 3.11+ или используйте готовую сборку exe/AppImage.")
        return False
    print(f"OK: Python {_version_text()} подходит для проекта (требуется >= {required}).")
    return True


def check_imports(imports: tuple[tuple[str, str], ...], *, title: str) -> bool:
    missing: list[str] = []
    for import_name, package_name in imports:
        if importlib.util.find_spec(import_name) is None:
            missing.append(package_name)
    if missing:
        print(f"ERROR: не найдены зависимости {title}: {_format_packages(missing)}.")
        return False
    print(f"OK: зависимости {title} доступны.")
    return True


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка окружения IT Cost Calc.")
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Дополнительно проверить runtime-зависимости из requirements/base.txt.",
    )
    parser.add_argument(
        "--check-build-deps",
        action="store_true",
        help="Дополнительно проверить зависимости для сборки exe/AppImage.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    ok = check_python_version()
    if args.check_deps:
        ok = check_imports(RUNTIME_IMPORTS, title="приложения") and ok
        if not ok:
            print("Установите их командой: pip install -r requirements/base.txt")
    if args.check_build_deps:
        ok = check_imports(BUILD_IMPORTS, title="сборки") and ok
        if not ok:
            print("Установите их командой: pip install -r requirements/build.txt")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
