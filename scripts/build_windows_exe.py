from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 11)
ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
ENTRYPOINT = ROOT / "scripts" / "run_app.py"
DEFAULT_NAME = "ITCostCalc"
PROJECT_HIDDEN_IMPORT_PACKAGES = ("ui_qt",)


def _require_python() -> None:
    if sys.version_info < (*MIN_PYTHON, 0):
        current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        required = ".".join(str(part) for part in MIN_PYTHON)
        raise SystemExit(
            f"ERROR: для сборки нужен Python >= {required}. Обнаружен Python {current}. "
            "Установите Python 3.11+ или используйте готовую сборку."
        )


def _require_pyinstaller() -> None:
    if shutil.which("pyinstaller") is None:
        try:
            subprocess.run(
                [sys.executable, "-m", "PyInstaller", "--version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise SystemExit(
                "ERROR: PyInstaller не установлен. Выполните: "
                "pip install -r requirements/build.txt"
            ) from None


def _add_data_arg(source: Path, target: str) -> str:
    separator = ";" if sys.platform.startswith("win") else ":"
    return f"{source}{separator}{target}"


def _iter_package_modules(package_name: str) -> list[str]:
    """Return project modules that PyInstaller cannot see through lazy imports."""

    package_dir = SRC_ROOT / package_name.replace(".", "/")
    if not package_dir.exists():
        return []

    modules: list[str] = []
    for path in sorted(package_dir.rglob("*.py")):
        if any(part == "__pycache__" for part in path.parts):
            continue
        relative = path.relative_to(SRC_ROOT).with_suffix("")
        if relative.name == "__init__":
            continue
        modules.append(".".join(relative.parts))
    return modules


def project_hidden_imports() -> list[str]:
    """Collect lazy Qt modules explicitly for frozen builds."""

    hidden_imports: list[str] = []
    for package_name in PROJECT_HIDDEN_IMPORT_PACKAGES:
        hidden_imports.extend(_iter_package_modules(package_name))
    return hidden_imports


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        args.name,
        "--paths",
        str(SRC_ROOT),
        "--distpath",
        str(ROOT / "dist" / "windows"),
        "--workpath",
        str(ROOT / "build" / "pyinstaller-windows"),
        "--specpath",
        str(ROOT / "build" / "pyinstaller-specs"),
        "--add-data",
        _add_data_arg(ROOT / "data", "data"),
        "--add-data",
        _add_data_arg(ROOT / "src" / "ui_qt" / "design", "ui_qt/design"),
        "--collect-data",
        "matplotlib",
        "--hidden-import",
        "matplotlib.backends.backend_qtagg",
    ]
    for module_name in project_hidden_imports():
        command.extend(("--hidden-import", module_name))
    if args.onefile:
        command.append("--onefile")
    else:
        command.append("--onedir")
    if not args.console:
        command.append("--windowed")
    command.append(str(ENTRYPOINT))
    return command


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Собрать Windows exe для IT Cost Calc.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Имя исполняемого файла.")
    parser.add_argument("--onefile", action="store_true", help="Собрать один exe вместо onedir.")
    parser.add_argument("--console", action="store_true", help="Оставить консольное окно для диагностики.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _require_python()
    _require_pyinstaller()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    command = build_command(args)
    subprocess.run(command, cwd=ROOT, check=True)
    output_dir = ROOT / "dist" / "windows"
    print(f"Готово. Артефакты сборки: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
