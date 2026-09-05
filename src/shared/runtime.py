from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping

_APP_DIR_NAME = "ITCostCalc"


def is_frozen_runtime() -> bool:
    """Return True for PyInstaller-style frozen executables."""

    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """Resolve the immutable project resources root.

    In source runs this is the repository root. In a PyInstaller build this is
    the bundle root (``sys._MEIPASS``), where ``--add-data ...:data`` is stored.
    """

    if is_frozen_runtime():
        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root:
            return Path(bundle_root).resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def writable_runtime_root() -> Path:
    """Resolve a persistent writable root for generated runtime data."""

    if not is_frozen_runtime():
        return resource_root()

    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or Path.home())
        return (base / _APP_DIR_NAME).resolve()
    if sys.platform == "darwin":
        return (Path.home() / "Library" / "Application Support" / _APP_DIR_NAME).resolve()

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data_home).expanduser() if xdg_data_home else Path.home() / ".local" / "share"
    return (base / _APP_DIR_NAME).resolve()


def catalog_parser_process(repo_root: str | Path) -> tuple[str, tuple[str, ...]]:
    """Return program/prefix for starting the catalog parser in a child process.

    Source runs use ``python -u scripts/update_equipment_catalog.py``. A frozen
    build must relaunch its own executable with the internal parser switch,
    because ``sys.executable -u script.py`` would feed Python options to the
    application executable itself.
    """

    if is_frozen_runtime():
        return sys.executable, ("--catalog-parser",)
    script = Path(repo_root) / "scripts" / "update_equipment_catalog.py"
    return sys.executable, ("-u", str(script))


def playwright_install_command(engine: str) -> list[str]:
    """Build an install command that also works from a frozen release."""

    if is_frozen_runtime():
        return [sys.executable, "--playwright-install", engine]
    return [sys.executable, "-m", "playwright", "install", engine]


def playwright_browsers_path(
    source: Mapping[str, str] | None = None,
) -> Path:
    """Return a writable Playwright browser cache shared by source/frozen runs.

    Playwright intentionally changes ``PLAYWRIGHT_BROWSERS_PATH`` to ``0`` in a
    frozen Python process. That makes it look for browsers next to the bundled
    driver, which is read-only inside AppImage and disposable in one-file builds.
    Use the normal per-user cache instead so browser installation and launch agree.
    """

    env = os.environ if source is None else source
    configured = str(env.get("PLAYWRIGHT_BROWSERS_PATH") or "").strip()
    if configured and not (is_frozen_runtime() and configured == "0"):
        if configured == "0":
            # Source/dev users may deliberately request Playwright's hermetic cache.
            return Path("0")
        return Path(configured).expanduser().resolve()

    if sys.platform.startswith("win"):
        base = Path(env.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        xdg_cache_home = str(env.get("XDG_CACHE_HOME") or "").strip()
        base = Path(xdg_cache_home).expanduser() if xdg_cache_home else Path.home() / ".cache"
    return (base / "ms-playwright").resolve()


def configure_playwright_environment(
    target: dict[str, str] | None = None,
) -> Path:
    """Keep Playwright browser lookup on a persistent writable user cache.

    The mapping defaults to ``os.environ`` because Playwright's Python transport
    copies the process environment when it starts its Node driver.
    """

    env = os.environ if target is None else target
    configured = str(env.get("PLAYWRIGHT_BROWSERS_PATH") or "").strip()
    if not configured or (is_frozen_runtime() and configured == "0"):
        path = playwright_browsers_path(env)
        env["PLAYWRIGHT_BROWSERS_PATH"] = str(path)
        return path
    if configured == "0":
        return Path("0")
    return Path(configured).expanduser().resolve()


def external_process_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment safe for launching host desktop applications.

    PyInstaller/AppImage may prepend bundled libraries to ``LD_LIBRARY_PATH``.
    Host tools such as a file manager must not inherit those private libraries.
    """

    env = dict(source or os.environ)
    if sys.platform.startswith("linux") and (is_frozen_runtime() or env.get("APPIMAGE")):
        original = env.get("LD_LIBRARY_PATH_ORIG")
        if original is None:
            env.pop("LD_LIBRARY_PATH", None)
        else:
            env["LD_LIBRARY_PATH"] = original
        for name in ("PYTHONHOME", "PYTHONPATH"):
            env.pop(name, None)
    return env


def open_directory(path: str | Path) -> Path:
    """Open an existing directory with the host file manager."""

    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(str(target))  # type: ignore[attr-defined]
        return target

    env = external_process_environment()
    if sys.platform == "darwin":
        command = ["open", str(target)]
    else:
        opener = shutil.which("xdg-open") or shutil.which("gio")
        if opener is None:
            raise RuntimeError("Не найден системный обработчик папок (xdg-open/gio).")
        command = [opener, str(target)] if Path(opener).name != "gio" else [opener, "open", str(target)]

    subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return target
