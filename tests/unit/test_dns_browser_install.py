from pathlib import Path
import sys

from tools.catalog_parser.dns_browser import ensure_playwright_browser


def test_browser_install_is_skipped_when_engine_exists(tmp_path: Path) -> None:
    executable = tmp_path / "firefox"
    executable.touch()
    install_calls: list[list[str]] = []

    result = ensure_playwright_browser(
        "firefox",
        progress=lambda _message: None,
        probe=lambda _engine: executable,
        installer=lambda command: install_calls.append(command) or 0,
    )

    assert result == executable
    assert install_calls == []


def test_missing_browser_is_installed_with_current_python(tmp_path: Path) -> None:
    executable = tmp_path / "chromium"
    install_calls: list[list[str]] = []

    def install(command: list[str]) -> int:
        install_calls.append(command)
        executable.touch()
        return 0

    result = ensure_playwright_browser(
        "chromium",
        progress=lambda _message: None,
        probe=lambda _engine: executable,
        installer=install,
    )

    assert result == executable
    assert install_calls == [
        [sys.executable, "-m", "playwright", "install", "chromium"]
    ]
