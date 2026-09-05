from pathlib import Path
import sys

from tools.catalog_parser.dns_browser import DnsBrowserSession, ensure_playwright_browser


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


def test_first_qrator_401_is_retried_after_challenge_wait(tmp_path: Path) -> None:
    class Response:
        def __init__(self, status: int) -> None:
            self.status = status

    class Page:
        url = "https://www.dns-shop.ru/"

        def __init__(self) -> None:
            self.reloaded = False
            self.waits: list[int] = []

        def goto(self, _url: str, *, wait_until: str):
            assert wait_until == "domcontentloaded"
            return Response(401)

        def reload(self, *, wait_until: str):
            assert wait_until == "domcontentloaded"
            self.reloaded = True
            return Response(200)

        def wait_for_timeout(self, milliseconds: int) -> None:
            self.waits.append(milliseconds)

        def content(self) -> str:
            if self.reloaded:
                return "<html><body>DNS catalog</body></html>"
            return '<script src="/__qrator/qauth_test.js"></script>'

        def title(self) -> str:
            return "DNS" if self.reloaded else ""

    page = Page()
    messages: list[str] = []
    session = DnsBrowserSession(
        profile_dir=tmp_path,
        engine="chromium",
        headless=False,
        first_page_wait_seconds=8,
        progress=messages.append,
    )
    session._page = page

    result = session.fetch("https://www.dns-shop.ru/")

    assert result.status_code == 200
    assert page.reloaded is True
    assert 20_000 in page.waits
    assert any("защитную проверку" in message for message in messages)


def test_dns_chromium_compatibility_launch_uses_system_executable(tmp_path: Path) -> None:
    from tools.catalog_parser.dns_browser import DnsBrowserSession

    captured: dict = {}

    class BrowserType:
        def launch_persistent_context(self, **kwargs):
            captured.update(kwargs)
            return object()

    executable = tmp_path / "chromium"
    session = DnsBrowserSession(
        profile_dir=tmp_path / "profile",
        engine="chromium",
        headless=False,
        first_page_wait_seconds=8,
        progress=lambda _message: None,
    )

    result = session._launch_context(BrowserType(), executable_path=executable)

    assert result is not None
    assert captured["executable_path"] == str(executable)
    assert "--disable-blink-features=AutomationControlled" in captured["args"]
    assert "--enable-automation" in captured["ignore_default_args"]


def test_system_chromium_probe_prefers_path_binary(monkeypatch, tmp_path: Path) -> None:
    from tools.catalog_parser import dns_browser

    executable = tmp_path / "chromium"
    executable.write_text("", encoding="utf-8")

    def fake_which(command: str) -> str | None:
        return str(executable) if command == "chromium" else None

    monkeypatch.setattr(dns_browser.shutil, "which", fake_which)

    assert dns_browser._system_chromium_executable() == executable.resolve()
