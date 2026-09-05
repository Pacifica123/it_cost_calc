from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable

from shared.runtime import (
    configure_playwright_environment,
    external_process_environment,
    playwright_install_command,
)

PLAYWRIGHT_BROWSER_ENGINES = ("firefox", "chromium")
_DNS_CHALLENGE_MARKERS = (
    "/__qrator/qauth_",
    "qauth_handle_validate_success",
)
_DNS_CHALLENGE_WAIT_SECONDS = 20.0


class DnsBrowserError(RuntimeError):
    """Expected browser setup or navigation error shown without a traceback."""


@dataclass(frozen=True, slots=True)
class DnsBrowserPage:
    requested_url: str
    final_url: str
    status_code: int | None
    title: str
    html: str


def _playwright_browser_revision(engine: str) -> str:
    """Read the browser revision bundled with the current Playwright package.

    This intentionally avoids parsing ``playwright install --dry-run`` output.
    The CLI text format has changed between Playwright releases and can differ
    in frozen builds, while ``browsers.json`` is the registry used by the
    bundled driver itself.
    """

    try:
        from playwright._impl._driver import compute_driver_executable
    except ModuleNotFoundError as exc:
        raise DnsBrowserError(
            "Playwright не установлен в окружении приложения. Переустановите зависимости проекта."
        ) from exc

    _driver_executable, driver_cli = compute_driver_executable()
    registry_path = Path(driver_cli).resolve().parent / "browsers.json"
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DnsBrowserError(
            f"Не удалось прочитать реестр браузеров Playwright: {registry_path}"
        ) from exc

    for item in payload.get("browsers", []):
        if str(item.get("name") or "") == engine:
            revision = str(item.get("revision") or "").strip()
            if revision:
                return revision
    raise DnsBrowserError(f"В реестре Playwright нет движка {engine}.")


def _playwright_browser_cache_root() -> Path:
    configured = configure_playwright_environment()
    if str(configured) != "0":
        return configured.expanduser().resolve()

    # Explicit PLAYWRIGHT_BROWSERS_PATH=0 is supported for source/dev runs.
    # Frozen runs are normalized to the writable user cache by runtime.py.
    try:
        from playwright._impl._driver import compute_driver_executable
    except ModuleNotFoundError as exc:
        raise DnsBrowserError(
            "Playwright не установлен в окружении приложения. Переустановите зависимости проекта."
        ) from exc
    _driver_executable, driver_cli = compute_driver_executable()
    return (Path(driver_cli).resolve().parent / ".local-browsers").resolve()


def _find_browser_executable(engine: str, install_dir: Path) -> Path:
    candidates: tuple[str, ...]
    if engine == "firefox":
        candidates = ("firefox.exe", "firefox")
    else:
        candidates = ("chrome.exe", "chrome", "Chromium")

    if install_dir.is_dir():
        for name in candidates:
            for path in install_dir.rglob(name):
                if path.is_file():
                    return path.resolve()
    # Return a useful diagnostic path even when installation is incomplete.
    if engine == "firefox":
        return install_dir / "firefox" / ("firefox.exe" if sys.platform.startswith("win") else "firefox")
    if sys.platform.startswith("win"):
        return install_dir / "chrome-win64" / "chrome.exe"
    if sys.platform == "darwin":
        return install_dir / "chrome-mac" / "Chromium.app" / "Contents" / "MacOS" / "Chromium"
    return install_dir / "chrome-linux" / "chrome"


def _playwright_executable_path(engine: str) -> Path:
    revision = _playwright_browser_revision(engine)
    install_dir = _playwright_browser_cache_root() / f"{engine}-{revision}"
    return _find_browser_executable(engine, install_dir)


def _system_chromium_executable() -> Path | None:
    """Find a host Chrome/Chromium that can be used for the DNS compatibility mode."""

    for command in (
        "google-chrome-stable",
        "google-chrome",
        "chromium",
        "chromium-browser",
    ):
        resolved = shutil.which(command)
        if resolved:
            return Path(resolved).resolve()

    candidates: list[Path] = []
    if sys.platform.startswith("win"):
        for env_name in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            base = os.environ.get(env_name)
            if not base:
                continue
            root = Path(base)
            candidates.extend(
                [
                    root / "Google" / "Chrome" / "Application" / "chrome.exe",
                    root / "Chromium" / "Application" / "chrome.exe",
                ]
            )
    elif sys.platform == "darwin":
        candidates.extend(
            [
                Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def ensure_playwright_browser(
    engine: str,
    *,
    progress: Callable[[str], None],
    probe: Callable[[str], Path] | None = None,
    installer: Callable[[list[str]], int] | None = None,
) -> Path:
    """Ensure a Playwright browser exists in the same writable cache used at launch."""

    if engine not in PLAYWRIGHT_BROWSER_ENGINES:
        raise DnsBrowserError(f"Неподдерживаемый Playwright-движок: {engine}")
    configure_playwright_environment()
    if probe is None:
        cache_root = _playwright_browser_cache_root()
        progress(f"Кэш Playwright: {cache_root}")
        probe = _playwright_executable_path
    executable = probe(engine)
    if executable.is_file():
        progress(f"Движок {engine} найден: {executable}")
        return executable

    command = playwright_install_command(engine)
    progress(f"Движок {engine} не найден. Запускаю автоматическую установку.")
    progress(f"Команда установки: {' '.join(command)}")
    if installer is None:
        return_code = subprocess.run(command, check=False).returncode
    else:
        return_code = installer(command)
    if return_code != 0:
        raise DnsBrowserError(
            f"Автоматическая установка {engine} завершилась с кодом {return_code}. "
            f"Повторите вручную: {' '.join(command)}"
        )
    executable = probe(engine)
    if not executable.is_file():
        raise DnsBrowserError(
            f"Playwright сообщил об установке {engine}, но executable не найден: {executable}"
        )
    progress(f"Движок {engine} установлен: {executable}")
    return executable


class DnsBrowserSession:
    """Visible, persistent Playwright session for user-initiated DNS capture."""

    def __init__(
        self,
        *,
        profile_dir: Path,
        engine: str,
        headless: bool,
        first_page_wait_seconds: float,
        progress: Callable[[str], None],
    ) -> None:
        self.profile_dir = Path(profile_dir)
        self.engine = engine
        self.headless = headless
        self.first_page_wait_seconds = max(0.0, float(first_page_wait_seconds))
        self.progress = progress
        self._playwright = None
        self._context = None
        self._page = None
        self._first_page = True

    def _launch_context(self, browser_type, *, executable_path: Path | None = None):
        options = {
            "user_data_dir": str(self.profile_dir),
            "headless": self.headless,
            "locale": "ru-RU",
            "viewport": {"width": 1280, "height": 800},
            "env": external_process_environment(),
        }
        if self.engine == "chromium":
            # Compatibility mode inspired by the separately supplied Selenium parser:
            # use the host Chromium when available and avoid the most obvious automation flag.
            options["args"] = ["--disable-blink-features=AutomationControlled"]
            options["ignore_default_args"] = ["--enable-automation"]
            if executable_path is not None:
                options["executable_path"] = str(executable_path)
        return browser_type.launch_persistent_context(**options)

    def __enter__(self) -> "DnsBrowserSession":
        configure_playwright_environment()
        try:
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:
            raise DnsBrowserError(
                "Playwright не установлен в окружении приложения. "
                "Переустановите зависимости проекта."
            ) from exc

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        system_chromium = _system_chromium_executable() if self.engine == "chromium" else None
        if self.engine != "chromium" or system_chromium is None:
            ensure_playwright_browser(self.engine, progress=self.progress)

        self._playwright = sync_playwright().start()
        try:
            browser_type = getattr(self._playwright, self.engine)
            if system_chromium is not None:
                self.progress(
                    f"DNS Chromium: найден системный браузер {system_chromium}. "
                    "Пробую совместимый режим перед bundled Chromium."
                )
                try:
                    self._context = self._launch_context(
                        browser_type,
                        executable_path=system_chromium,
                    )
                except Exception:
                    self.progress(
                        "Системный Chromium не запустился через Playwright; "
                        "переключаюсь на bundled Chromium."
                    )
                    ensure_playwright_browser("chromium", progress=self.progress)
                    self._context = self._launch_context(browser_type)
            else:
                self._context = self._launch_context(browser_type)
        except Exception as exc:
            self._playwright.stop()
            self._playwright = None
            raise DnsBrowserError(
                f"Не удалось запустить Playwright {self.engine}. "
                f"Повторите установку командой из журнала для движка {self.engine}."
            ) from exc

        if self.engine == "chromium":
            try:
                self._context.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
                )
            except Exception:
                # Launch itself succeeded; this compatibility hint is best-effort only.
                pass
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        self._page.set_default_navigation_timeout(45000)
        self.progress(f"Профиль браузера: {self.profile_dir}")
        return self

    def fetch(self, url: str) -> DnsBrowserPage:
        if self._page is None:
            raise DnsBrowserError("DNS browser session is not started")
        self.progress(f"Открываю {url}")
        response = self._page.goto(url, wait_until="domcontentloaded")
        self._page.wait_for_timeout(1500)
        status_code = response.status if response is not None else None
        html = self._page.content()
        challenge = any(marker in html.lower() for marker in _DNS_CHALLENGE_MARKERS)
        if challenge:
            wait_seconds = max(self.first_page_wait_seconds, _DNS_CHALLENGE_WAIT_SECONDS)
            self.progress(
                "DNS показал защитную проверку. Жду её завершения в открытом браузере."
            )
            self._page.wait_for_timeout(round(wait_seconds * 1000))
            self.progress("Повторно открываю страницу после защитной проверки DNS.")
            response = self._page.reload(wait_until="domcontentloaded")
            self._page.wait_for_timeout(2000)
            status_code = response.status if response is not None else None
            html = self._page.content()
        elif (
            self._first_page
            and self.first_page_wait_seconds
            and status_code not in {401, 403, 429}
        ):
            self.progress(
                "Первичная пауза: можно выбрать регион, принять cookies или пройти проверку в браузере."
            )
            self._page.wait_for_timeout(round(self.first_page_wait_seconds * 1000))
        self._first_page = False
        return DnsBrowserPage(
            requested_url=url,
            final_url=self._page.url,
            status_code=status_code,
            title=self._page.title(),
            html=html,
        )

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        try:
            if self._context is not None:
                try:
                    self._context.close()
                except Exception:
                    # The target may already be closed after a rejected navigation.
                    pass
        finally:
            if self._playwright is not None:
                try:
                    self._playwright.stop()
                except Exception:
                    pass
            self._context = self._page = self._playwright = None
