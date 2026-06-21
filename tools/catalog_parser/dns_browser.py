from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Callable

PLAYWRIGHT_BROWSER_ENGINES = ("firefox", "chromium")


class DnsBrowserError(RuntimeError):
    """Expected browser setup or navigation error shown without a traceback."""


@dataclass(frozen=True, slots=True)
class DnsBrowserPage:
    requested_url: str
    final_url: str
    status_code: int | None
    title: str
    html: str


def _playwright_executable_path(engine: str) -> Path:
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise DnsBrowserError(
            "Playwright не установлен в окружении приложения. Переустановите зависимости проекта."
        ) from exc
    playwright = sync_playwright().start()
    try:
        browser_type = getattr(playwright, engine)
        return Path(browser_type.executable_path)
    finally:
        playwright.stop()


def ensure_playwright_browser(
    engine: str,
    *,
    progress: Callable[[str], None],
    probe: Callable[[str], Path] | None = None,
    installer: Callable[[list[str]], int] | None = None,
) -> Path:
    """Install a missing Playwright browser with the current Python interpreter."""

    if engine not in PLAYWRIGHT_BROWSER_ENGINES:
        raise DnsBrowserError(f"Неподдерживаемый Playwright-движок: {engine}")
    probe = probe or _playwright_executable_path
    executable = probe(engine)
    if executable.is_file():
        progress(f"Движок {engine} найден: {executable}")
        return executable

    command = [sys.executable, "-m", "playwright", "install", engine]
    progress(f"Движок {engine} не найден. Запускаю автоматическую установку.")
    progress(f"Команда установки: {' '.join(command)}")
    if installer is None:
        return_code = subprocess.run(command, check=False).returncode
    else:
        return_code = installer(command)
    if return_code != 0:
        raise DnsBrowserError(
            f"Автоматическая установка {engine} завершилась с кодом {return_code}. "
            f"Повторите вручную: {sys.executable} -m playwright install {engine}"
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

    def __enter__(self) -> "DnsBrowserSession":
        ensure_playwright_browser(self.engine, progress=self.progress)
        try:
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:
            raise DnsBrowserError(
                "Playwright не установлен в окружении приложения. "
                "Переустановите зависимости проекта."
            ) from exc

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()
        try:
            browser_type = getattr(self._playwright, self.engine)
            self._context = browser_type.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                headless=self.headless,
                locale="ru-RU",
                viewport={"width": 1280, "height": 800},
            )
        except Exception as exc:
            self._playwright.stop()
            self._playwright = None
            raise DnsBrowserError(
                f"Не удалось запустить Playwright {self.engine}. "
                f"Повторите установку: {sys.executable} -m playwright install {self.engine}"
            ) from exc
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
        if (
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
            html=self._page.content(),
        )

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        try:
            if self._context is not None:
                self._context.close()
        finally:
            if self._playwright is not None:
                self._playwright.stop()
            self._context = self._page = self._playwright = None
