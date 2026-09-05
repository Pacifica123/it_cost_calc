from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from shared.runtime import external_process_environment

from .dns_browser import DnsBrowserError, ensure_playwright_browser


class YandexMarketBrowserError(DnsBrowserError):
    """Expected Yandex Market browser setup or navigation error."""


@dataclass(frozen=True, slots=True)
class YandexMarketBrowserPage:
    requested_url: str
    final_url: str
    status_code: int | None
    title: str
    html: str


class YandexMarketBrowserSession:
    """Visible persistent Playwright session for a bounded Market capture."""

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

    def __enter__(self) -> "YandexMarketBrowserSession":
        try:
            ensure_playwright_browser(self.engine, progress=self.progress)
            from playwright.sync_api import sync_playwright
        except (DnsBrowserError, ModuleNotFoundError) as exc:
            raise YandexMarketBrowserError(str(exc)) from exc

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()
        try:
            browser_type = getattr(self._playwright, self.engine)
            self._context = browser_type.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                headless=self.headless,
                locale="ru-RU",
                viewport={"width": 1366, "height": 900},
                env=external_process_environment(),
            )
        except Exception as exc:
            self._playwright.stop()
            self._playwright = None
            raise YandexMarketBrowserError(
                f"Не удалось запустить Playwright {self.engine} для Яндекс Маркета."
            ) from exc
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        self._page.set_default_navigation_timeout(60000)
        self.progress(f"Профиль браузера Яндекс Маркета: {self.profile_dir}")
        return self

    def fetch(self, url: str) -> YandexMarketBrowserPage:
        if self._page is None:
            raise YandexMarketBrowserError("Yandex Market browser session is not started")
        self.progress(f"Открываю {url}")
        try:
            response = self._page.goto(url, wait_until="domcontentloaded")
            self._page.wait_for_timeout(1800)
            status_code = response.status if response is not None else None
            if self._first_page and self.first_page_wait_seconds and status_code not in {401, 403, 429}:
                self.progress(
                    "Первичная пауза: можно выбрать регион, принять cookies или пройти проверку."
                )
                self._page.wait_for_timeout(round(self.first_page_wait_seconds * 1000))
            if "/category/" in self._page.url or "/search" in self._page.url:
                self._page.evaluate("window.scrollTo(0, Math.min(document.body.scrollHeight, 2400))")
                self._page.wait_for_timeout(1200)
                self._page.evaluate("window.scrollTo(0, 0)")
            html = self._page.content()
        except Exception as exc:
            raise YandexMarketBrowserError(
                f"Не удалось открыть страницу Яндекс Маркета: {url}"
            ) from exc
        self._first_page = False
        return YandexMarketBrowserPage(
            requested_url=url,
            final_url=self._page.url,
            status_code=status_code,
            title=self._page.title(),
            html=html,
        )

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        try:
            if self._context is not None:
                self._context.close()
        finally:
            if self._playwright is not None:
                self._playwright.stop()
            self._context = self._page = self._playwright = None
