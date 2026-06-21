from __future__ import annotations

from pathlib import Path
from typing import Callable


class DnsBrowserSession:
    """Visible, persistent Playwright session for user-initiated DNS capture."""

    def __init__(
        self,
        *,
        profile_dir: Path,
        headless: bool,
        first_page_wait_seconds: float,
        progress: Callable[[str], None],
    ) -> None:
        self.profile_dir = Path(profile_dir)
        self.headless = headless
        self.first_page_wait_seconds = max(0.0, float(first_page_wait_seconds))
        self.progress = progress
        self._playwright = None
        self._context = None
        self._page = None
        self._first_page = True

    def __enter__(self) -> "DnsBrowserSession":
        try:
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Playwright не установлен. Установите зависимости проекта и выполните "
                "'python -m playwright install chromium'."
            ) from exc

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()
        try:
            self._context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                headless=self.headless,
                locale="ru-RU",
                viewport={"width": 1280, "height": 800},
            )
        except Exception as exc:
            self._playwright.stop()
            self._playwright = None
            raise RuntimeError(
                "Не удалось запустить Chromium. Выполните "
                "'python -m playwright install chromium' и повторите запуск."
            ) from exc
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        self._page.set_default_navigation_timeout(45000)
        self.progress(f"Профиль браузера: {self.profile_dir}")
        return self

    def fetch(self, url: str) -> str:
        if self._page is None:
            raise RuntimeError("DNS browser session is not started")
        self.progress(f"Открываю {url}")
        self._page.goto(url, wait_until="domcontentloaded")
        self._page.wait_for_timeout(1500)
        if self._first_page and self.first_page_wait_seconds:
            self.progress(
                "Первичная пауза: можно выбрать регион, принять cookies или пройти проверку в браузере."
            )
            self._page.wait_for_timeout(round(self.first_page_wait_seconds * 1000))
        self._first_page = False
        return self._page.content()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        try:
            if self._context is not None:
                self._context.close()
        finally:
            if self._playwright is not None:
                self._playwright.stop()
            self._context = self._page = self._playwright = None
