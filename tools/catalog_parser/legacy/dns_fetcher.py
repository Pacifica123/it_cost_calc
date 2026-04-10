from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_fetcher(pw_profile_dir: str = "./pw_profile"):
    from playwright.sync_api import TimeoutError, sync_playwright

    Path(pw_profile_dir).mkdir(parents=True, exist_ok=True)
    pw = sync_playwright().start()
    logger.info("Запущен Playwright persistent profile: %s", pw_profile_dir)

    context = pw.chromium.launch_persistent_context(
        user_data_dir=pw_profile_dir,
        headless=False,
        locale="ru-RU",
        timezone_id="Europe/Riga",
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        viewport={"width": 1280, "height": 800},
    )

    page = context.new_page()
    page.set_default_navigation_timeout(30000)

    def fetch(url: str, wait: float | None = None) -> str:
        try:
            logger.debug("Playwright fetch: %s", url)
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except TimeoutError:
                logger.debug("networkidle timeout для %s", url)

            try:
                page.wait_for_selector("a[href]", timeout=10000)
            except TimeoutError:
                html = page.content()
                if len(html) < 20000:
                    logger.warning("Недостаточно HTML-контента для %s", url)
                    return ""
                return html

            html = page.content()
            if wait:
                page.wait_for_timeout(int(wait * 1000))
            if len(html) < 20000:
                logger.warning("HTML слишком короткий для %s", url)
                return ""
            return html
        except Exception:
            logger.exception("Ошибка при загрузке страницы: %s", url)
            return ""

    def close():
        try:
            context.close()
            pw.stop()
            logger.info("Playwright context закрыт")
        except Exception:
            logger.exception("Не удалось корректно закрыть Playwright context")

    return fetch, close
