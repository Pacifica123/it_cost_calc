"""Alternative parser experiment preserved as a separate tool script."""

from playwright.sync_api import sync_playwright
from pathlib import Path


def start_persistent(profile_dir: str = "./pw_profile"):
    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    pw = sync_playwright().start()
    ctx = pw.chromium.launch_persistent_context(
        user_data_dir=profile_dir,
        headless=False,  # обязательно headful при антиботе
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        locale="ru-RU",
        timezone_id="Europe/Riga",
        viewport={"width": 1280, "height": 800},
    )
    page = ctx.new_page()
    # можно установить дополнительные заголовки
    page.set_extra_http_headers(
        {
            "Accept-Language": "ru-RU,ru;q=0.9",
            "Referer": "https://www.google.com/",
        }
    )
    return pw, ctx, page


# использование
pw, ctx, page = start_persistent()
page.goto("https://www.dns-shop.ru", wait_until="domcontentloaded")
print("HTML len:", len(page.content()))
# при успешном проходе профиль будет содержать cookies — можно переиспользовать
# ctx.close(); pw.stop() когда закончите
