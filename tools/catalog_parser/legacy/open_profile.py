"""Utility for opening a local Playwright profile. Left as a tool; profile data is excluded from repo."""

# open_profile.py
from playwright.sync_api import sync_playwright
from pathlib import Path

profile_dir = "./pw_profile"
Path(profile_dir).mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(
        user_data_dir=profile_dir,
        headless=False,
        locale="ru-RU",
        timezone_id="Europe/Riga",
        args=["--disable-blink-features=AutomationControlled"],
    )
    page = ctx.new_page()
    page.goto("https://www.dns-shop.ru", wait_until="domcontentloaded")
    print("Открыл профиль. Пройдите капчу/ожидание вручную в открывшемся окне.")
    input(
        "Нажмите Enter после того, как страница загрузится и вы убедитесь, что видите товары..."
    )
    ctx.close()
