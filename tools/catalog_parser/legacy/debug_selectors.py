"""Selector debugging helper for catalog parser research."""

# debug_selectors.py
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from pathlib import Path
import time

BASE_URL = "https://www.dns-shop.ru"
QUERY = "Маршрутизаторы"
OUT = "debug_search.html"
SELECTORS = [
    "a.catalog-product__name",
    "a.product-info__title-link",
    "a.product-card__name",
    ".product-card .product-card__title a",
    ".catalog-product__name a",
    ".product-card__name a",
    ".catalog-product__title a",
    ".products-list__item a",
    ".product-name a",
    "a[href*='/product/']",
    "a[href*='/product/'] > span",
]


def fetch_and_save():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False
        )  # headful — удобно для визуальной инспекции
        ctx = browser.new_context()
        page = ctx.new_page()
        url = f"{BASE_URL}/search/?q={QUERY}"
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        # подождём загрузки сетевых ресурсов; при необходимости увеличить
        page.wait_for_load_state("networkidle", timeout=30000)
        time.sleep(1.2)
        html = page.content()
        Path(OUT).write_text(html, encoding="utf-8")
        print(
            f"Saved {OUT} ({len(html)} bytes). Откройте файл в браузере для инспекции."
        )
        # подсчёт селекторов
        soup = BeautifulSoup(html, "lxml")
        for sel in SELECTORS:
            cnt = len(soup.select(sel))
            print(f"{sel}: {cnt}")
        browser.close()


if __name__ == "__main__":
    fetch_and_save()
