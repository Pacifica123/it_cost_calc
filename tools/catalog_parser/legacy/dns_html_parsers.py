from typing import Any, Dict, List

from bs4 import BeautifulSoup

from .dns_constants import BASE_URL, DNS_SPECS_SELECTORS
from .dns_spec_utils import extract_price_int, norm_key, parse_wattage

def parse_search_html(html: str, limit: int) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    items = []
    # пробуем несколько селекторов, в порядке приоритетов
    selectors = [
        "a.catalog-product__name",
        "a.product-info__title-link",
        "a.product-card__name",
        ".product-card .product-card__title a",
        ".catalog-product__name a",
        ".product-card-top__specs-item",
    ]
    for sel in selectors:
        for a in soup.select(sel):
            href = a.get("href") or a.get("data-href") or a.get("data-url")
            title = a.get_text(strip=True)
            if not href or not title:
                continue
            if href.startswith("/"):
                href = BASE_URL + href
            items.append({"title": title, "url": href})
            if len(items) >= limit:
                return items
        if items:
            return items
    # fallback: карточки
    for card in soup.select(".product-card, .catalog-product, .catalog-item"):
        a = card.select_one("a")
        if not a:
            continue
        href = a.get("href") or a.get("data-href")
        title = a.get_text(strip=True)
        if href and href.startswith("/"):
            href = BASE_URL + href
        items.append({"title": title, "url": href})
        if len(items) >= limit:
            break
    return items

def parse_product_html(html: str, url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    # try ld+json
    title = ""
    price_text = ""
    specs = {}
    # ld+json
    for s in soup.select('script[type="application/ld+json"]'):
        try:
            j = json.loads(s.string or "{}")
            if isinstance(j, dict) and j.get("name"):
                title = title or j.get("name", "")
            offers = j.get("offers")
            if offers and isinstance(offers, dict):
                price_text = price_text or str(offers.get("price", ""))
        except Exception:
            continue
    # fallback selectors
    if not title:
        h1 = soup.select_one("h1")
        if h1:
            title = h1.get_text(strip=True)
    # price
    if not price_text:
        el = soup.select_one(
            ".product-buy__price, .product-price__current, .price__current, .price-block__final-price"
        )
        if el:
            price_text = el.get_text(strip=True)
    # specs: таблицы и блоки
    # таблицы

    # Новый парсер specs для DNS
    for selector in DNS_SPECS_SELECTORS:
        for item in soup.select(selector):
            # Вариант 1: ключ-значение в соседних элементах
            key_el = item.select_one(
                ".name, .label, .key, [class*='name'], [class*='key']"
            )
            val_el = item.select_one(".value, .data, [class*='value']")
            if key_el and val_el:
                specs[key_el.get_text(strip=True)] = val_el.get_text(strip=True)
                continue

            # Вариант 2: dt/dd или th/td в item
            dts = item.select_one("dt, th, .dt")
            dds = item.select_one("dd, td, .dd")
            if dts and dds:
                specs[dts.get_text(strip=True)] = dds.get_text(strip=True)
                continue

            # Вариант 3: два span'а подряд
            spans = item.select("span")
            if len(spans) >= 2:
                specs[spans[0].get_text(strip=True)] = spans[1].get_text(strip=True)

    # Дополнительно: JSON-LD и микроданные
    scripts = soup.select('script[type="application/ld+json"]')
    for script in scripts:
        try:
            data = json.loads(script.string or "{}")
            if "additionalProperty" in data:
                for prop in data.get("additionalProperty", []):
                    name = prop.get("name", "")
                    value = prop.get("value", "")
                    if name and value:
                        specs[name] = str(value)
        except:
            pass

    # for table in soup.select(" "):
    #     for row in table.select("tr"):
    #         cols = row.find_all(["td", "th"])
    #         if len(cols) >= 2:
    #             k = cols[0].get_text(strip=True)
    #             v = cols[1].get_text(strip=True)
    #             if k:
    #                 specs[k] = v
    # # другие блоки
    # for item in soup.select(
    #     ".product-params__item, .product-props__item, .characteristics__item"
    # ):
    #     k_el = item.select_one(".product-params__name, .char__name, .prop__name")
    #     v_el = item.select_one(".product-params__value, .char__value, .prop__value")
    #     if k_el and v_el:
    #         specs[k_el.get_text(strip=True)] = v_el.get_text(strip=True)

    return {
        "title": title or "",
        "url": url,
        "price": price_text or "",
        "price_int": extract_price_int(price_text or ""),
        "specs": specs,
    }
