BASE_URL = "https://www.dns-shop.ru"
OUTPUT_FILE = "dns_data_playwright.json"
MAX_JSON_BYTES = 10 * 1024 * 1024  # 10 MB
PER_CATEGORY_LIMIT = 500  # максимум товаров на одну категорию (регулируйте)
TOTAL_TIME_LIMIT_SECONDS = 300  # ваш желаемый лимит (5 минут)

# категории: query -> type_tag
CATEGORIES = {
    "routers": "Маршрутизаторы",
    "prebuilt_pcs": "Готовые сборки ПК",  # подстройте, если в вашем регионе по-другому
    "motherboards": "Материнская плата",
    "cpus": "Процессор",
    "gpus": "Видеокарта",
    "rams": "Оперативная память",
    "psus": "Блок питания",
    "ssds": "SSD",
    "hdds": "HDD",
    "cases": "Корпус",
}

DNS_SPECS_SELECTORS = [
    "[data-testid='characteristic-item']",
    ".ProductCharacteristics__list-item",
    ".ui-characteristics__item",
    ".params-table tr",
    ".specifications__item",
    "product-card-top__specs_item",
]

# Случайные сборки, сколько сгенерировать
GENERATED_BUILDS_COUNT = 12

# ====== Вспомогательные парсеры ======

