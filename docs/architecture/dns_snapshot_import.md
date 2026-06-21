# Офлайн-импорт DNS HTML/JSON-LD

## Назначение

Инструмент каталога умеет преобразовать заранее сохранённые карточки товаров DNS в catalog schema v2. Импорт выполняется отдельно от GUI, не делает сетевых запросов и не пытается обходить challenge или антибот-защиту.

Цепочка данных:

```text
сохранённые HTML + snapshot_manifest.json
  -> JSON-LD Product/Offer или ограниченный HTML fallback
  -> нормализация характеристик и provenance
  -> equipment_catalog.json schema v2
  -> staging-редактор
  -> явно подтверждённые варианты ТО
```

## Контракт снимка

Каталог снимка содержит `snapshot_manifest.json` и локальные `.html`/`.htm` файлы. Минимальный manifest:

```json
{
  "schema_version": 1,
  "source": "dns",
  "region": "Москва",
  "observed_at": "2026-06-21T12:00:00Z",
  "items": [
    {
      "file": "router.html",
      "category": "routers",
      "url": "https://www.dns-shop.ru/product/.../"
    }
  ]
}
```

Допустимые категории совпадают с `DNS_CATEGORY_MAP`: `routers`, `prebuilt_pcs`, `components`, `motherboards`, `cpus`, `gpus`, `rams`, `psus`, `ssds`, `hdds`, `cases`. Путь `file` обязан быть относительным, оставаться внутри каталога снимка и иметь HTML-расширение.

## Извлечение данных

Приоритет имеет `application/ld+json` с типом `Product`. Поддерживаются `@graph`, `Offer`, `Brand`, `model`, `mpn`/`sku`, GTIN и `additionalProperty`.

Если Product JSON-LD отсутствует или повреждён, ограниченный fallback читает `h1`/`title`, price meta и пары `dt`/`dd`. Причина fallback и отсутствующие цена/название записываются в warnings источника и review записи.

Из характеристик нормализуются runtime-поля:

- `ram_gb`, `cpu_cores`, `storage_gb`, `max_power_watts`;
- `lan_ports`, `lan_speed_mbps`, `wifi_total_mbps`, `ipv6_support`.

Единицы ТБ переводятся в ГБ по коэффициенту 1024, Гбит/с — в Мбит/с по коэффициенту 1000. Номинальная мощность блока питания намеренно не считается энергопотреблением устройства.

## Запуск

```bash
python scripts/update_equipment_catalog.py \
  --mode dns-snapshot \
  --input data/examples/parser/dns_snapshot \
  --output data/generated/catalog/equipment_catalog.json
```

Synthetic-пример не содержит сохранённых страниц реального магазина и служит только воспроизводимой проверкой контракта.

## Ограничения

- инструмент не скачивает страницы и не управляет браузером;
- HTML fallback намеренно ограничен и не зависит от CSS-селекторов DNS;
- распознанные значения не являются автоматически подтверждёнными;
- итоговые строки проходят существующий staging и ручное утверждение перед переносом в варианты ТО.
