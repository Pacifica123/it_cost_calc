# Структурированные источники каталога (P1)

## Зачем нужен этот слой

Основной продукт — система поддержки принятия решений по ИТ-инфраструктуре, а не web-scraper.
Поэтому получение рынка переносится с consumer-витрин на структурированные источники, которые
изначально пригодны для машинной обработки: прайсы поставщиков, XLSX/CSV и YML/XML feeds.

```text
локальный XLSX/CSV/YML/XML ─┐
URL XLSX/CSV/YML/XML ───────┤
публичный supplier feed ────┘
                            ↓
                     feed-download
                            ↓
                 raw file + fetch_manifest
                            ↓
                     Catalog staging
                            ↓
              review / approve / runtime ТО
                            ↓
                GA / AHP / Pareto / Hybrid
```

DNS-Shop и Яндекс Маркет не удалены из CLI, но больше не являются пользовательским live-путём
экрана «Каталог». Их browser/HTTP режимы остаются исследовательскими и диагностическими.

## Поддерживаемые форматы

- `XLSX` — первый лист; заголовок может находиться после нескольких служебных строк;
- `CSV` — UTF-8/UTF-8-BOM или Windows-1251, разделитель `,`, `;` или tab;
- `YML/XML` — `categories` + `offers`; читаются `name`, `vendor`, `vendorCode`, `price`,
  `currencyId`, `categoryId`, `url`, `barcode/gtin` и `param`;
- локальный `JSON` schema v1/v2 по-прежнему читается staging напрямую.

Для плоских таблиц поддерживаются распространённые русские и английские названия колонок:
`Наименование`, `Категория`, `Цена`, `Бренд/Производитель`, `PN/MPN`, `Код товара/SKU`,
`Наличие`, `URL` и их английские варианты.

## Источник и provenance

У feed-запуска есть отдельный `fetch_manifest.json`:

- `source_id`, `source_name`;
- исходный и разрешённый URL;
- формат;
- регион;
- `price_kind`;
- `observed_at`;
- SHA-256 и размер скачанного файла.

При переносе в staging эти данные становятся first-class metadata:

- `catalog_item.source`;
- `offer.observed_at`;
- `offer.price_kind`;
- `offer.source_url`;
- `field_provenance.feed`;
- provenance для `title`, `category`, `price` и `identity`.

Таким образом цена трактуется как наблюдение конкретного источника в конкретный момент, а не как
вечное свойство модели оборудования.

## Конфигурация источников

Встроенные пресеты находятся в:

`data/catalog/source_presets.json`

Пользовательские источники сохраняются в writable runtime data:

`data/generated/catalog/catalog_sources.json`

Текущие пресеты:

- **ТехноСити** — публичный XLSX `https://www.technocity.ru/upload/tc_price.xlsx`;
- **B2BCORP** — публичный XLSX на Яндекс Диске; приложение получает временную ссылку загрузки
  через API публичного ресурса Яндекс Диска.

Публичный feed остаётся внешним источником: его доступность и формат могут измениться. Ошибка
загрузки не изменяет staging и не подменяется синтетическими данными.

## Категории

Staging пытается привести русские категории к текущему runtime-контракту:

- серверы;
- рабочие станции / готовые ПК / ноутбуки;
- роутеры / коммутаторы / точки доступа;
- принтеры и мониторы.

Комплектующие (`CPU`, `GPU`, RAM, материнские платы, SSD/HDD и т.п.) намеренно остаются
заблокированными. Например, «Серверная оперативная память» не должна ошибочно становиться
готовым сервером только из-за слова «серверная».

## CLI

Структурированный feed можно получить без GUI:

```bash
python scripts/update_equipment_catalog.py \
  --mode feed-download \
  --input https://www.technocity.ru/upload/tc_price.xlsx \
  --feed-source-id technocity \
  --feed-source-name ТехноСити \
  --feed-format xlsx \
  --feed-price-kind supplier_price \
  --feed-manifest data/generated/catalog/feed_runs/manual/fetch_manifest.json \
  --region Новосибирск \
  --output data/generated/catalog/feed_runs/manual/technocity.xlsx
```

Команда только получает raw feed и provenance-манифест. Нормализация в runtime по-прежнему
проходит через staging и явное подтверждение пользователя.


## Продолжение: P2

После загрузки feed staging больше не рассматривает источники изолированно.
Федерация товаров, `offers[]`, выбор effective price и правила refresh описаны в
`docs/architecture/catalog_federation_p2.md`.
