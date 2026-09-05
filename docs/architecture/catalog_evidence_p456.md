# P4–P6 — независимые доказательства рынка и fallback-ввод

После P1–P3 каталог разделён на feeds, федерацию цен и enrichment характеристик.
P4–P6 завершают эту модель: проект получает независимый benchmark закупок,
коммерческие предложения и единичный browser capture без автоматизации браузера.

```text
supplier feeds ─────────────┐
commercial quote ───────────┼─> offers[] -> effective offer -> ТО / GA / AHP / NPV
browser capture ────────────┘

EIS contracts ──────────────> procurement_benchmark (только статистика)
Icecat ─────────────────────> specification_sources (только характеристики)
```

## P4 — ЕИС / procurement benchmark

ЕИС не трактуется как интернет-магазин. XML/ZIP/JSON/CSV выгрузки используются как
источник цен реально заключённых закупок. Поддерживается локальный файл и прямой
URL машиночитаемой выгрузки.

Парсер intentionally schema-tolerant: он не привязан к одному namespace/XSD и
извлекает только строки, где есть наименование и положительная unit price либо
пара total/quantity. Для каждой staging-позиции строится:

- `median_rub`;
- `p25_rub`, `p75_rub`;
- `min_rub`, `max_rub`;
- число наблюдений;
- интервал дат;
- уровень сопоставления `identity` или `category`;
- несколько примеров контрактов.

`procurement_benchmark` **никогда не добавляется в `offers[]`** и не меняет цену,
по которой работает расчёт. После refresh supplier-источника старый benchmark
сохраняется, но помечается `needs_refresh=true`.

GUI: `Каталог → Доп. данные → Бенчмарк ЕИС`.

CLI:

```text
python scripts/update_equipment_catalog.py \
  --mode eis-benchmark \
  --input contracts.zip \
  --staging-path data/generated/catalog/catalog_staging.json \
  --eis-manifest data/generated/catalog/eis_runs/run/benchmark_manifest.json
```

## P5 — commercial quote

XLSX/CSV/JSON/YML/XML коммерческого предложения проходит тот же staging mapping,
но получает:

- `price_kind = commercial_quote`;
- стабильный source id по поставщику и номеру КП;
- quote number/date/supplier в provenance;
- `availability = quoted` по умолчанию (можно отключить в GUI).

Политика effective offer теперь: доступность → доверие типа цены → свежесть →
минимальная цена. Поэтому валидное КП имеет приоритет над retail/supplier feed,
даже если у менее доверенного источника дата немного новее.

GUI: `Каталог → Доп. данные → Импорт КП`.

## P6 — browser capture без Playwright

Это fallback для единичной позиции, которой нет в feeds. Приложение **не открывает
URL и не управляет браузером**. Пользователь открывает карточку в обычном браузере
и передаёт приложению:

- сохранённый HTML/исходный код; или
- JSON-LD Product из буфера обмена.

Извлекаются schema.org Product/Offer, identity (brand/model/MPN/GTIN), цена,
валюта, availability и URL. Если JSON-LD отсутствует, допускается осторожный
fallback на OpenGraph/meta. Слабый capture получает warnings и требует review.

GUI: `Каталог → Доп. данные → Захват браузера`.

CLI:

```text
python scripts/update_equipment_catalog.py \
  --mode browser-capture \
  --input saved-product.html \
  --capture-url https://shop.example/product/1 \
  --staging-path data/generated/catalog/catalog_staging.json \
  --output data/generated/catalog/browser_captures/capture.json
```

## Ручная идентичность

Редактор staging теперь позволяет поправить `brand`, `model`, `MPN`, `GTIN`.
Это особенно важно для browser capture и коммерческих предложений: строгая P2
федерация не должна склеивать товары только по похожему названию.

## Зависимости

P4–P6 не добавляют новых runtime-зависимостей. ZIP/XML/CSV/JSON/HTML разбираются
стандартной библиотекой Python. XLSX коммерческого предложения использует уже
существующий staging XLSX reader.
