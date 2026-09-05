# Инструмент обновления каталога оборудования

Этот каталог хранит **отдельный вспомогательный инструмент**, который подготавливает каталог оборудования
для дальнейшего использования в основном приложении.

## Главное правило

`tools/catalog_parser/` **не является частью runtime настольного GUI-приложения**.

Правильная схема такая:
- приложение выполняет расчёты и анализ;
- парсер собирает и нормализует данные об оборудовании;
- результат сохраняется в отдельный каталог данных.

Экран `Каталог` запускает отдельный CLI-процесс и показывает его журнал. По умолчанию GUI использует HTTP-режимы без Playwright. Старые Playwright live-режимы сохранены для диагностики и экспериментов, но доступны только из CLI.

## Что лежит в этом каталоге

### Канонические модули инструмента
- `catalog_schema.py` — схема нормализованной записи каталога;
- `catalog_builder.py` — нормализация и сборка итогового JSON-каталога;
- `paths.py` — стандартные пути к примерам и выходным файлам;
- `cli.py` — командный интерфейс инструмента;
- `sources/` — адаптеры источников данных.

### Исследовательская и переходная зона
- `legacy/` — старые и экспериментальные скрипты парсинга, сохранённые как техническая база.

## Как использовать

### Построить каталог из уже сохранённых example-снимков
```bash
python scripts/update_equipment_catalog.py --mode examples
```

Итоговый файл будет сохранён в:
- `data/generated/catalog/equipment_catalog.json`

### Построить каталог из сохранённых DNS HTML-снимков

```bash
python scripts/update_equipment_catalog.py \
  --mode dns-snapshot \
  --input data/examples/parser/dns_snapshot
```

Каталог снимка должен содержать `snapshot_manifest.json` и перечисленные в нём локальные HTML-файлы. Импорт сначала читает Product/Offer JSON-LD, затем использует ограниченный fallback по `h1`, meta и `dt`/`dd`. Сетевые запросы не выполняются.

### Импортировать HAR или HTML из обычного браузера

```bash
python scripts/update_equipment_catalog.py \
  --mode dns-har \
  --input browser-capture.har \
  --region Кемерово \
  --output data/generated/catalog/equipment_catalog.json
```

Для одной сохранённой страницы используйте `--mode dns-html`. В GUI оба варианта доступны кнопкой `Импорт HAR / HTML` в диалоге DNS. HAR обрабатывается только локально: importer читает response bodies DNS, но игнорирует headers, cookies, POST data и пользовательские отзывы. Категорийный HTML объединяется с региональными ценами `product-buy`, microdata и полной PWA-карточкой по UUID товара.

В обычном Chrome откройте `F12 → Network`, включите `Preserve log`, посетите нужные категории и карточки, затем сохраните `HAR (sanitized)`. Передавать sensitive HAR не требуется.

### Собрать каталог DNS через HTTP без Playwright

```bash
python scripts/update_equipment_catalog.py \
  --mode dns-http-live \
  --categories routers,switches,prebuilt_pcs,servers \
  --limit 10 \
  --time-limit 300 \
  --snapshot-output data/generated/catalog/dns_http_runs/manual/snapshot \
  --region Москва \
  --output data/generated/catalog/dns_http_runs/manual/equipment_catalog.json
```

Этот режим является основным live-сценарием GUI. Он использует обычную cookie-сессию `requests`, сначала прогревает DNS и `/catalog/markdown/`, затем извлекает из ответа пары `product-buy hash + оригинальные containers` и запрашивает цены через `/ajax-state/product-buy/`. Логика hash/containers адаптирована из предоставленного `simple_dns_parser.py`; зависимость на внешний пакет `dns_shop_parser` не переносилась.

HTTP 401/403/429 и защитные страницы не обходятся. Сырые ответы и `snapshot_manifest.json` сохраняются в `data/generated/catalog/dns_http_runs/`. Если DNS принимает HTTP-сессию, каталог строится без браузерного движка.

### Playwright-сбор DNS из CLI

```bash
python scripts/update_equipment_catalog.py \
  --mode dns-live \
  --categories routers,switches,prebuilt_pcs,servers \
  --browser-engine firefox \
  --limit 10 \
  --time-limit 300 \
  --snapshot-output data/generated/catalog/dns_runs/manual/snapshot \
  --profile data/generated/catalog/dns_browser_profiles/firefox \
  --region Москва \
  --output data/generated/catalog/dns_runs/manual/equipment_catalog.json
```

По умолчанию используется видимый Firefox; можно выбрать Chromium параметром CLI. Если выбранный Playwright browser engine ещё не установлен, CLI устанавливает его текущим Python-интерпретатором. Этот режим больше не показывается в GUI и остаётся диагностическим fallback для ручных экспериментов.

Все четыре поддерживаемые группы открываются по прямым URL категорий DNS. Если поздняя категория получает `403/429`, уже собранные карточки сохраняются как частичный результат.

Ответы DNS `403/429` распознаются до парсинга ссылок. Сбор останавливается после первого отказа, а причина и URL сохраняются в `snapshot_manifest.json`; смена селекторов в этой ситуации не поможет.

### Собрать Яндекс Маркет через HTTP без Playwright

```bash
python scripts/update_equipment_catalog.py \
  --mode yandex-market-http-live \
  --categories routers,switches,prebuilt_pcs,servers \
  --limit 10 \
  --time-limit 300 \
  --snapshot-output data/generated/catalog/yandex_market_http_runs/manual/snapshot \
  --region Москва \
  --output data/generated/catalog/yandex_market_http_runs/manual/equipment_catalog.json
```

GUI использует этот режим для Яндекс Маркета. Обычная `requests`-сессия получает category/card HTML, а дальнейший разбор выполняет уже существующий replayable snapshot-parser. CAPTCHA/401/403/429 классифицируются и сохраняются в диагностике; обход защитных механизмов не реализуется.

### Playwright-сбор Яндекс Маркета из CLI

```bash
python scripts/update_equipment_catalog.py \
  --mode yandex-market-live \
  --categories routers,switches,prebuilt_pcs,servers \
  --browser-engine firefox \
  --limit 10 \
  --time-limit 300 \
  --snapshot-output data/generated/catalog/yandex_market_runs/manual/snapshot \
  --profile data/generated/catalog/yandex_market_browser_profiles/firefox \
  --region Москва \
  --output data/generated/catalog/yandex_market_runs/manual/equipment_catalog.json
```

Режимы `yandex-market-snapshot`, `yandex-market-har` и `yandex-market-html` обеспечивают повторный офлайн-разбор. Общедоступный API каталога не используется: seller API относится к кабинету продавца, а новые ключи старого Content API не выдаются. Подробности, ограничения и критерий теста: `docs/architecture/yandex_market_catalog_parser.md`.

Playwright live-сбор теперь оставлен только в CLI. GUI-кнопка `Собрать из Яндекс Маркета (HTTP)` запускает `yandex-market-http-live`. HAR importer по-прежнему доступен из GUI как локальный fallback и не переносит headers, cookies, POST data, изображения или отзывы.

## Для чего это нужно

Полученный каталог можно открыть на экране **«Каталог»**. GUI не занимается
HTML-разбором: он читает готовый JSON, помещает строки в staging и переносит в
ТО только явно подтверждённые готовые устройства.

Новые каталоги создаются по schema v2. Staging сохраняет совместимость с v1.


### Best-effort извлечение сетевых характеристик

При нормализации роутеров из DNS example-снимков инструмент пытается извлечь из названия товара:
- количество LAN-портов;
- LAN-скорость;
- стандарты Wi-Fi и условное поколение;
- суммарную Wi-Fi скорость;
- поддержку IPv6.

Результат сохраняется в `attributes` вместе с `parsed_metrics`, `parse_warnings`, `confidence` и `parse_source`. Ручные поля из `specs` имеют приоритет над распознанными значениями.

## Где лежат связанные данные

- сырые example-снимки: `data/examples/parser/`
- synthetic-пример офлайн-снимка: `data/examples/parser/dns_snapshot/`
- пример нормализованного каталога: `data/examples/catalog/normalized_dns_catalog.json`
- рабочий выходной каталог: `data/generated/catalog/`
