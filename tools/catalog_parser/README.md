# Инструмент обновления каталога оборудования

Этот каталог хранит **отдельный вспомогательный инструмент**, который подготавливает каталог оборудования
для дальнейшего использования в основном приложении.

## Главное правило

`tools/catalog_parser/` **не является частью runtime настольного GUI-приложения**.

Правильная схема такая:
- приложение выполняет расчёты и анализ;
- парсер собирает и нормализует данные об оборудовании;
- результат сохраняется в отдельный каталог данных.

Экран `Каталог` может запустить CLI как отдельный процесс и показать его журнал, но HTML-разбор и Playwright по-прежнему остаются внутри инструмента.

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

### Собрать ограниченный каталог DNS через браузер

```bash
python scripts/update_equipment_catalog.py \
  --mode dns-live \
  --categories routers,prebuilt_pcs,servers \
  --limit 10 \
  --time-limit 300 \
  --snapshot-output data/generated/catalog/dns_runs/manual/snapshot \
  --profile data/generated/catalog/dns_browser_profile \
  --region Москва \
  --output data/generated/catalog/dns_runs/manual/equipment_catalog.json
```

По умолчанию Chromium видим. Первый экран содержит паузу для выбора региона/cookies. Каждый товар сохраняется как HTML, после чего каталог строится каноническим `dns-snapshot` importer. Тот же сценарий доступен кнопкой `Собрать из DNS` на экране каталога.

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
