# Каталог оборудования

Каталог — стабильная граница между внешними рыночными данными и расчётным ядром.
Основное приложение не зависит от конкретного интернет-магазина и не требует
успешного scraping витрины.

## Поток данных

```text
XLSX / CSV / YML / XML feeds ─┐
commercial quote ─────────────┼─> federation -> offers[] -> staging
browser capture ──────────────┘                    │
                                                   ├─ Icecat -> specs
EIS contract exports ──────────────────────────────└─ procurement benchmark
                                                        │
                                                        v
                                                  server/client/network
                                                        │
                                                   GA/AHP/NPV/TCO
```

## Источники

- структурированные supplier feeds через `Источник данных`;
- коммерческие предложения через `Доп. данные → Импорт КП`;
- Icecat как независимый источник технических характеристик;
- ЕИС XML/ZIP/JSON/CSV как статистика реально заключённых закупок;
- единичный HTML/JSON-LD capture из обычного браузера как последний fallback;
- legacy DNS/Yandex collectors остаются в CLI для исследований и replay.

## Где лежат данные

- `data/examples/parser/` — сохранённые legacy parser-снимки;
- `data/examples/catalog/` — примеры нормализованных каталогов;
- `data/generated/catalog/catalog_staging.json` — рабочий staging;
- `data/generated/catalog/feed_runs/` — provenance скачанных feeds;
- `data/generated/catalog/icecat_runs/` — manifests enrichment;
- `data/generated/catalog/eis_runs/` — manifests procurement benchmark;
- `data/generated/catalog/browser_captures/` — единичные локальные captures.

## Review boundary

В staging отдельно сохраняются исходная карточка и ручные overrides. Пользователь
может исправить категорию, цену, характеристики и identity (`brand/model/MPN/GTIN`).
Только подтверждённые готовые устройства переходят в ТО.

ЕИС benchmark не является ценой расчёта. Icecat не является ценовым источником.
Это позволяет отдельно оценивать коммерческое предложение, техническую полноту и
рыночную правдоподобность данных.

Подробности: `docs/architecture/catalog_evidence_p456.md`.
