# P3 — обогащение технических характеристик Icecat

## Зачем нужен отдельный слой характеристик

P1 получает структурированные российские feeds, P2 объединяет предложения одного
товара и сохраняет несколько ценовых наблюдений. P3 сознательно перестаёт требовать,
чтобы один поставщик одновременно был источником цены, идентичности и полной
технической спецификации.

```text
российские XLSX / CSV / YML       Icecat
        │                           │
        ├─ цена / наличие           ├─ RAM / CPU / storage
        ├─ URL / регион             ├─ сеть / мощность
        └─ brand / MPN / GTIN       └─ стандартизированные features
                    │               │
                    └──── identity ─┘
                           │
                    CatalogItem + staging
                           │
                       ТО / GA / AHP / NPV
```

Цена расчёта по-прежнему выбирается из `offers[]`. Icecat не становится ценовым
источником и не меняет контракт алгоритмов принятия решений.

## Идентификация

Enrichment выполняется только при достаточно строгой идентичности:

1. `GTIN` — приоритетный ключ;
2. `brand + MPN` — fallback.

По одному названию товара запрос не выполняется. После ответа идентичность
проверяется ещё раз: возвращённый GTIN либо пара brand/MPN должны совпасть с
запрошенными. Несовпадение блокирует применение ответа.

## Icecat JSON API

Интеграция использует Product JSON API `https://live.icecat.biz/api` и запрашивает
только `essentialinfo,featuregroups`. Встроенный mapping рассчитан на английские
имена features, поэтому GUI запускает запросы с `lang=EN`.

Параметры запроса:

- `shopname` — логин Icecat;
- `GTIN` либо `Brand + ProductCode`;
- `content=essentialinfo,featuregroups`;
- API token, если он используется аккаунтом, передаётся заголовком `api-token`.

Актуальная документация API:
`https://iceclog.com/manual-for-icecat-json-product-requests/`.

Open Icecat не гарантирует наличие каждой модели. Отсутствие товара или недоступная
для аккаунта карточка считаются нормальным результатом enrichment и не повреждают
staging.

## Секреты

`ICECAT_API_TOKEN` не является настройкой проекта:

- GUI передаёт его только через окружение дочернего процесса;
- токен отсутствует в argv;
- токен не записывается в `catalog_staging.json`;
- токен не записывается в `enrichment_manifest.json`;
- пользовательский source registry его не хранит.

CLI также читает токен только из `ICECAT_API_TOKEN`. Логин можно передать через
`--icecat-username` либо `ICECAT_USERNAME`.

## Политика merge

Icecat применяет стратегию **fill missing only**:

- отсутствующая метрика может быть заполнена;
- существующая метрика поставщика не перезаписывается;
- расхождение сохраняется в `specification_sources[].conflicts`;
- ручной override пользователя остаётся верхним слоем staging.

Это принципиально: обогащение не должно незаметно подменять данные, уже проверенные
пользователем или пришедшие от поставщика.

Поддерживаемые нормализованные поля P3:

- `ram_gb`;
- `cpu_cores`;
- `storage_gb`;
- `max_power_watts`;
- `lan_ports`;
- `lan_speed_mbps`;
- `wifi_total_mbps`;
- `ipv6_support`.

Для каждой реально заполненной метрики сохраняется Feature ID, исходное значение,
единица измерения и timestamp в `field_provenance.specifications`.

## Persistency при refresh

P2 пересобирает federated item при обновлении supplier feed. Поэтому P3 хранит
обогащение отдельно в `specification_sources[]` и повторно применяет этот слой после
перестроения федерации.

Если обновлённый supplier feed сам начинает передавать метрику, supplier-значение
остаётся главным, а старое Icecat-значение становится наблюдаемым конфликтом.

## Review lifecycle

Обогащение меняет исходные данные staging, поэтому ранее `approved` запись снова
становится `pending` и должна быть подтверждена. Уже импортированные в ТО записи P3
не меняет: их изменение без синхронизации runtime создало бы скрытое расхождение.

## GUI и CLI

GUI: `Каталог → Обогатить Icecat`.

- если выбраны строки — обогащаются выбранные;
- без выбора — все доступные записи;
- imported и позиции без GTIN/brand+MPN пропускаются;
- журнал выполнения остаётся видимым пользователю.

CLI:

```text
python scripts/update_equipment_catalog.py \
  --mode icecat-enrich \
  --staging-path data/generated/catalog/catalog_staging.json \
  --icecat-username <login> \
  --icecat-manifest data/generated/catalog/icecat_runs/run/enrichment_manifest.json
```

`ICECAT_API_TOKEN` задаётся только в окружении.

## Зависимости

Новых зависимостей P3 не добавляет. HTTP-клиент использует уже существующий
`requests`. Импорт `requests` ленивый, поэтому portable devctl smoke-check может
проверять mapping/staging только стандартной библиотекой Python.

## Ограничения P3

- полнота зависит от покрытия конкретного Icecat аккаунта;
- mapping намеренно консервативен и использует только известные feature names;
- пока поддержан один specification provider — Icecat, но контракт
  `specification_sources[]` допускает vendor/API adapters в следующих этапах;
- enrichment не выполняется автоматически при каждом feed refresh: сохранённый
  последний слой переносится, а новый сетевой запрос запускает пользователь.
