# Исследовательский сбор каталога Яндекс Маркета

## Решение по источнику

На 2026-08-22 у Яндекс Маркета есть два разных класса API:

- [API Маркета для продавцов](https://yandex.ru/dev/market/partner-api/doc/ru/) работает с ассортиментом, ценами, заказами и отчётами кабинета продавца. Он требует API-Key, созданный в кабинете, и не является публичным поиском по общей витрине;
- юридические документы всё ещё описывают [Контентный API](https://yandex.ru/legal/market_api_content/ru/), но условия расширенного доступа прямо указывают, что с 26.12.2018 предложение отозвано для новых пользователей и новые ключи не выдаются: [условия услуги](https://yandex.ru/legal/market_api_content_conditions/ru/).

Поэтому патч не требует токен продавца и не использует недокументированные внутренние JSON endpoint'ы Маркета как контракт. Исследуемый источник — публичная веб-витрина, открываемая пользователем через Playwright. Это best-effort интеграция, а не стабильный официальный data feed.

## Контур

```text
Экран «Каталог»
  -> отдельный CLI yandex-market-live
  -> persistent Playwright profile
  -> публичные category/card HTML
  -> локальный replayable snapshot
  -> нормализация catalog v2
  -> существующий staging и ручное подтверждение
```

Поддерживаются группы:

- `routers` — `https://market.yandex.ru/category/routery`;
- `switches` — `https://market.yandex.ru/category/kommutatory`;
- `prebuilt_pcs` — `https://market.yandex.ru/category/gotovyye-kompyutery`;
- `servers` — `https://market.yandex.ru/category/servernyye-kompyutery`.

Collector принимает только HTTPS-ссылки хоста `market.yandex.ru`, убирает tracking query, ограничивает число карточек и общий таймаут. Между карточками есть пауза. Сырые HTML и manifest остаются в `data/generated/catalog/yandex_market_runs/`.

## Извлечение

Карточка разбирается слоями:

1. schema.org Product/Offer JSON-LD;
2. встроенный JSON страницы с выбором product-like узла по идентификатору карточки;
3. `h1`, canonical/meta и ограниченный набор видимых пар «характеристика — значение»;
4. best-effort цена из публично отображаемого текста.

Результат получает `source=yandex_market`, собственный provenance, URL, регион, время наблюдения и warnings. Нормализация технических величин использует тот же канонический слой, что DNS, но источник не подменяется значением `dns`.

## Ошибки доступа и fallback

HTTP `401/403/429`, `showcaptcha` и SmartCaptcha классифицируются до разбора карточек и записываются в manifest. Уже собранные карточки сохраняются как частичный результат.

Режимы `yandex-market-har` и `yandex-market-html` работают полностью локально. HAR importer читает только успешные HTML response bodies `market.yandex.ru`; headers, cookies, POST data, изображения и отзывы игнорируются. Максимальный HAR — 256 МБ, отдельный HTML body — 24 МБ.

Наличие fallback не означает, что live-сбор обязательно будет заблокирован: это определяется реальным подключением, регионом и текущей защитой Маркета при пользовательском тесте.

## Команды

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

Повторный офлайн-разбор:

```bash
python scripts/update_equipment_catalog.py \
  --mode yandex-market-snapshot \
  --input data/generated/catalog/yandex_market_runs/manual/snapshot \
  --output data/generated/catalog/yandex_market_runs/manual/replayed_catalog.json
```

Локальный capture:

```bash
python scripts/update_equipment_catalog.py \
  --mode yandex-market-har \
  --input browser-capture.har \
  --region Москва \
  --output data/generated/catalog/yandex_market_imports/manual/equipment_catalog.json
```

## Критерий пользовательского теста

Потенциал источника считается подтверждённым, если хотя бы для двух категорий live-сбор формирует catalog v2, где у большинства строк присутствуют:

- стабильный URL и `source_product_id`;
- текущая цена и регион;
- минимум три предметные характеристики;
- отсутствие блокирующих ошибок staging после проверки пользователем.

Если live-сбор завершится кодом `3` или `4`, для вывода о причине нужно смотреть `snapshot_manifest.json` и сохранённый listing HTML, а не только факт пустого каталога.
