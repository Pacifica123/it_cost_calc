# Каталог Яндекс Маркета: HTTP по умолчанию, Playwright только CLI

## Назначение

GUI использует `yandex-market-http-live`: отдельный процесс без Playwright получает публичные category/card HTML через обычную cookie-сессию `requests`, сохраняет replayable snapshot и передаёт его существующему нормализатору catalog v2.

```text
Экран «Каталог»
  -> yandex-market-http-live
  -> requests.Session
  -> category/card HTML
  -> snapshot_manifest.json
  -> существующий Yandex Market snapshot parser
  -> catalog v2
  -> staging
```

Поддерживаются группы `routers`, `switches`, `prebuilt_pcs`, `servers` с теми же публичными URL категорий, что и прежний browser collector.

## Извлечение

Карточки разбираются прежним проверенным слоем:

1. schema.org Product/Offer JSON-LD;
2. встроенный JSON страницы;
3. `h1`, canonical/meta и видимые пары характеристик;
4. best-effort цена из публичного текста.

HTTP collector отвечает только за сессию, получение страниц, лимиты, таймауты и сохранение диагностики. Парсер карточки не дублируется.

## Ошибки доступа

HTTP `401/403/429`, `showcaptcha` и SmartCaptcha фиксируются в `snapshot_manifest.json`. Автоматический обход защитных механизмов не реализуется. Уже полученные карточки могут дать частичный результат.

Рабочие каталоги GUI:

- `data/generated/catalog/yandex_market_http_runs/<timestamp>/` — HTTP live;
- `data/generated/catalog/yandex_market_imports/<timestamp>/` — локальный HAR/HTML.

## Команда HTTP-сбора

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

## Playwright fallback

Предыдущий `yandex-market-live` сохранён, но GUI его не показывает. Он остаётся CLI-only исследовательским режимом для случаев, когда нужно сравнить результат HTTP и браузерной сессии.

```bash
python scripts/update_equipment_catalog.py \
  --mode yandex-market-live \
  --browser-engine firefox \
  --categories routers,switches \
  --limit 10 \
  --time-limit 300 \
  --snapshot-output data/generated/catalog/yandex_market_runs/manual/snapshot \
  --profile data/generated/catalog/yandex_market_browser_profiles/firefox \
  --output data/generated/catalog/yandex_market_runs/manual/equipment_catalog.json
```

## Зависимости

Отдельная HTTP-библиотека не добавлялась: используется уже существующий `requests`. Поэтому `requirements/base.txt` продолжает подключать проект через `-e .`, а список runtime-зависимостей остаётся в `pyproject.toml` без нового пакета.
