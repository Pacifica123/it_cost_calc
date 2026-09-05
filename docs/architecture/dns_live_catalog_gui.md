# HTTP-сбор каталога DNS из GUI

## Назначение

Экран `Каталог` запускает сбор DNS отдельным процессом через `QProcess`, показывает журнал и принимает только готовый catalog v2. GUI больше не предлагает Playwright-движок: основной пользовательский сценарий — `dns-http-live`.

```text
Qt dialog
  -> отдельный CLI dns-http-live
  -> requests.Session + обычные cookies
  -> DNS category JSON / AjaxState product-buy
  -> catalog v2
  -> staging review
  -> подтверждение и импорт в ТО
```

## Что перенесено из simple_dns_parser.py

Из предоставленного варианта парсера адаптирована наиболее полезная часть протокола DNS:

- предварительное обращение к `/catalog/markdown/` для формирования обычной сессии;
- извлечение UUID товаров из `AjaxState.register`;
- сохранение пары `hash + оригинальные containers`;
- `POST /ajax-state/product-buy/` с исходными container id;
- разбор `states` в название, цену и идентификатор товара.

Файл нельзя было подключить буквально: он зависит от отсутствующих в проекте `dns_shop_parser.config`, `SessionManager`, моделей и logger. Поэтому сетевой слой адаптирован к уже имеющемуся `requests`, а выход сразу переводится в существующий catalog v2.

Новая сторонняя зависимость не добавлена: `requests` уже входит в базовые зависимости проекта через `pyproject.toml` / `requirements/base.txt`.

## Пользовательский сценарий

1. Открыть `Каталог` и нажать `Собрать из DNS (HTTP)`.
2. Выбрать категории, лимит, таймаут и метку региона.
3. Запустить HTTP-сбор и следить за журналом.
4. При успехе нажать `Загрузить в staging`.
5. Проверить warnings, цену и характеристики перед подтверждением.

Локальный `HAR / HTML` остаётся доступным в том же диалоге как воспроизводимый fallback.

## Диагностика и ограничения

Каждый запуск сохраняется в `data/generated/catalog/dns_http_runs/<timestamp>/`. В `snapshot/` остаются сырые category/product-buy ответы и `snapshot_manifest.json`.

HTTP `401/403/429` и Qrator/challenge не обходятся. Они классифицируются до нормализации и возвращают код `3`. Отсутствие product-buy batch либо пригодных товаров возвращает код `4`. Это позволяет отличить сетевой запрет от изменения формата ответа.

HTTP JSON DNS часто содержит цену и название, но не полный набор технических характеристик. Такие записи получают warning и требуют проверки в staging.

## Playwright

Старый `dns-live` не удалён. Он остаётся доступен напрямую из CLI для диагностики и повторных экспериментов:

```bash
python scripts/update_equipment_catalog.py \
  --mode dns-live \
  --browser-engine firefox \
  --categories routers,switches \
  --limit 10 \
  --time-limit 300 \
  --snapshot-output data/generated/catalog/dns_runs/manual/snapshot \
  --profile data/generated/catalog/dns_browser_profiles/firefox \
  --output data/generated/catalog/dns_runs/manual/equipment_catalog.json
```

GUI этот режим больше не вызывает.
