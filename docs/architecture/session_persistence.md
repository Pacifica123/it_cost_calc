# Сессии и каноническое runtime-хранение

## Цель

`Session` — самодостаточная, возобновляемая и копируемая капсула рабочего состояния для внешнего контекста: защиты, обсуждения, эксперимента или обычной работы. Загрузка сессии должна восстанавливать уже введённые данные, промежуточный аналитический pipeline и созданные экспортные артефакты без обязательного повторного ввода или пересчёта.

Каноническое mutable runtime-хранилище — `data/generated/runtime.sqlite3`. JSON остаётся для fixture/interchange и однократного импорта старого `runtime_entities.json`.

## State graph

| Класс состояния | Что относится | Хранение |
|---|---|---|
| Global persistent | переиспользуемые ПО/ТО/расходы/`solution_components` и прочие runtime entities | `global_state` в SQLite |
| Session-owned | deep-copy entities, workflow position, candidate pools, параметры и результаты | `sessions.state_json` |
| Derived/computed | GA, AHP, Pareto, Hybrid, electricity, NPV и их provenance/validity | versioned JSON payload активной Session |
| Export artifacts | CSV/JSON/Markdown/HTML и другие уже созданные результаты | immutable BLOB + metadata в `export_artifacts` |
| Cache/projection | `ScopedCandidatePoolService`, catalog staging SQLite read-model | восстанавливаемая RAM/read-model, не source of truth |
| Preferences | тема, плотность, прочие UI settings | отдельные global settings; не Session |
| Secrets/environment | env, ключи, ОС-зависимые пути | никогда не сериализуются в Session |

Nested domain state намеренно не пере-нормализован: SQLite хранит relational core (identity, relations, revisions, export metadata) и versioned JSON payload с состоянием сессии. Pickle не используется.

## Изоляция

Глобальные сохранённые данные и жизненный цикл сессий разделены:

```text
create/import Session = deep snapshot(global/current data)
mutate(global) != mutate(existing sessions)
mutate(session) != mutate(global)
fork(S1) -> S2; future(S1) independent from future(S2)
```

Обратная синхронизация выполняется только явной операцией. Дублирование сессии копирует и state, и историю экспортных артефактов в новые строки с собственными identity.

## Revision / validity

У Session есть `revision` и `input_revision`.

- любое meaningful сохранение увеличивает `revision`;
- изменение исходных entities увеличивает `input_revision`;
- derived state хранит `validity`, `input_revision`, `produced_from_session_revision`, `parameters`, `provenance`, `produced_at`;
- изменение входов переводит существующие зависимые результаты в `stale`, но не удаляет их;
- UI и новый экспорт читают только `current` результаты.

Так сохраняется воспроизводимость старого результата, но устаревший расчёт не выдаётся за актуальный.

## Partial pipeline и restart

Сессия сохраняет не только итоговый Hybrid/DecisionReport, но и промежуточные стадии: scoped GA result, candidate pool, AHP/Pareto/Hybrid, energy profile/result, NPV basis/input/result и workflow position. RAM-кэш candidate pool восстанавливается из SQLite после запуска и переключения Session.

Следствие: сценарий `Data -> GA -> AHP -> закрыть приложение -> открыть Session -> продолжить` не требует повторного GA.

## Экспорт

Экспорт — immutable session-owned artifact. Для каждого файла SQLite хранит:

- type/kind и format;
- имя и время;
- revision Session на момент создания;
- SHA-256 payload;
- metadata/provenance;
- полный BLOB payload.

`data/generated/reports/<session-id>/` — только материализация. Если файл удалён, его можно восстановить из БД без исходного пути. Старые exports не меняются при дальнейшем редактировании Session.

## Legacy migration

При первом запуске с существующим `data/generated/runtime_entities.json`:

1. файл читается как legacy source;
2. entities атомарно импортируются в `global_state` и отдельную мигрированную Session;
3. hash/path записываются в `legacy_imports`;
4. тот же source больше не импортируется повторно;
5. приложение больше не пишет runtime JSON.

Исходный файл не удаляется автоматически и остаётся резервным legacy/interchange артефактом.

## Writable runtime root

Путь SQLite строится от существующего `writable_runtime_root()`. Поэтому source checkout и packaged/frozen запуск используют один контракт: writable runtime state не пытается записываться в read-only resource root.

## UX boundary

`QtAppPresenter` является session-context boundary. Экраны не ветвятся по Session ID: они получают активные entities/results через presenter/repository. Диалог «Сессии» предоставляет create/open/switch/rename/duplicate/delete, а заголовок окна показывает активную Session.

Архитектура допускает будущий portable session export/import: state payload и export BLOB уже не зависят от materialized filesystem paths.
