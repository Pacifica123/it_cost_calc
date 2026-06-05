# Qt workspace layout scaffold fix

Патч стабилизирует layout-контракт рабочих пространств ПО/ТО перед переносом AHP, Pareto и Hybrid.

## Что меняется

- Шаги workspace помещены в `QScrollArea`.
- Верх workspace (`Данные`, stepper, статус) остаётся статичным.
- Текущий шаг прокручивается при нехватке высоты, а не сжимает карточки и поля.
- `CollapsibleSection` больше не управляет высотой через `setMaximumHeight(sizeHint())`.
- `SmartTable` больше не задаёт `maximumHeight` всему контейнеру.
- `EmptyState` получил compact-режим для секций.
- На GA-экране секции `Кандидаты` и `Состав` скрыты до запуска GA.

## Почему это нужно

Серия локальных правок GA-полей ухудшала экран, потому что проблема была системной. В одном вертикальном layout конкурировали summary-карточки, поля, раскрываемые секции, таблицы, empty-state и footer. При нехватке высоты Qt сжимал соседние блоки вместо прокрутки.

Новый контракт меняет поведение: дефицит высоты решается scroll-контейнером, а не ручной подгонкой пикселей.

## Основные файлы

- `src/ui_qt/screens/workspace_data.py`
- `src/ui_qt/screens/decision/ga_screen.py`
- `src/ui_qt/widgets/collapsible_section.py`
- `src/ui_qt/widgets/empty_state.py`
- `src/ui_qt/widgets/smart_table.py`

## Риски

- Визуальный вид секций может немного измениться на уже перенесённых экранах.
- Таблицы внутри секций теперь меньше ограничивают родителя, поэтому часть высот зависит от внешнего scroll-контейнера.
- Полный визуальный smoke не выполнен в текущем окружении без PySide6.

## Проверки

- `python -B tools/check_ui_text_density.py`
- `ast.parse` изменённых Python-файлов
- `pytest -q tests/unit/test_qt_ga_presenter.py`
- полный `pytest` с известным старым падением `test_demo_to_export_flow.py`
