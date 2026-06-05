# Qt app services adapter

## Назначение

Патч `qt-app-services-adapter` добавляет первую рабочую границу между будущим PySide6/Qt-интерфейсом и существующим application/domain-слоем.

Цель слоя — дать новым Qt-экранам доступ к runtime-данным без `ttk.Treeview`, Tkinter-виджетов и старого `TreeviewCrudRepository`.

## Принципы

- Qt UI не хранит расчётную логику.
- Qt UI не зависит от Tkinter.
- Runtime-данные остаются в `JsonEntityRepository`.
- CRUD идёт через `EntityCatalogService` и `EquipmentService`.
- Табличные модели получают только plain `dict`-строки.
- PySide6 остаётся runtime-зависимостью UI, но adapter-тесты могут идти без GUI.

## Новые элементы

- `QtAppPresenter` — точка входа Qt-экранов к runtime-хранилищу.
- `QtRuntimePaths` — единое описание путей `repo_root`, runtime JSON и demo dataset.
- `EntityTableModel` — модель runtime-строк для будущих таблиц.
- `CandidateTableModel` — модель кандидатов GA/AHP.
- `DecisionTableModel` — модель итогов GA/AHP/Pareto/Hybrid.
- `RecordFormDialog` — заготовка компактного Qt-диалога CRUD.

## Граница ответственности

`ui_qt/presenters` отвечает за подготовку данных для экранов и вызов application services.

`ui_qt/models` отвечает за табличное представление строк.

`ui_qt/dialogs` отвечает только за ввод пользователя и не должен знать о хранилище.

## Что не меняется

- Основной запуск остаётся Tkinter.
- Старый UI не удаляется.
- GA/AHP/Pareto/Hybrid не переносятся в этом патче.
- `TreeviewCrudRepository` остаётся для legacy UI, но новый Qt-слой его не использует.
