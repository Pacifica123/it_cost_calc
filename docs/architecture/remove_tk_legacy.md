# Remove Tk legacy

## Итог

Патч завершает миграцию UI: активный desktop-интерфейс проекта теперь только PySide6/Qt. Tkinter/ttkbootstrap runtime, Treeview-CRUD и совместимые wrappers старого UI удалены.

## Что удалено

- `src/ui/` — старые Tkinter tabs/widgets/dialogs.
- `src/ui_legacy/` — временный fallback после переключения на Qt.
- `src/it_cost_calc/ui/` — namespace wrappers старого UI.
- `TreeviewCrudRepository` и связанные `crud_compat`/`legacy_crud` алиасы.
- runtime-зависимость `ttkbootstrap`.
- CLI-флаг `--legacy-tk`.

## Что остаётся

- `src/ui_qt/` — единственный активный desktop UI.
- `src/application/`, `src/domain/`, `src/infrastructure/`, `src/shared/` — расчётная и прикладная логика.
- `src/it_cost_calc/` — совместимость package namespace для не-UI модулей.

## Проверочный контракт

- `python scripts/run_app.py` запускает Qt.
- `src/app.py` и `src/bootstrap.py` не импортируют Tkinter.
- В активном `src/` нет `ttkbootstrap` и `ttk.Treeview`.
- UI-density scan проверяет только `src/ui_qt`.
