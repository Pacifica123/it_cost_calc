# Qt default switch

## Что изменилось

Обычный запуск приложения открывает PySide6/Qt-интерфейс:

```bash
python scripts/run_app.py
```

После финального удаления legacy runtime старый Tkinter-интерфейс больше не входит в активный код проекта и не доступен как fallback.

## Runtime-пути

| Сценарий | Команда | UI |
|---|---|---|
| Основной запуск | `python scripts/run_app.py` | Qt |
| Qt smoke | `python scripts/run_app.py --smoke-check` | Qt offscreen |
| Прямой Qt запуск | `python scripts/run_qt_app.py` | Qt |

## Границы безопасности

- `src/app.py` не импортирует `tkinter`.
- `src/bootstrap.py` не содержит legacy Tkinter fallback.
- `scripts/run_app.py` не принимает `--legacy-tk`.
- `ttkbootstrap` удалён из runtime-зависимостей.
- Новые изменения должны идти через `src/ui_qt/**`, presenters и application/domain services.

## Проверки

- Основной entrypoint указывает на Qt.
- Tkinter/ttkbootstrap не импортируются активным runtime-кодом.
- Smoke-путь Qt доступен через `--smoke-check`.
- Документация запуска обновлена.
