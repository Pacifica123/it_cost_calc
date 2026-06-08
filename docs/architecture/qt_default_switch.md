# Qt default switch

## Что изменилось

Обычный запуск приложения теперь открывает PySide6/Qt-интерфейс:

```bash
python scripts/run_app.py
```

Старый Tkinter-интерфейс перенесён в legacy-зону и больше не импортируется нормальным runtime-путём Qt.

## Runtime-пути

| Сценарий | Команда | UI |
|---|---|---|
| Основной запуск | `python scripts/run_app.py` | Qt |
| Qt smoke | `python scripts/run_app.py --smoke-check` | Qt offscreen |
| Legacy fallback | `python scripts/run_app.py --legacy-tk` | Tkinter |
| Прямой Qt запуск | `python scripts/run_qt_app.py` | Qt |

## Зачем сохранён legacy fallback

Fallback нужен только на время финальной миграции, чтобы можно было сравнить старый экран с новым при регрессионных ошибках. Он не является основным интерфейсом и не должен использоваться новыми патчами.

## Границы безопасности

- `src/app.py` больше не импортирует `tkinter`.
- `src/bootstrap.py` лениво импортирует legacy UI только при `--legacy-tk`.
- Новый Qt-код не должен зависеть от `src/ui_legacy/`.
- `ttkbootstrap` пока остаётся в зависимостях из-за fallback; его удаление относится к следующему патчу удаления legacy.

## Проверки

- Основной entrypoint указывает на Qt.
- Legacy UI импортируется только явно.
- Smoke-путь Qt доступен через `--smoke-check`.
- Документация запуска обновлена.
