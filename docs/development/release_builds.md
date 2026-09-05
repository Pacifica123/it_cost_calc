# Release-сборки без установки Python у пользователя

Проект остаётся Python-приложением для разработки, поэтому исходный код поддерживает один явный диапазон интерпретатора: **Python 3.11+**. Ошибка вида
`requires a different Python: 3.10.0 not in '>=3.11'` означает, что локальная среда не подходит для разработки из исходников.

Для пользователей, проверяющих и коллег лучше использовать готовые сборки:

- Windows: `exe` через PyInstaller;
- Linux: `AppImage` поверх PyInstaller onedir-сборки.

## Проверка окружения

```bash
python scripts/doctor.py
```

Дополнительно можно проверить зависимости:

```bash
python scripts/doctor.py --check-deps --check-build-deps
```

## Подготовка build-зависимостей

```bash
pip install -r requirements/build.txt
```

## Windows exe

На Windows с Python 3.11+:

```powershell
python scripts/build_windows_exe.py
```

По умолчанию создаётся onedir-сборка в `dist/windows/ITCostCalc/`. Для одного исполняемого файла:

```powershell
python scripts/build_windows_exe.py --onefile
```

Если нужна консоль для диагностики ошибок запуска:

```powershell
python scripts/build_windows_exe.py --console
```

## Диагностика запуска exe

Qt-экраны и виджеты в проекте подключаются лениво, поэтому build-скрипты явно добавляют внутренние модули `ui_qt.*` в hidden imports PyInstaller. Если после сборки появляется ошибка вида `No module named 'ui_qt.screens...'`, значит использован старый build-скрипт или вручную собранная команда без этих hidden imports.

Для диагностики запуска можно временно собрать приложение с консолью:

```powershell
python scripts/build_windows_exe.py --console
```

## Linux AppImage

На Linux с Python 3.11+:

```bash
bash scripts/build_linux_appimage.sh
```

Скрипт сначала собирает `dist/linux/ITCostCalc/`. Если в `PATH` есть `appimagetool`, дополнительно создаётся `dist/linux/ITCostCalc-x86_64.AppImage`.
Если `appimagetool` не установлен, onedir-сборка остаётся готовой, а для AppImage нужно установить `appimagetool` и повторить команду.

## Playwright в EXE/AppImage

PyInstaller включает Python-пакет и Node-драйвер Playwright, но сами Firefox/Chromium устанавливаются отдельно при первом запуске parser-сценария. В frozen-режиме приложение **не** использует Playwright-режим `PLAYWRIGHT_BROWSERS_PATH=0`, потому что он направляет браузеры внутрь `_internal/playwright/driver/package/.local-browsers`: внутри AppImage этот каталог read-only, а в onefile-сборке он непостоянный.

Вместо этого браузеры хранятся в обычном пользовательском кэше Playwright:

- Linux: `$XDG_CACHE_HOME/ms-playwright` либо `~/.cache/ms-playwright`;
- Windows: `%LOCALAPPDATA%\ms-playwright`;
- macOS: `~/Library/Caches/ms-playwright`.

Поэтому source-run и готовая сборка видят одну и ту же установку браузера. Если пользователь явно задал собственный `PLAYWRIGHT_BROWSERS_PATH`, он сохраняется. Команда `--playwright-install` и фактический Playwright driver получают один и тот же путь, что предотвращает ситуацию «установка успешна, executable не найден».

## Почему не поддерживать Python 3.10

Снижение `requires-python` до 3.10 создаёт риск скрытых проблем с Qt, matplotlib и будущими зависимостями. Для дипломного проекта безопаснее иметь воспроизводимую dev-среду на Python 3.11+ и отдельные release-артефакты для запуска без Python.
