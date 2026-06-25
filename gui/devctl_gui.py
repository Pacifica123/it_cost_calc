from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Callable

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6 import QtSvg
except Exception as exc:  # Qt не нужен в --devctl-child, поэтому GUI проверяет это в main().
    QtCore = None  # type: ignore[assignment]
    QtGui = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    QtSvg = None  # type: ignore[assignment]
    QT_IMPORT_ERROR: Exception | None = exc
else:
    QT_IMPORT_ERROR = None

try:
    from devctl_runner import DevctlRunner, RunResult
except ImportError:  # запуск как python -m gui.devctl_gui
    from .devctl_runner import DevctlRunner, RunResult  # type: ignore

APP_NAME = "devctl GUI"
APP_VERSION = "0.5.4"
BUNDLED_DEVCTL_VERSION = "0.7.0"

PATCH_PROMPT_TEMPLATE = """Ты работаешь с devctl workspace и должен вернуть не полный архив проекта, а полноценный devctl-патч.

Контекст devctl:
- workspace содержит project/, patches/, archives/, UserTestSpace/ и .devctl/;
- devctl применяет patch.zip из patches/ к папке project/;
- патч должен быть безопасным, воспроизводимым и понятным человеку;
- GUI/CLI ожидают структуру patch.zip с manifest.json, PATCH_SUMMARY.md и files/.
- devctl умеет reset, init --upgrade, автооткат failed start, UTS, Patch Intake (`inbox`) и автоочистку Python bytecode/cache.

Твоя задача:
1. Изучи текущие файлы проекта, которые нужно менять. Не придумывай содержимое вслепую.
2. Реализуй изменение минимально и аккуратно.
3. Собери devctl patch.zip, а не весь проект.
4. Проверь патч хотя бы синтаксически и, если возможно, через devctl plan/start на временном workspace.
5. В ответе дай ссылку на patch.zip, SHA-256, краткое описание и список проверок.

Обязательная структура архива:

```text
patch_YYYYMMDD_HHMMSS_short_slug.zip
  manifest.json
  PATCH_SUMMARY.md
  files/
    relative/path/in/project.ext
```

Правила для files/:
- пути внутри files/ должны быть относительными к project/;
- не клади абсолютные пути;
- не клади .git/, .env, секреты, __pycache__/, *.pyc, *.pyo, .pytest_cache/, .venv/, dist/, build/, node_modules/;
- если devctl копирует целые файлы, клади в files/ уже финальные версии изменённых файлов;
- по возможности заполняй manifest.target: projectId/workspaceId/projectName и expectedFiles, чтобы `devctl inbox grab` мог сам доставить patch.zip в правильный workspace;
- не меняй unrelated-файлы ради косметики.

Минимальный manifest.json:

```json
{
  "formatVersion": 1,
  "patchId": "YYYY-MM-DD-short-slug",
  "title": "Короткое название патча",
  "summary": "Что делает патч и зачем.",
  "kind": "feature-or-fix",
  "createdAt": "YYYY-MM-DDTHH:MM:SSZ",
  "base": {
    "branch": "main",
    "expectedHead": null
  },
  "target": {
    "projectId": "devctl",
    "workspaceId": "devctl",
    "projectName": "devctl universal",
    "expectedFiles": ["devctl.py", "README.md"]
  },
  "apply": {
    "filesRoot": "files",
    "delete": []
  },
  "checks": [
    {
      "name": "Python syntax без генерации __pycache__",
      "cwd": ".",
      "command": "python -c \"import ast,pathlib; files=['devctl.py']; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8'), filename=p) for p in files]\"",
      "requiredCommands": ["python"],
      "timeoutSeconds": 120
    }
  ],
  "commit": {
    "message": "feat: кратко описать изменение"
  },
  "push": {
    "remote": "origin",
    "branch": "main"
  },
  "archive": {
    "nameSlug": "short-slug",
    "exclude": [
      ".git/",
      ".venv/",
      "node_modules/",
      "target/",
      "dist/",
      "build/",
      "coverage/",
      "__pycache__/",
      ".pytest_cache/",
      ".env",
      ".env.*",
      "*.sqlite",
      "*.db"
    ]
  }
}
```

Для Python-проектов предпочитай проверку через ast.parse, а не py_compile, чтобы проверка не создавала __pycache__.

PATCH_SUMMARY.md должен объяснять:
- что меняется;
- зачем это нужно;
- основные файлы;
- риски;
- проверки;
- особые инструкции применения, если они есть.

Финальный ответ пользователю должен быть коротким:
- ссылка на patch.zip;
- SHA-256;
- что меняется;
- какие проверки прогнаны;
- если что-то не удалось проверить — честно указать.
"""


def configure_standard_streams() -> None:
    """Принудительно держим UTF-8 для child-процессов на Windows."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def bundled_root() -> Path:
    """Где лежат devctl.py и bundled-ресурсы."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)).resolve()
    return Path(__file__).resolve().parents[1]


def app_dir() -> Path:
    """Каталог exe/скрипта, видимый пользователю."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def default_workspace_path() -> Path:
    base = app_dir()
    if base.name.lower() == "release":
        return base.parent
    return base


def repo_root() -> Path:
    return bundled_root()


def run_devctl_child(argv: list[str]) -> int:
    if len(argv) < 2:
        print("[ОШИБКА] child-режим ожидает: --devctl-child <workspace> <devctl args...>", file=sys.stderr)
        return 2
    workspace = Path(argv[0]).expanduser().resolve()
    os.environ["DEVCTL_WORKSPACE"] = str(workspace)
    devctl_args = argv[1:]
    if devctl_args and devctl_args[0] == "init":
        workspace.mkdir(parents=True, exist_ok=True)
    os.chdir(workspace)
    configure_standard_streams()
    root = bundled_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import devctl
    except Exception as exc:
        print(f"[ОШИБКА] Не удалось импортировать devctl.py: {exc}", file=sys.stderr)
        return 2
    return int(devctl.main(devctl_args))


def app_config_path() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "devctl-gui" / "config.json"
    return Path.home() / ".config" / "devctl-gui" / "config.json"


def looks_like_pyinstaller_temp(path: Path) -> bool:
    return any(part.upper().startswith("_MEI") for part in path.parts)


def devctl_global_config_path() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "devctl" / "config.json"
    return Path.home() / ".config" / "devctl" / "config.json"


def load_devctl_global_config() -> dict:
    path = devctl_global_config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def registered_workspaces() -> list[dict]:
    records = load_devctl_global_config().get("workspaces")
    if not isinstance(records, list):
        return []
    result: list[dict] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        path = str(record.get("path") or "").strip()
        if not path:
            continue
        try:
            normalized = str(Path(path).expanduser().resolve()).casefold()
        except Exception:
            normalized = path.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(record)
    return result


def initial_workspace(config_data: dict) -> str:
    configured = config_data.get("lastWorkspace")
    if configured:
        try:
            candidate = Path(str(configured)).expanduser().resolve()
            if candidate.exists() and not looks_like_pyinstaller_temp(candidate):
                return str(candidate)
        except Exception:
            pass
    return str(default_workspace_path())


def load_config() -> dict:
    path = app_config_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(data: dict) -> None:
    path = app_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def open_path(path: str | Path | None) -> None:
    if not path:
        _show_info("Путь пока неизвестен.")
        return
    target = Path(path)
    try:
        if os.name == "nt":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except Exception as exc:
        _show_error(f"Не удалось открыть путь:\n{target}\n\n{exc}")


def _show_info(text: str, title: str = APP_NAME) -> None:
    if QtWidgets is not None:
        QtWidgets.QMessageBox.information(None, title, text)
    else:
        print(text)


def _show_error(text: str, title: str = APP_NAME) -> None:
    if QtWidgets is not None:
        QtWidgets.QMessageBox.critical(None, title, text)
    else:
        print(text, file=sys.stderr)


def _confirm(text: str, title: str = APP_NAME) -> bool:
    if QtWidgets is None:
        return False
    answer = QtWidgets.QMessageBox.question(
        None,
        title,
        text,
        QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        QtWidgets.QMessageBox.StandardButton.No,
    )
    return answer == QtWidgets.QMessageBox.StandardButton.Yes


class StatusCard(QtWidgets.QFrame if QtWidgets else object):
    def __init__(self, title: str) -> None:
        if QtWidgets is None:
            raise RuntimeError("PySide6 не импортирован")
        super().__init__()
        self.setObjectName("StatusCard")
        self.setProperty("cardKind", "neutral")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("CardTitle")
        self.value_label = QtWidgets.QLabel("—")
        self.value_label.setObjectName("CardValue")
        self.value_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label, 1)

    def set(self, text: str, kind: str = "neutral") -> None:
        self.setProperty("cardKind", kind)
        self.value_label.setProperty("kind", kind)
        self.value_label.setText(text)
        for widget in (self, self.value_label):
            widget.style().unpolish(widget)
            widget.style().polish(widget)


class InitWorkspaceDialog(QtWidgets.QDialog if QtWidgets else object):
    """Modal-dialog для создания нового devctl workspace."""

    def __init__(self, parent, *, initial_parent: Path, initial_name: str = "devctl-workspace") -> None:
        if QtWidgets is None:
            raise RuntimeError("PySide6 не импортирован")
        super().__init__(parent)
        self.setWindowTitle("Инициализация workspace")
        self.setModal(True)
        self.setMinimumWidth(620)
        self.result: dict[str, str] | None = None

        body = QtWidgets.QVBoxLayout(self)
        body.setContentsMargins(20, 20, 20, 20)
        body.setSpacing(12)

        title = QtWidgets.QLabel("Новый workspace")
        title.setObjectName("DialogTitle")
        body.addWidget(title)
        intro = QtWidgets.QLabel(
            "GUI создаст папку workspace и структуру devctl. Если указан GitHub/Git URL, "
            "project/ будет клонирован или синхронизирован через fetch/pull."
        )
        intro.setWordWrap(True)
        body.addWidget(intro)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.parent_edit = QtWidgets.QLineEdit(str(initial_parent))
        choose_btn = QtWidgets.QPushButton("Выбрать...")
        choose_btn.clicked.connect(self.choose_parent)
        parent_row = QtWidgets.QHBoxLayout()
        parent_row.addWidget(self.parent_edit, 1)
        parent_row.addWidget(choose_btn)
        form.addRow("Куда поместить workspace:", parent_row)
        self.name_edit = QtWidgets.QLineEdit(initial_name)
        form.addRow("Название workspace:", self.name_edit)
        self.remote_edit = QtWidgets.QLineEdit("")
        form.addRow("GitHub/Git remote URL, необязательно:", self.remote_edit)
        self.branch_edit = QtWidgets.QLineEdit("main")
        self.branch_edit.setMaximumWidth(180)
        form.addRow("Основная ветка:", self.branch_edit)
        body.addLayout(form)

        note = QtWidgets.QLabel(
            "Если URL задан, GUI загрузит существующий remote-проект в project/ и явно выполнит "
            "fetch/pull выбранной ветки."
        )
        note.setWordWrap(True)
        note.setObjectName("MutedLabel")
        body.addWidget(note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Создать и открыть")
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self.accept_request)
        buttons.rejected.connect(self.reject)
        body.addWidget(buttons)

    def choose_parent(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку, внутри которой создать workspace")
        if selected:
            self.parent_edit.setText(selected)

    @staticmethod
    def _validate_folder_name(name: str) -> str | None:
        value = name.strip()
        if not value:
            return "Название workspace не должно быть пустым."
        if value in {".", ".."}:
            return "Название workspace не должно быть '.' или '..'."
        if any(char in value for char in '<>:"/\\|?*'):
            return "Название workspace не должно содержать символы: < > : \" / \\ | ? *"
        return None

    def accept_request(self) -> None:
        parent = self.parent_edit.text().strip()
        name = self.name_edit.text().strip()
        branch = self.branch_edit.text().strip() or "main"
        error = self._validate_folder_name(name)
        if error:
            QtWidgets.QMessageBox.critical(self, APP_NAME, error)
            return
        if not parent:
            QtWidgets.QMessageBox.critical(self, APP_NAME, "Выберите родительскую директорию для workspace.")
            return
        self.result = {
            "parent": parent,
            "name": name,
            "remoteUrl": self.remote_edit.text().strip(),
            "branch": branch,
        }
        self.accept()


class WorkspaceChoiceDialog(QtWidgets.QDialog if QtWidgets else object):
    """Modal-dialog для ручного выбора workspace при неоднозначном Patch Intake."""

    def __init__(self, parent, *, workspaces: list[dict]) -> None:
        if QtWidgets is None:
            raise RuntimeError("PySide6 не импортирован")
        super().__init__(parent)
        self.setWindowTitle("Выбор workspace для патча")
        self.setModal(True)
        self.resize(680, 360)
        self.result: str | None = None
        self.workspaces = workspaces
        self.buttons: list[QtWidgets.QRadioButton] = []

        body = QtWidgets.QVBoxLayout(self)
        body.setContentsMargins(20, 20, 20, 20)
        body.setSpacing(12)
        title = QtWidgets.QLabel("Patch target is unclear")
        title.setObjectName("DialogTitle")
        body.addWidget(title)
        note = QtWidgets.QLabel(
            "devctl не смог уверенно выбрать workspace. Укажите, куда импортировать последний patch.zip из Patch Inbox."
        )
        note.setWordWrap(True)
        body.addWidget(note)

        group = QtWidgets.QGroupBox("Workspace")
        group_layout = QtWidgets.QVBoxLayout(group)
        for index, record in enumerate(workspaces):
            label = f"{record.get('id') or '(без id)'} · {record.get('name') or ''}\n{record.get('path') or ''}"
            radio = QtWidgets.QRadioButton(label)
            radio.setProperty("workspaceId", str(record.get("id") or ""))
            if index == 0:
                radio.setChecked(True)
            self.buttons.append(radio)
            group_layout.addWidget(radio)
        body.addWidget(group, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Импортировать сюда")
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self.accept_request)
        buttons.rejected.connect(self.reject)
        body.addWidget(buttons)

    def accept_request(self) -> None:
        for button in self.buttons:
            if button.isChecked():
                self.result = str(button.property("workspaceId") or "")
                break
        if not self.result:
            QtWidgets.QMessageBox.critical(self, APP_NAME, "Выберите workspace.")
            return
        self.accept()



class PatchInboxSettingsDialog(QtWidgets.QDialog if QtWidgets else object):
    """Настройки глобального Patch Inbox и регистрации текущего workspace."""

    def __init__(self, parent, *, workspace_path: Path, scan_data: dict) -> None:
        if QtWidgets is None:
            raise RuntimeError("PySide6 не импортирован")
        super().__init__(parent)
        self.setWindowTitle("Настройки Patch Inbox")
        self.setModal(True)
        self.resize(760, 430)
        self.workspace_path = workspace_path
        self.scan_data = scan_data if isinstance(scan_data, dict) else {}
        self.result: dict[str, object] | None = None

        body = QtWidgets.QVBoxLayout(self)
        body.setContentsMargins(20, 20, 20, 20)
        body.setSpacing(12)

        title = QtWidgets.QLabel("Patch Inbox и workspace registry")
        title.setObjectName("DialogTitle")
        body.addWidget(title)
        intro = QtWidgets.QLabel(
            "Здесь можно явно указать общий склад patch.zip и сразу зарегистрировать текущий workspace, "
            "чтобы кнопка «Забрать патч» работала без ручных команд в терминале."
        )
        intro.setWordWrap(True)
        body.addWidget(intro)

        config_path = self.scan_data.get("configPath") or "будет создан автоматически"
        config_label = QtWidgets.QLabel(f"Глобальный config: {config_path}")
        config_label.setObjectName("MutedLabel")
        config_label.setWordWrap(True)
        body.addWidget(config_label)

        inbox_group = QtWidgets.QGroupBox("Склад патчей")
        inbox_layout = QtWidgets.QVBoxLayout(inbox_group)
        inbox_layout.setSpacing(8)
        dirs = [str(item) for item in (self.scan_data.get("patchInboxDirs") or []) if str(item).strip()]
        initial_inbox = dirs[0] if dirs else str((Path.home() / "PatchInbox").resolve())
        inbox_row = QtWidgets.QHBoxLayout()
        self.inbox_edit = QtWidgets.QLineEdit(initial_inbox)
        choose_inbox = QtWidgets.QPushButton("Выбрать...")
        choose_inbox.clicked.connect(self.choose_inbox)
        inbox_row.addWidget(self.inbox_edit, 1)
        inbox_row.addWidget(choose_inbox)
        inbox_layout.addLayout(inbox_row)
        dirs_note = QtWidgets.QLabel("Текущие склады: " + ("; ".join(dirs) if dirs else "не настроены"))
        dirs_note.setObjectName("MutedLabel")
        dirs_note.setWordWrap(True)
        inbox_layout.addWidget(dirs_note)
        body.addWidget(inbox_group)

        workspace_group = QtWidgets.QGroupBox("Регистрация текущего workspace")
        workspace_layout = QtWidgets.QFormLayout(workspace_group)
        workspace_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        registered_record = self._registered_workspace_record()
        self.register_check = QtWidgets.QCheckBox("Зарегистрировать или обновить текущий workspace в глобальном config")
        self.register_check.setChecked(registered_record is None)
        workspace_layout.addRow(self.register_check)
        self.workspace_path_label = QtWidgets.QLabel(str(workspace_path))
        self.workspace_path_label.setWordWrap(True)
        workspace_layout.addRow("Workspace:", self.workspace_path_label)
        default_id = str((registered_record or {}).get("id") or self._suggest_workspace_id(workspace_path))
        default_name = str((registered_record or {}).get("name") or workspace_path.name or default_id)
        self.workspace_id_edit = QtWidgets.QLineEdit(default_id)
        self.workspace_id_edit.setMaximumWidth(220)
        self.workspace_name_edit = QtWidgets.QLineEdit(default_name)
        workspace_layout.addRow("ID:", self.workspace_id_edit)
        workspace_layout.addRow("Name:", self.workspace_name_edit)
        body.addWidget(workspace_group)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Сохранить настройки")
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self.accept_request)
        buttons.rejected.connect(self.reject)
        body.addWidget(buttons)

    def choose_inbox(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку Patch Inbox")
        if selected:
            self.inbox_edit.setText(selected)

    def _registered_workspace_record(self) -> dict | None:
        current = str(self.workspace_path.resolve())
        for record in self.scan_data.get("workspaces") or []:
            if not isinstance(record, dict):
                continue
            raw = record.get("path")
            if not raw:
                continue
            try:
                candidate = str(Path(str(raw)).expanduser().resolve())
            except Exception:
                candidate = str(raw)
            if os.path.normcase(candidate) == os.path.normcase(current):
                return record
        return None

    @staticmethod
    def _suggest_workspace_id(path: Path) -> str:
        raw = path.name or "workspace"
        value = "".join(ch.lower() if ch.isalnum() else "-" for ch in raw).strip("-")
        while "--" in value:
            value = value.replace("--", "-")
        return value or "workspace"

    def accept_request(self) -> None:
        inbox_path = self.inbox_edit.text().strip()
        if not inbox_path:
            QtWidgets.QMessageBox.critical(self, APP_NAME, "Укажите путь до склада Patch Inbox.")
            return
        workspace_id = self.workspace_id_edit.text().strip()
        workspace_name = self.workspace_name_edit.text().strip()
        if self.register_check.isChecked() and not workspace_id:
            QtWidgets.QMessageBox.critical(self, APP_NAME, "Укажите короткий ID workspace для регистрации.")
            return
        self.result = {
            "inboxPath": inbox_path,
            "registerWorkspace": self.register_check.isChecked(),
            "workspaceId": workspace_id,
            "workspaceName": workspace_name,
        }
        self.accept()


class DevctlGui(QtWidgets.QMainWindow if QtWidgets else object):
    def __init__(self) -> None:
        if QtWidgets is None or QtCore is None or QtGui is None:
            raise RuntimeError("PySide6 не импортирован")
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1120, 760)
        self.setMinimumSize(980, 640)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.current_process = None
        self.last_status: dict | None = None
        self.last_plan: dict | None = None
        self.last_report_path: str | None = None
        self.last_archive_path: str | None = None
        self.last_uts_path: str | None = None
        self.recommended_action_code = "refresh_status"
        self._text_tabs: list[tuple[str, QtWidgets.QPlainTextEdit]] = []
        self._workspace_combo_updating = False

        self.config_data = load_config()
        default_workspace = initial_workspace(self.config_data)
        self.runner = DevctlRunner(default_workspace)

        self._setup_styles()
        self._build_ui(default_workspace)
        self.refresh_registered_workspaces()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._drain_events)
        self._timer.start(100)
        QtCore.QTimer.singleShot(100, self.refresh_status)

    def _guard(self, func: Callable, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            self._show_callback_exception(*sys.exc_info())
            return None

    def _show_callback_exception(self, exc_type, exc, tb) -> None:
        details = "".join(traceback.format_exception(exc_type, exc, tb))
        self.set_text(self.report_text, "== ошибка GUI ==\n\n" + details)
        self.notebook.setCurrentIndex(2)
        QtWidgets.QMessageBox.critical(self, APP_NAME, f"Ошибка в обработчике GUI:\n{exc}")

    def _setup_styles(self) -> None:
        self.colors = {
            "bg": "#121212",
            "panel": "#1e1e1e",
            "panel2": "#181818",
            "border": "#30363d",
            "text": "#c9d1d9",
            "muted": "#8b949e",
            "ok": "#3fb950",
            "warn": "#d29922",
            "bad": "#f85149",
            "button": "#2a2a2a",
            "button_active": "#343434",
            "entry": "#202020",
            "select": "#4a4a4a",
            "accent": "#238636",
            "accent_hover": "#2ea043",
        }
        c = self.colors
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                color: {c['text']};
                font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
                font-size: 10pt;
            }}
            QMainWindow, QWidget#AppRoot, QDialog {{
                background: {c['bg']};
            }}
            QLabel, QCheckBox, QRadioButton {{
                background: transparent;
            }}
            QLabel#HeaderLabel {{
                color: #f0f6fc;
                font-size: 15pt;
                font-weight: 700;
            }}
            QLabel#SubtleLabel, QLabel#MutedLabel {{
                color: {c['muted']};
                font-size: 9pt;
            }}
            QLabel#DialogTitle {{
                color: #f0f6fc;
                font-size: 14pt;
                font-weight: 700;
            }}
            QFrame#StatusCard, QFrame#NextActionCard {{
                background: {c['panel']};
                border: none;
                border-radius: 14px;
            }}
            QLabel#CardTitle {{
                color: #f0f6fc;
                font-weight: 700;
            }}
            QLabel#CardValue {{ color: {c['muted']}; }}
            QLabel#CardValue[kind="ok"] {{ color: {c['ok']}; font-weight: 700; }}
            QLabel#CardValue[kind="warn"] {{ color: {c['warn']}; font-weight: 700; }}
            QLabel#CardValue[kind="bad"] {{ color: {c['bad']}; font-weight: 700; }}
            QLabel#NextActionTitle {{
                color: #f0f6fc;
                font-size: 12pt;
                font-weight: 700;
            }}
            QLabel#NextActionTitle[kind="ok"] {{ color: {c['ok']}; }}
            QLabel#NextActionTitle[kind="warn"] {{ color: {c['warn']}; }}
            QLabel#NextActionTitle[kind="bad"] {{ color: {c['bad']}; }}
            QLabel#NextActionBody {{ color: {c['text']}; }}
            QLineEdit, QComboBox {{
                background: {c['entry']};
                color: {c['text']};
                border: none;
                border-radius: 10px;
                padding: 8px 10px;
                selection-background-color: {c['select']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 26px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {c['muted']};
                width: 0;
                height: 0;
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background: {c['panel']};
                color: {c['text']};
                border: none;
                selection-background-color: {c['select']};
                padding: 4px;
            }}
            QPushButton {{
                background: {c['button']};
                color: {c['text']};
                border: none;
                border-radius: 10px;
                padding: 8px 12px;
                min-height: 22px;
            }}
            QPushButton:hover {{ background: {c['button_active']}; }}
            QPushButton:pressed {{ background: #1f6feb; }}
            QPushButton:disabled {{ color: #6e7681; background: {c['button']}; }}
            QPushButton#MagicButton {{
                background: {c['accent']};
                color: white;
                border: none;
                border-radius: 14px;
                padding: 13px 18px;
                font-size: 13pt;
                font-weight: 700;
            }}
            QPushButton#MagicButton:hover {{ background: {c['accent_hover']}; }}
            QToolButton#ToolIconButton {{
                background: {c['button']};
                color: {c['text']};
                border: none;
                border-radius: 13px;
                padding: 8px;
                min-width: 36px;
                min-height: 34px;
            }}
            QToolButton#ToolIconButton:hover {{
                background: {c['button_active']};
            }}
            QToolButton#ToolIconButton:pressed {{
                background: #24364a;
            }}
            QToolButton#ToolIconButton:disabled {{
                color: #6e7681;
                background: #171c23;
            }}
            QTabWidget::pane {{
                border: none;
                border-radius: 12px;
                top: -1px;
                background: {c['panel2']};
            }}
            QTabBar::tab {{
                background: {c['button']};
                color: {c['muted']};
                border: none;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                padding: 8px 14px;
                margin-right: 4px;
            }}
            QTabBar::tab:selected {{ color: #f0f6fc; background: {c['panel']}; }}
            QPlainTextEdit {{
                background: {c['panel2']};
                color: {c['text']};
                border: none;
                selection-background-color: {c['select']};
                font-family: Consolas, 'Cascadia Mono', monospace;
                font-size: 10pt;
                padding: 10px;
            }}
            QScrollBar:vertical {{
                background: {c['panel2']};
                width: 12px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical {{
                background: {c['border']};
                border-radius: 6px;
                min-height: 24px;
            }}
            QGroupBox {{
                border: none;
                border-radius: 12px;
                margin-top: 10px;
                padding: 14px 10px 10px 10px;
                background: {c['panel']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #f0f6fc;
            }}
        """)

    def _flat_icon(self, name: str) -> QtGui.QIcon:
        """Return an app-owned flat SVG icon for the toolbar.

        The previous GUI versions used either OS standard icons or QPainter
        glyphs. Standard icons look inconsistent on Windows dark themes, while
        code-drawn glyphs are hard to audit and tune. The local SVG set keeps
        the toolbar dependency-free, editable, minimalist, and bundled into the
        EXE by PyInstaller.
        """
        icon_path = bundled_root() / "gui" / "assets" / "icons" / f"{name}.svg"
        if icon_path.exists():
            return QtGui.QIcon(str(icon_path))
        fallback = {
            "init": QtWidgets.QStyle.StandardPixmap.SP_FileDialogNewFolder,
            "status": QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton,
            "settings": QtWidgets.QStyle.StandardPixmap.SP_FileDialogDetailedView,
            "sync": QtWidgets.QStyle.StandardPixmap.SP_BrowserReload,
            "inbox": QtWidgets.QStyle.StandardPixmap.SP_ArrowDown,
            "plan": QtWidgets.QStyle.StandardPixmap.SP_FileDialogContentsView,
            "start": QtWidgets.QStyle.StandardPixmap.SP_MediaPlay,
            "no_push": QtWidgets.QStyle.StandardPixmap.SP_MediaPlay,
            "upgrade": QtWidgets.QStyle.StandardPixmap.SP_ArrowUp,
            "reset": QtWidgets.QStyle.StandardPixmap.SP_ArrowBack,
            "report": QtWidgets.QStyle.StandardPixmap.SP_FileIcon,
            "archives": QtWidgets.QStyle.StandardPixmap.SP_DirIcon,
            "uts": QtWidgets.QStyle.StandardPixmap.SP_ComputerIcon,
            "project": QtWidgets.QStyle.StandardPixmap.SP_DirHomeIcon,
            "copy": QtWidgets.QStyle.StandardPixmap.SP_FileDialogListView,
            "prompt": QtWidgets.QStyle.StandardPixmap.SP_DialogHelpButton,
        }.get(name, QtWidgets.QStyle.StandardPixmap.SP_FileIcon)
        return self.style().standardIcon(fallback)


    def _make_tool_button(self, icon_name: str, tooltip: str, command: Callable) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton()
        button.setObjectName("ToolIconButton")
        button.setAutoRaise(False)
        button.setIcon(self._flat_icon(icon_name))
        button.setIconSize(QtCore.QSize(20, 20))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.clicked.connect(lambda _checked=False, cmd=command: self._guard(cmd))
        return button

    def _build_ui(self, default_workspace: str) -> None:
        root = QtWidgets.QWidget()
        root.setObjectName("AppRoot")
        self.setCentralWidget(root)
        main = QtWidgets.QVBoxLayout(root)
        main.setContentsMargins(16, 16, 16, 16)
        main.setSpacing(12)

        top = QtWidgets.QHBoxLayout()
        header = QtWidgets.QLabel("Рабочая область")
        header.setObjectName("HeaderLabel")
        version = QtWidgets.QLabel(f"devctl v{BUNDLED_DEVCTL_VERSION} · GUI v{APP_VERSION} · Qt/PySide6")
        version.setObjectName("SubtleLabel")
        top.addWidget(header)
        top.addStretch(1)
        top.addWidget(version)
        main.addLayout(top)

        path_row = QtWidgets.QHBoxLayout()
        self.path_entry = QtWidgets.QLineEdit(default_workspace)
        path_row.addWidget(self.path_entry, 1)
        self.workspace_combo = QtWidgets.QComboBox()
        self.workspace_combo.setMinimumWidth(240)
        self.workspace_combo.setToolTip("Быстрый выбор workspace из глобального devctl registry")
        self.workspace_combo.activated.connect(lambda _index=0: self._guard(self.choose_registered_workspace))
        path_row.addWidget(self.workspace_combo)
        choose_btn = QtWidgets.QPushButton("Выбрать")
        choose_btn.clicked.connect(lambda: self._guard(self.choose_workspace))
        self.init_top_btn = QtWidgets.QPushButton("Новый workspace")
        self.init_top_btn.clicked.connect(lambda: self._guard(self.init_workspace))
        self.settings_top_btn = QtWidgets.QPushButton("Настройки")
        self.settings_top_btn.clicked.connect(lambda: self._guard(self.configure_patch_inbox))
        refresh_btn = QtWidgets.QPushButton("Обновить")
        refresh_btn.clicked.connect(lambda: self._guard(self.refresh_status))
        path_row.addWidget(choose_btn)
        path_row.addWidget(self.init_top_btn)
        path_row.addWidget(self.settings_top_btn)
        path_row.addWidget(refresh_btn)
        main.addLayout(path_row)

        cards_layout = QtWidgets.QGridLayout()
        cards_layout.setSpacing(8)
        self.cards = {
            "project": StatusCard("Проект"),
            "git": StatusCard("Git"),
            "patch": StatusCard("Патч"),
            "uts": StatusCard("UTS"),
            "push": StatusCard("Push"),
        }
        for index, card in enumerate(self.cards.values()):
            cards_layout.addWidget(card, 0, index)
            cards_layout.setColumnStretch(index, 1)
        main.addLayout(cards_layout)

        self.next_action_frame = QtWidgets.QFrame()
        self.next_action_frame.setObjectName("NextActionCard")
        next_layout = QtWidgets.QVBoxLayout(self.next_action_frame)
        next_layout.setContentsMargins(14, 12, 14, 12)
        next_layout.setSpacing(6)
        next_label = QtWidgets.QLabel("Следующее действие")
        next_label.setObjectName("CardTitle")
        self.next_action_title = QtWidgets.QLabel("Проверить workspace")
        self.next_action_title.setObjectName("NextActionTitle")
        self.next_action_title.setWordWrap(True)
        self.next_action_body = QtWidgets.QLabel("GUI сейчас обновит статус и подскажет безопасный следующий шаг.")
        self.next_action_body.setObjectName("NextActionBody")
        self.next_action_body.setWordWrap(True)
        next_layout.addWidget(next_label)
        next_layout.addWidget(self.next_action_title)
        next_layout.addWidget(self.next_action_body)
        main.addWidget(self.next_action_frame)

        self.main_button = QtWidgets.QPushButton("Проверить workspace")
        self.main_button.setObjectName("MagicButton")
        self.main_button.clicked.connect(lambda: self._guard(self.perform_recommended_action))
        main.addWidget(self.main_button)

        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(6)
        self.init_btn = self._make_tool_button("init", "Инициализировать workspace", self.init_workspace)
        self.status_btn = self._make_tool_button("status", "Показать статус", self.refresh_status)
        self.settings_btn = self._make_tool_button("settings", "Настройки Patch Inbox и регистрации workspace", self.configure_patch_inbox)
        self.sync_btn = self._make_tool_button("sync", "Синхронизировать workspace с GitHub: project -> archives -> UTS", self.sync_workspace)
        self.inbox_btn = self._make_tool_button("inbox", "Забрать patch.zip из Patch Inbox", self.grab_inbox_patch)
        self.plan_btn = self._make_tool_button("plan", "Построить dry-run план", self.build_plan)
        self.start_btn = self._make_tool_button("start", "Запустить конвейер с push по плану", lambda: self.start_pipeline(False))
        self.no_push_btn = self._make_tool_button("no_push", "Запустить конвейер без push", lambda: self.start_pipeline(True))
        self.upgrade_btn = self._make_tool_button("upgrade", "Безопасно обновить структуру workspace", self.upgrade_workspace)
        self.reset_btn = self._make_tool_button("reset", "Reset project: git reset --hard + git clean", self.reset_project)
        self.report_btn = self._make_tool_button("report", "Открыть последний отчёт", self.open_report)
        self.archives_btn = self._make_tool_button("archives", "Открыть archives/", self.open_archives)
        self.uts_btn = self._make_tool_button("uts", "Открыть UserTestSpace/ или свежую UTS-копию", self.open_uts)
        self.project_btn = self._make_tool_button("project", "Открыть project/", self.open_project)
        self.copy_output_btn = self._make_tool_button("copy", "Скопировать текущий вывод активной вкладки", self.copy_current_output)
        self.copy_prompt_btn = self._make_tool_button("prompt", "Скопировать prompt-патча", self.copy_patch_prompt)
        self.action_buttons = (
            self.init_btn, self.status_btn, self.settings_btn, self.sync_btn, self.inbox_btn, self.plan_btn,
            self.start_btn, self.no_push_btn, self.upgrade_btn, self.reset_btn, self.report_btn,
            self.archives_btn, self.uts_btn, self.project_btn, self.copy_output_btn, self.copy_prompt_btn,
        )
        for widget in self.action_buttons:
            actions.addWidget(widget)
        actions.addStretch(1)
        main.addLayout(actions)

        self.notebook = QtWidgets.QTabWidget()
        self.plan_text = self._make_text_tab("План")
        self.run_text = self._make_text_tab("Запуск")
        self.report_text = self._make_text_tab("Отчёт")
        main.addWidget(self.notebook, 1)

    def _make_text_tab(self, title: str) -> QtWidgets.QPlainTextEdit:
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.notebook.addTab(text, title)
        self._text_tabs.append((title, text))
        return text

    def set_text(self, widget: QtWidgets.QPlainTextEdit, value: str) -> None:
        widget.setPlainText(value or "")
        cursor = widget.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.Start)
        widget.setTextCursor(cursor)

    def append_run(self, value: str) -> None:
        cursor = self.run_text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        cursor.insertText(value)
        self.run_text.setTextCursor(cursor)
        self.run_text.ensureCursorVisible()

    def _set_next_action(self, code: str, title: str, details: str, button_text: str, kind: str = "neutral") -> None:
        self.recommended_action_code = code
        self.next_action_title.setProperty("kind", kind)
        self.next_action_title.setText(title)
        self.next_action_body.setText(details)
        self.main_button.setText(button_text)
        self.next_action_title.style().unpolish(self.next_action_title)
        self.next_action_title.style().polish(self.next_action_title)

    def perform_recommended_action(self) -> None:
        actions = {
            "init_workspace": self.init_workspace,
            "choose_workspace": self.choose_workspace,
            "refresh_status": self.refresh_status,
            "sync_workspace": self.sync_workspace,
            "grab_inbox_patch": self.grab_inbox_patch,
            "configure_patch_inbox": self.configure_patch_inbox,
            "open_patches": self.open_patches,
            "open_project": self.open_project,
            "show_git_status": self.show_git_status,
            "reset_project": self.reset_project,
            "upgrade_workspace": self.upgrade_workspace,
            "build_plan": self.build_plan,
            "start_pipeline": lambda: self.start_pipeline(False),
            "open_report": self.open_report,
            "open_uts": self.open_uts,
            "copy_patch_prompt": self.copy_patch_prompt,
        }
        actions.get(self.recommended_action_code, self.refresh_status)()

    @staticmethod
    def _workspace_upgrade_details(workspace_config: dict) -> str:
        missing = []
        missing.extend(workspace_config.get("missingFields") or [])
        missing.extend(workspace_config.get("missingArchiveExcludes") or [])
        missing.extend(workspace_config.get("missingDirs") or [])
        return ", ".join(str(item) for item in missing[:8]) or "структура workspace устарела"

    @staticmethod
    def _workspace_needs_structural_upgrade(workspace_config: dict) -> bool:
        """Return True only when the workspace is missing real infrastructure.

        `devctl status` may also report optional archive exclude hints. Those are
        useful maintenance details, but they should not become the yellow primary
        recommendation when project/, patches/, archives/ and UserTestSpace/ are
        already present and there are simply no new patches to run.
        """
        if not workspace_config.get("upgradeAvailable"):
            return False
        missing_fields = [str(item) for item in (workspace_config.get("missingFields") or [])]
        missing_dirs = [str(item) for item in (workspace_config.get("missingDirs") or [])]
        if workspace_config.get("error"):
            return True
        return bool(missing_fields or missing_dirs)

    @staticmethod
    def _format_idle_workspace_details(workspace_config: dict) -> str:
        optional_excludes = [str(item) for item in (workspace_config.get("missingArchiveExcludes") or [])]
        details = "Структура workspace готова, Git чистый, неприменённых patch.zip нет. Можно положить новый архив в Patch Inbox/incoming/ и нажать «Забрать патч из склада» либо просто оставить workspace как есть."
        if optional_excludes:
            details += " Есть только необязательные подсказки по archive.exclude: " + ", ".join(optional_excludes[:6]) + ". Они не означают, что workspace сломан."
        return details

    def _recommend_from_status(self, data: dict) -> None:
        workspace = data.get("workspace", {}) if isinstance(data.get("workspace"), dict) else {}
        git = data.get("git", {}) if isinstance(data.get("git"), dict) else {}
        patches = data.get("patches", {}) if isinstance(data.get("patches"), dict) else {}
        latest = patches.get("latest")
        project_root = workspace.get("projectRoot") or "project/"
        workspace_config = data.get("workspaceConfig", {}) if isinstance(data.get("workspaceConfig"), dict) else {}
        upgrade_available = bool(workspace.get("projectExists") and workspace_config.get("upgradeAvailable"))
        structural_upgrade_needed = bool(workspace.get("projectExists") and self._workspace_needs_structural_upgrade(workspace_config))
        upgrade_details = self._workspace_upgrade_details(workspace_config)
        if not workspace.get("projectExists"):
            self._set_next_action("init_workspace", "Создать или выбрать рабочую область devctl", f"В текущем workspace не найден project/: {project_root}. Можно создать новый workspace мастером или выбрать уже существующий.", "Инициализировать workspace", "warn")
            return
        if not git.get("available"):
            self._set_next_action("refresh_status", "Установить или добавить Git в PATH", "devctl не видит git.exe/git. Установите Git for Windows или добавьте его в PATH, затем обновите статус.", "Обновить статус", "bad")
            return
        if not git.get("isRepository"):
            self._set_next_action("open_project", "Проверить Git-репозиторий в project/", f"Папка project/ найдена, но не является Git-репозиторием: {project_root}. Откройте её и выполните git init или создайте workspace через мастер.", "Открыть project/", "bad")
            return
        if git.get("clean") is False:
            self._set_next_action("reset_project", "Рабочее дерево project/ загрязнено", "Можно открыть git status для ручной проверки или нажать reset: GUI вызовет devctl reset после отдельного предупреждения и откатит project/ через git reset --hard + git clean -fd.", "Reset project", "warn")
            return
        if not latest:
            if structural_upgrade_needed:
                self._set_next_action("upgrade_workspace", "Безопасно обновить структуру workspace", f"devctl видит, что workspace можно актуализировать: {upgrade_details}. Команда init --upgrade не трогает project/ и пользовательские пути, а только добавляет недостающую инфраструктуру.", "Обновить структуру workspace", "warn")
            elif self.last_report_path:
                self._set_next_action("open_report", "Workspace готов, новых патчей нет", "Неприменённых patch.zip сейчас нет. Можно открыть последний отчёт или положить новый архив в Patch Inbox/incoming/.", "Открыть последний отчёт", "ok")
            else:
                self._set_next_action("grab_inbox_patch", "Workspace готов, новых патчей нет", self._format_idle_workspace_details(workspace_config), "Забрать патч из склада", "ok")
            return
        if latest.get("manifestError"):
            self._set_next_action("build_plan", "Посмотреть ошибку патча", f"Найден patch.zip, но его манифест некорректен: {latest.get('manifestError')}. Постройте план, чтобы увидеть детали валидации.", "Показать ошибку патча", "bad")
            return
        title = latest.get("title") or latest.get("name") or "следующий патч"
        upgrade_note = ""
        if structural_upgrade_needed:
            upgrade_note = f" Дополнительно devctl предлагает обновить структуру workspace: {upgrade_details}. Это не блокирует построение плана и запуск патча."
        self._set_next_action("build_plan", "Построить прозрачный план запуска", f"Workspace выглядит готовым. Следующий патч: {title}. Перед запуском лучше посмотреть dry-run план: файлы, проверки, commit и push.{upgrade_note}", "Построить план", "ok")

    def _recommend_from_plan(self, data: dict, result: RunResult) -> None:
        if not isinstance(data, dict) or not data:
            self._set_next_action("refresh_status", "Не удалось разобрать план", result.stderr or result.stdout or "Команда plan не вернула JSON. Обновите статус и проверьте вкладку «План».", "Обновить статус", "bad")
            return
        if not data.get("ok"):
            validation = data.get("validation") if isinstance(data.get("validation"), dict) else {}
            error = validation.get("error") or data.get("error") or "Патч не прошёл dry-run проверку."
            self._set_next_action("open_patches", "Исправить или заменить patch.zip", f"План построить нельзя: {error} После исправления положите обновлённый архив в patches/ и нажмите «Обновить».", "Открыть patches/", "bad")
            return
        patch = data.get("patch") if isinstance(data.get("patch"), dict) else {}
        manifest = data.get("manifest") if isinstance(data.get("manifest"), dict) else {}
        apply = data.get("apply") if isinstance(data.get("apply"), dict) else {}
        checks = data.get("checks") if isinstance(data.get("checks"), list) else []
        push = data.get("push") if isinstance(data.get("push"), dict) else {}
        patch_title = manifest.get("title") or patch.get("title") or patch.get("name") or "патч"
        push_text = "без push"
        if push.get("enabled"):
            push_text = f"push в {push.get('remote') or 'origin'}/{push.get('branch') or 'текущую ветку'}"
        self._set_next_action("start_pipeline", f"Запустить конвейер для патча «{patch_title}»", "План готов: " f"файлов к копированию — {apply.get('copyCount', 0)}, " f"удалений — {apply.get('deleteCount', 0)}, " f"проверок — {len(checks)}, {push_text}. После нажатия devctl применит патч, выполнит проверки, сделает commit и при необходимости push.", "Запустить конвейер", "ok")

    def _recommend_after_result(self, data: dict, result: RunResult) -> None:
        status = data.get("status") if isinstance(data, dict) else None
        if result.returncode == 0:
            uts_project = data.get("utsProjectDir") if isinstance(data, dict) else None
            if uts_project:
                self._set_next_action("open_uts", "Открыть свежую UTS-копию", f"Конвейер завершился со статусом {status or 'OK'}. Успешный post-снимок уже развёрнут для ручного тестирования: {uts_project}", "Открыть UTS", "ok")
                return
            self._set_next_action("open_report", "Открыть отчёт успешного запуска", f"Конвейер завершился со статусом {status or 'OK'}. Отчёт уже создан и содержит commit/push-сводку.", "Открыть отчёт", "ok")
        else:
            self._set_next_action("open_report", "Разобрать ошибку по отчёту", f"Конвейер остановился со статусом {status or 'ошибка'}. Откройте отчёт: там есть ошибки, предупреждения и путь к диагностическому архиву.", "Открыть отчёт", "bad")

    def show_git_status(self) -> None:
        data = self.last_status or {}
        git = data.get("git", {}) if isinstance(data, dict) else {}
        workspace = data.get("workspace", {}) if isinstance(data, dict) else {}
        porcelain = git.get("porcelain") or git.get("statusShort") or ""
        lines = ["== git status ==", f"Project: {workspace.get('projectRoot') or 'неизвестно'}", f"Ветка: {git.get('branch') or 'неизвестно'}", ""]
        if porcelain.strip():
            lines.append(porcelain.rstrip())
        else:
            lines.append("Локальные изменения не перечислены в status JSON. Откройте project/ и выполните git status вручную.")
        lines.extend(["", "Перед запуском devctl приведите рабочее дерево к чистому состоянию: закоммитьте нужное, уберите временное или откатите лишнее."])
        self.set_text(self.plan_text, "\n".join(lines) + "\n")
        self.notebook.setCurrentIndex(0)

    def _current_output_widget(self) -> tuple[str, QtWidgets.QPlainTextEdit]:
        index = self.notebook.currentIndex()
        if 0 <= index < len(self._text_tabs):
            return self._text_tabs[index]
        return "План", self.plan_text

    def copy_current_output(self) -> None:
        title, widget = self._current_output_widget()
        text = widget.toPlainText()
        if not text.strip():
            QtWidgets.QMessageBox.information(self, APP_NAME, f"Во вкладке «{title}» пока нет текста для копирования.")
            return
        QtWidgets.QApplication.clipboard().setText(text)
        QtWidgets.QMessageBox.information(self, APP_NAME, f"Текущий вывод вкладки «{title}» скопирован в буфер обмена.")

    def copy_patch_prompt(self) -> None:
        text = PATCH_PROMPT_TEMPLATE.strip() + "\n"
        QtWidgets.QApplication.clipboard().setText(text)
        self.set_text(self.report_text, "== prompt-шаблон devctl-патча ==\n\nШаблон скопирован в буфер обмена. Его можно вставить в новую ChatGPT-сессию, чтобы попросить собрать корректный devctl-патч для этого workspace.\n\n" + text)
        self.notebook.setCurrentIndex(2)
        QtWidgets.QMessageBox.information(self, APP_NAME, "Prompt-шаблон для devctl-патча скопирован в буфер обмена.")

    def configure_patch_inbox(self) -> None:
        self._save_workspace()
        self.set_running(True)
        self.set_text(self.report_text, "Загружаю настройки Patch Inbox...\n")
        self.notebook.setCurrentIndex(2)
        self._run_async(["inbox", "scan", "--json"], self._on_patch_inbox_settings_scan, timeout=120)

    def _on_patch_inbox_settings_scan(self, result: RunResult) -> None:
        self.set_running(False)
        data = result.json_data if isinstance(result.json_data, dict) else {}
        workspace = Path(self.path_entry.text()).expanduser().resolve()
        dialog = PatchInboxSettingsDialog(self, workspace_path=workspace, scan_data=data)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or not dialog.result:
            self.set_text(self.report_text, self._format_inbox_scan_result(data, result) if data else (result.stdout or "Настройка отменена.\n"))
            return
        self._apply_patch_inbox_settings(dialog.result)

    def _apply_patch_inbox_settings(self, settings: dict[str, object]) -> None:
        inbox_path = str(settings.get("inboxPath") or "").strip()
        commands: list[list[str]] = []
        if inbox_path:
            commands.append(["inbox", "init", "--path", inbox_path, "--json"])
        if bool(settings.get("registerWorkspace")):
            register_cmd = ["workspace", "register", ".", "--json"]
            workspace_id = str(settings.get("workspaceId") or "").strip()
            workspace_name = str(settings.get("workspaceName") or "").strip()
            if workspace_id:
                register_cmd.extend(["--id", workspace_id])
            if workspace_name:
                register_cmd.extend(["--name", workspace_name])
            commands.append(register_cmd)
        if not commands:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Нет изменений для сохранения.")
            return
        self.set_running(True)
        self.set_text(self.report_text, "Сохраняю настройки Patch Inbox...\n\n" + "\n".join("devctl " + " ".join(cmd) for cmd in commands) + "\n")
        self.notebook.setCurrentIndex(2)
        active_runner = self.runner

        def worker() -> None:
            results: list[tuple[list[str], RunResult]] = []
            for command in commands:
                result = active_runner.run(command, timeout=120)
                results.append((command, result))
                if not result.ok:
                    break
            self.events.put(("result", (self._on_patch_inbox_settings_done, results)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_patch_inbox_settings_done(self, results: list[tuple[list[str], RunResult]]) -> None:
        self.set_running(False)
        self.set_text(self.report_text, self._format_settings_results(results))
        ok = all(result.ok and (not isinstance(result.json_data, dict) or result.json_data.get("ok", True)) for _command, result in results)
        if ok:
            self._set_next_action("grab_inbox_patch", "Patch Inbox настроен", "Склад патчей и registry workspace обновлены. Теперь можно положить patch.zip в incoming/ и забрать его кнопкой Patch Inbox.", "Забрать патч из склада", "ok")
            QtWidgets.QMessageBox.information(self, APP_NAME, "Настройки Patch Inbox сохранены.")
            self.refresh_registered_workspaces()
            self.refresh_status()
        else:
            self._set_next_action("configure_patch_inbox", "Настройки Patch Inbox не сохранены полностью", "Проверьте отчёт: одна из команд завершилась с ошибкой.", "Открыть настройки", "bad")
            QtWidgets.QMessageBox.critical(self, APP_NAME, "Не удалось сохранить настройки Patch Inbox. Подробности во вкладке «Отчёт».")

    def _format_settings_results(self, results: list[tuple[list[str], RunResult]]) -> str:
        lines = ["== настройки Patch Inbox =="]
        for command, result in results:
            lines.extend(["", "$ devctl " + " ".join(command), f"Код возврата: {result.returncode}"])
            data = result.json_data if isinstance(result.json_data, dict) else None
            if data:
                if command[:2] == ["inbox", "init"]:
                    lines.extend([f"Склад: {data.get('path')}", f"Config: {data.get('configPath')}", f"Добавлен в config: {data.get('addedToConfig')}"])
                    subdirs = data.get("subdirs") if isinstance(data.get("subdirs"), dict) else {}
                    for name, path in subdirs.items():
                        lines.append(f"  {name}: {path}")
                elif command[:2] == ["workspace", "register"]:
                    workspace = data.get("workspace") if isinstance(data.get("workspace"), dict) else {}
                    lines.extend([f"Workspace ID: {workspace.get('id')}", f"Workspace name: {workspace.get('name')}", f"Workspace path: {workspace.get('path')}", f"Config: {data.get('configPath')}"])
                else:
                    lines.append(json.dumps(data, ensure_ascii=False, indent=2))
            if result.stdout.strip() and not data:
                lines.append(result.stdout.strip())
            if result.stderr.strip():
                lines.append("stderr:")
                lines.append(result.stderr.strip())
        return "\n".join(lines) + "\n"

    def refresh_registered_workspaces(self) -> None:
        """Refresh the quick workspace switcher from devctl global config."""
        combo = getattr(self, "workspace_combo", None)
        if combo is None:
            return
        current_text = self.path_entry.text().strip() if hasattr(self, "path_entry") else ""
        try:
            current_path = str(Path(current_text).expanduser().resolve()).casefold() if current_text else ""
        except Exception:
            current_path = current_text.casefold()
        records = registered_workspaces()
        self._workspace_combo_updating = True
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("workspace registry", "")
        combo.setItemData(0, "Быстрый выбор workspace из ~/.config/devctl/config.json / %APPDATA%\\devctl\\config.json", QtCore.Qt.ItemDataRole.ToolTipRole)
        selected_index = 0
        for record in records:
            path = str(record.get("path") or "").strip()
            if not path:
                continue
            name = str(record.get("name") or record.get("id") or Path(path).name or "workspace")
            wid = str(record.get("id") or "").strip()
            label = f"{name} · {wid}" if wid and wid != name else name
            combo.addItem(label, path)
            combo.setItemData(combo.count() - 1, path, QtCore.Qt.ItemDataRole.ToolTipRole)
            try:
                normalized = str(Path(path).expanduser().resolve()).casefold()
            except Exception:
                normalized = path.casefold()
            if normalized == current_path:
                selected_index = combo.count() - 1
        combo.setCurrentIndex(selected_index)
        combo.setEnabled(bool(records))
        combo.blockSignals(False)
        self._workspace_combo_updating = False

    def choose_registered_workspace(self) -> None:
        combo = getattr(self, "workspace_combo", None)
        if combo is None or self._workspace_combo_updating:
            return
        path = str(combo.currentData() or "").strip()
        if not path:
            return
        self.path_entry.setText(path)
        self._save_workspace()
        self.refresh_registered_workspaces()
        self.refresh_status()


    def choose_workspace(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите корень рабочей области devctl")
        if not selected:
            return
        self.path_entry.setText(selected)
        self._save_workspace()
        self.refresh_registered_workspaces()
        self.refresh_status()

    def init_workspace(self) -> None:
        current = Path(self.path_entry.text()).expanduser()
        initial_parent = current.parent if current.name else Path.home()
        dialog = InitWorkspaceDialog(self, initial_parent=initial_parent)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or not dialog.result:
            return
        request = dialog.result
        parent = Path(request["parent"]).expanduser().resolve()
        workspace = (parent / request["name"]).resolve()
        remote_url = request.get("remoteUrl", "").strip()
        branch = request.get("branch", "main").strip() or "main"
        try:
            parent.mkdir(parents=True, exist_ok=True)
            if workspace.exists() and any(workspace.iterdir()):
                QtWidgets.QMessageBox.critical(self, APP_NAME, "Папка workspace уже существует и не пуста.\n\n" f"{workspace}\n\n" "Выберите другое название или пустую директорию, чтобы GUI ничего случайно не перезаписал.")
                return
            workspace.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"Не удалось подготовить папку workspace:\n{workspace}\n\n{exc}")
            return
        self.set_running(True)
        self.set_text(self.report_text, "Инициализирую workspace...\n\n" f"Workspace: {workspace}\n" f"Project:   {workspace / 'project'}\n" f"Remote:    {remote_url or 'не задан'}\n")
        self.notebook.setCurrentIndex(2)
        args = ["init", "--json", "--workspace", str(workspace), "--project", "project", "--patches", "patches", "--archives", "archives", "--uts", "UserTestSpace", "--create-project", "--git-init", "--branch", branch]
        if remote_url:
            args.extend(["--remote-url", remote_url])
        init_runner = DevctlRunner(workspace)
        self._run_async(args, lambda result, target=workspace: self._on_init_workspace_done(result, target), runner=init_runner, save_workspace=False)

    def _on_init_workspace_done(self, result: RunResult, workspace: Path) -> None:
        self.set_running(False)
        data = result.json_data or {}
        self.set_text(self.report_text, self._format_init_result(data if isinstance(data, dict) else {}, result, workspace))
        if result.ok and isinstance(data, dict) and data.get("ok"):
            self.path_entry.setText(str(workspace))
            self._save_workspace()
            self.refresh_registered_workspaces()
            self.last_report_path = None
            self.last_archive_path = None
            self.refresh_status()
            QtWidgets.QMessageBox.information(self, APP_NAME, "Workspace создан и открыт в GUI.")
        else:
            QtWidgets.QMessageBox.critical(self, APP_NAME, "Инициализация workspace завершилась с ошибкой. Подробности во вкладке «Отчёт».")

    def _format_init_result(self, data: dict, result: RunResult, workspace: Path) -> str:
        if not data:
            return (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        git_data = data.get("git") if isinstance(data.get("git"), dict) else {}
        lines = ["== инициализация workspace ==", f"Статус: {'OK' if data.get('ok') else 'ошибка'}", f"Код возврата: {result.returncode}", f"Workspace: {data.get('workspaceRoot') or workspace}", f"Project: {data.get('projectRoot') or workspace / 'project'}", f"Patches: {data.get('patchesDir') or workspace / 'patches'}", f"Archives: {data.get('archivesDir') or workspace / 'archives'}", f"UTS: {data.get('userTestSpaceDir') or workspace / 'UserTestSpace'}", f"Config: {data.get('configPath') or workspace / '.devctl' / 'workspace.json'}", "", "== git ==", f"Git доступен: {git_data.get('available')}", f"Репозиторий создан: {git_data.get('initialized')}", f"Ветка: {git_data.get('branch') or 'неизвестно'}", f"Remote origin: {git_data.get('remoteUrl') or 'не задан'}", f"Remote связан: {git_data.get('remoteLinked')}", f"Операция: {git_data.get('operation') or 'нет'}", f"Синхронизация remote: {git_data.get('synced')}", f"Pull выполнен: {git_data.get('pulled')}", f"Pull пропущен: {git_data.get('pullSkipped')}", "", "== создано =="]
        lines.extend([f"- {item}" for item in data.get("created") or []] or ["всё уже существовало"])
        operations = git_data.get("operations") if isinstance(git_data, dict) else []
        lines.append(""); lines.append("== git-шаги ==")
        lines.extend([f"- {item}" for item in operations] or ["нет"])
        warnings = data.get("warnings") or []
        lines.append(""); lines.append("== предупреждения ==")
        lines.extend([f"- {item}" for item in warnings] or ["нет"])
        errors = []
        if data.get("error"):
            errors.append(data.get("error"))
        if isinstance(git_data, dict):
            errors.extend(git_data.get("errors") or [])
        lines.append(""); lines.append("== ошибки ==")
        lines.extend([f"- {item}" for item in errors] or ["нет"])
        return "\n".join(lines) + "\n"

    def sync_workspace(self) -> None:
        self._save_workspace()
        if not self._confirm("Синхронизировать workspace с GitHub как источником истины?\n\nGUI вызовет `devctl sync --discard-local --json`: project/ будет приведён к origin/ветке через fetch + reset --hard + git clean, затем devctl создаст свежий архив в archives/ и развернёт копию в UserTestSpace/.\n\nЛокальные незакоммиченные изменения и локальные commit'ы, которых нет в remote, могут быть отброшены."):
            return
        self.set_running(True)
        self.set_text(self.report_text, "Синхронизирую workspace с GitHub как источником истины...\n\nКоманда: devctl sync --discard-local --json\nЭтапы: fetch/reset/clean -> свежий archives/ snapshot -> свежий UserTestSpace/.\n")
        self.notebook.setCurrentIndex(2)
        self._run_async(["sync", "--discard-local", "--json"], self._on_sync_workspace_done, timeout=900)

    def _on_sync_workspace_done(self, result: RunResult) -> None:
        self.set_running(False)
        data = result.json_data or {}
        if isinstance(data, dict):
            archive = data.get("archive") if isinstance(data.get("archive"), dict) else {}
            uts = data.get("uts") if isinstance(data.get("uts"), dict) else {}
            self.last_archive_path = archive.get("path") or self.last_archive_path
            self.last_uts_path = uts.get("projectDir") or self.last_uts_path
            self.set_text(self.report_text, self._format_sync_result(data, result))
        else:
            self.set_text(self.report_text, result.stdout + ("\n" + result.stderr if result.stderr else ""))
        self.notebook.setCurrentIndex(2)
        self.refresh_status()
        if result.ok and isinstance(data, dict) and data.get("ok"):
            QtWidgets.QMessageBox.information(self, APP_NAME, "Workspace синхронизирован: project, archives и UserTestSpace актуализированы.")
        else:
            QtWidgets.QMessageBox.critical(self, APP_NAME, "Не удалось синхронизировать workspace. Подробности во вкладке «Отчёт».")

    def _format_sync_result(self, data: dict, result: RunResult) -> str:
        if not data:
            return (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        workspace = data.get("workspace") if isinstance(data.get("workspace"), dict) else {}
        git = data.get("git") if isinstance(data.get("git"), dict) else {}
        archive = data.get("archive") if isinstance(data.get("archive"), dict) else {}
        uts = data.get("uts") if isinstance(data.get("uts"), dict) else {}
        lines = ["== sync workspace ==", f"Статус: {'OK' if data.get('ok') else 'ошибка'}", f"Код возврата: {result.returncode}", f"Workspace: {workspace.get('workspaceRoot') or 'неизвестно'}", f"Project: {workspace.get('projectRoot') or 'неизвестно'}", "", "== git ==", f"Remote: {git.get('remote') or 'origin'}", f"Remote URL: {git.get('remoteUrl') or 'не задан'}", f"Ветка: {git.get('branch') or 'неизвестно'}", f"Discard local: {git.get('discardLocal')}", f"Head after: {git.get('headAfter') or 'нет'}", f"Clean after: {git.get('cleanAfter')}", "", "== git-шаги =="]
        lines.extend([f"- {item}" for item in git.get("operations") or []] or ["нет"])
        lines.extend(["", "== archives / UTS ==", f"Архив: {archive.get('path') or 'не создавался'}", f"Файлов в архиве: {archive.get('fileCount', 0)}", f"UTS project: {uts.get('projectDir') or 'не обновлялся'}", "", "== создано =="])
        lines.extend([f"- {item}" for item in data.get("created") or []] or ["служебные папки уже существовали"])
        warnings = []
        warnings.extend(data.get("warnings") or []); warnings.extend(git.get("warnings") or [])
        lines.append(""); lines.append("== предупреждения ==")
        lines.extend([f"- {item}" for item in warnings] or ["нет"])
        errors = []
        if data.get("error"):
            errors.append(data.get("error"))
        errors.extend(git.get("errors") or [])
        lines.append(""); lines.append("== ошибки ==")
        lines.extend([f"- {item}" for item in errors] or ["нет"])
        return "\n".join(lines) + "\n"

    def grab_inbox_patch(self, workspace_id: str | None = None, *, allow_choice: bool = True) -> None:
        self._save_workspace()
        self.set_running(True)
        self.set_text(self.report_text, "Забираю patch.zip из Patch Inbox...\n")
        self.notebook.setCurrentIndex(2)
        args = ["inbox", "grab", "--json"]
        if workspace_id:
            args.extend(["--workspace", workspace_id])
        self._run_async(args, lambda result: self._on_inbox_grab_done(result, allow_choice=allow_choice), timeout=240)

    def _on_inbox_grab_done(self, result: RunResult, *, allow_choice: bool = True) -> None:
        self.set_running(False)
        data = result.json_data or {}
        if result.ok and isinstance(data, dict) and data.get("ok"):
            self.set_text(self.report_text, self._format_inbox_grab_result(data, result))
            imported = [item for item in (data.get("results") or []) if isinstance(item, dict) and item.get("status") == "imported"]
            if imported:
                self._set_next_action("build_plan", "Патч импортирован в patches/", "Patch Intake только доставил архив. Следующий безопасный шаг — построить dry-run план и уже потом запускать конвейер.", "Построить план", "ok")
                QtWidgets.QMessageBox.information(self, APP_NAME, "Патч импортирован. Теперь можно построить план.")
            else:
                self._set_next_action("refresh_status", "Patch Inbox обработан", "Импорт не создал нового patch.zip для запуска. Проверьте отчёт и обновите статус.", "Обновить статус", "warn")
            self.refresh_status()
            return
        self.set_text(self.report_text, self._format_inbox_grab_result(data if isinstance(data, dict) else {}, result))
        if allow_choice:
            self.set_text(self.report_text, self.report_text.toPlainText() + "\nПробую получить список workspace для ручного выбора...\n")
            self.set_running(True)
            self._run_async(["inbox", "scan", "--json"], self._on_inbox_scan_for_choice, timeout=120)
            return
        self._set_next_action("grab_inbox_patch", "Не удалось забрать патч из склада", result.stderr or result.stdout or "Проверьте, настроен ли Patch Inbox (`devctl inbox init`) и зарегистрирован ли workspace (`devctl workspace register`).", "Повторить inbox grab", "bad")
        QtWidgets.QMessageBox.critical(self, APP_NAME, "Не удалось забрать патч. Подробности во вкладке «Отчёт».")

    def _on_inbox_scan_for_choice(self, result: RunResult) -> None:
        self.set_running(False)
        data = result.json_data or {}
        if not isinstance(data, dict) or not data.get("ok"):
            self.set_text(self.report_text, (result.stdout or "") + ("\n" + result.stderr if result.stderr else ""))
            QtWidgets.QMessageBox.critical(self, APP_NAME, "Не удалось получить список workspace для выбора.")
            return
        self.set_text(self.report_text, self._format_inbox_scan_result(data, result))
        workspaces = data.get("workspaces") if isinstance(data.get("workspaces"), list) else []
        if not workspaces:
            self._set_next_action("configure_patch_inbox", "Workspace не зарегистрирован для Patch Intake", "Откройте настройки: GUI сможет указать путь до склада patch.zip и выполнить `devctl workspace register` для текущей рабочей области.", "Открыть настройки", "bad")
            QtWidgets.QMessageBox.critical(self, APP_NAME, "Нет зарегистрированных workspace для Patch Intake. Откройте настройки Patch Inbox.")
            return
        dialog = WorkspaceChoiceDialog(self, workspaces=workspaces)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or not dialog.result:
            self._set_next_action("grab_inbox_patch", "Импорт патча отменён", "Выбор workspace был отменён. Patch.zip остался в исходном складе.", "Забрать патч из склада", "warn")
            return
        self.grab_inbox_patch(dialog.result, allow_choice=False)

    def _format_inbox_scan_result(self, data: dict, result: RunResult) -> str:
        if not data:
            return (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        lines = ["== Patch Inbox scan ==", f"devctl: v{data.get('version')}", f"Config: {data.get('configPath')}", "", "== склады =="]
        lines.extend([f"- {item}" for item in data.get("patchInboxDirs") or []] or ["не настроены"])
        lines.append(""); lines.append("== зарегистрированные workspace ==")
        for record in data.get("workspaces") or []:
            lines.append(f"- {record.get('id')} · {record.get('name')} · {record.get('path')}")
        if not data.get("workspaces"):
            lines.append("нет")
        lines.append(""); lines.append("== патчи ==")
        patches = data.get("patches") if isinstance(data.get("patches"), dict) else {}
        for item in patches.get("items") or []:
            target = item.get("target") if isinstance(item.get("target"), dict) else {}
            lines.append(f"- {item.get('name')}: {item.get('status')}, target={target.get('workspaceId') or 'unknown'}, confidence={target.get('confidence')}, reason={target.get('reason') or 'нет'}")
        if not patches.get("items"):
            lines.append("zip-патчи не найдены")
        return "\n".join(lines) + "\n"

    def _format_inbox_grab_result(self, data: dict, result: RunResult) -> str:
        if not data:
            return (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        lines = ["== Patch Inbox grab ==", f"Статус: {'OK' if data.get('ok') else 'ошибка'}", f"Dry-run: {data.get('dryRun')}", f"Config: {data.get('configPath')}", f"Index: {data.get('indexPath')}", "", "== результаты =="]
        results = data.get("results") if isinstance(data.get("results"), list) else []
        for item in results:
            if not isinstance(item, dict):
                continue
            status = item.get("status")
            if status == "imported":
                lines.append(f"- imported: {item.get('originalName')} -> {item.get('importedTo')}")
                lines.append(f"  original: {item.get('movedOriginalTo')}")
                lines.append(f"  workspace: {item.get('workspaceId')} confidence={item.get('confidence')}")
            elif status == "would_import":
                lines.append(f"- would import: {item.get('name')} -> {item.get('copyTo')}")
            elif status == "duplicate":
                lines.append(f"- duplicate: {item.get('name')} -> {item.get('movedTo') or 'оставлен на месте'}")
            else:
                lines.append(f"- {status}: {item}")
        if not results:
            lines.append("нет")
        errors = data.get("errors") if isinstance(data.get("errors"), list) else []
        lines.append(""); lines.append("== ошибки ==")
        lines.extend([f"- {item}" for item in errors] or [result.stderr or result.stdout or "нет"])
        return "\n".join(lines) + "\n"

    def upgrade_workspace(self) -> None:
        self._save_workspace()
        if not self._confirm("Безопасно обновить структуру workspace?\n\nGUI вызовет `devctl init --upgrade`: команда добавляет недостающие поля конфигурации и служебные папки вроде UserTestSpace/, но не трогает содержимое project/."):
            return
        self.set_running(True)
        self.set_text(self.report_text, "Обновляю структуру workspace через devctl init --upgrade...\n")
        self.notebook.setCurrentIndex(2)
        self._run_async(["init", "--upgrade", "--json", "--workspace", self.path_entry.text()], self._on_upgrade_workspace_done)

    def _on_upgrade_workspace_done(self, result: RunResult) -> None:
        self.set_running(False)
        data = result.json_data or {}
        self.set_text(self.report_text, self._format_upgrade_result(data if isinstance(data, dict) else {}, result))
        self.notebook.setCurrentIndex(2)
        self.refresh_status()
        if result.ok and isinstance(data, dict) and data.get("ok"):
            QtWidgets.QMessageBox.information(self, APP_NAME, "Структура workspace обновлена.")
        else:
            QtWidgets.QMessageBox.critical(self, APP_NAME, "Не удалось обновить структуру workspace. Подробности во вкладке «Отчёт».")

    def _format_upgrade_result(self, data: dict, result: RunResult) -> str:
        if not data:
            return (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        lines = ["== обновление структуры workspace ==", f"Статус: {'OK' if data.get('ok') else 'ошибка'}", f"Код возврата: {result.returncode}", f"Workspace: {data.get('workspaceRoot')}", f"Config: {data.get('configPath')}", f"Config изменён: {data.get('changed')}", "", "== создано =="]
        lines.extend([f"- {item}" for item in data.get("created") or []] or ["ничего"])
        lines.append(""); lines.append("== обновлённые поля/исключения ==")
        lines.extend([f"- {item}" for item in data.get("updatedFields") or []] or ["уже актуально"])
        lines.append(""); lines.append("== предупреждения ==")
        lines.extend([f"- {item}" for item in data.get("warnings") or []] or ["нет"])
        if data.get("error"):
            lines.extend(["", "== ошибка ==", str(data.get("error"))])
        return "\n".join(lines) + "\n"

    def reset_project(self) -> None:
        self._save_workspace()
        if not self._confirm("Откатить project/?\n\nБудет выполнен `devctl reset`: git reset --hard HEAD и git clean -fd. Локальные незакоммиченные изменения и untracked-файлы в project/ будут удалены."):
            return
        self.set_running(True)
        self.set_text(self.report_text, "Выполняю devctl reset...\n")
        self.notebook.setCurrentIndex(2)
        self._run_async(["reset", "--json"], self._on_reset_done)

    def _on_reset_done(self, result: RunResult) -> None:
        self.set_running(False)
        data = result.json_data or {}
        self.set_text(self.report_text, self._format_reset_result(data if isinstance(data, dict) else {}, result))
        self.notebook.setCurrentIndex(2)
        self.refresh_status()
        if result.ok and isinstance(data, dict) and data.get("ok"):
            QtWidgets.QMessageBox.information(self, APP_NAME, "project/ откатан и очищен.")
        else:
            QtWidgets.QMessageBox.critical(self, APP_NAME, "devctl reset завершился с ошибкой. Подробности во вкладке «Отчёт».")

    def _format_reset_result(self, data: dict, result: RunResult) -> str:
        if not data:
            return (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        lines = ["== reset project ==", f"Статус: {'OK' if data.get('ok') else 'ошибка'}", f"Код возврата: {result.returncode}", f"Project: {data.get('projectRoot')}", f"Target: {data.get('target')}", f"Clean mode: {data.get('cleanMode')}", f"Patch удалён: {data.get('patchDeleted') or 'нет'}", "", "== git status before ==", data.get("gitStatusBefore") or "clean/нет данных", "", "== git status after ==", data.get("gitStatusAfter") or "clean"]
        if data.get("error"):
            lines.extend(["", "== ошибка ==", str(data.get("error"))])
        return "\n".join(str(item) for item in lines) + "\n"

    def _save_workspace(self) -> None:
        workspace = str(Path(self.path_entry.text()).expanduser().resolve())
        if not looks_like_pyinstaller_temp(Path(workspace)):
            self.config_data["lastWorkspace"] = workspace
            save_config(self.config_data)
        self.runner.set_workspace(workspace)

    def _run_async(self, args: list[str], callback, *, runner: DevctlRunner | None = None, save_workspace: bool = True, timeout: int = 180) -> None:
        if save_workspace:
            self._save_workspace()
        active_runner = runner or self.runner
        def worker() -> None:
            result = active_runner.run(args, timeout=timeout)
            self.events.put(("result", (callback, result)))
        threading.Thread(target=worker, daemon=True).start()

    def refresh_status(self) -> None:
        self.set_text(self.plan_text, "Обновляю статус...\n")
        self._run_async(["status", "--json"], self._on_status)

    def build_plan(self) -> None:
        self.set_text(self.plan_text, "Строю dry-run план...\n")
        self._run_async(["plan", "--json"], self._on_plan)

    def _on_status(self, result: RunResult) -> None:
        data = result.json_data or {}
        self.last_status = data if isinstance(data, dict) else None
        if not isinstance(data, dict) or not data.get("ok"):
            text = result.stderr or result.stdout or (data.get("error") if isinstance(data, dict) else None) or "Не удалось получить статус."
            self.cards["project"].set("ошибка", "bad")
            self.cards["git"].set("недоступно", "bad")
            self.cards["patch"].set("неизвестно", "warn")
            self.cards["uts"].set("неизвестно", "warn")
            self.cards["push"].set("неизвестно", "warn")
            self._set_next_action("choose_workspace", "Выбрать корректную рабочую область", str(text), "Выбрать workspace", "bad")
            self.set_text(self.plan_text, str(text))
            return
        workspace = data.get("workspace", {})
        git = data.get("git", {})
        latest = (data.get("patches", {}) or {}).get("latest")
        self.cards["project"].set("найден" if workspace.get("projectExists") else "не найден", "ok" if workspace.get("projectExists") else "bad")
        if not git.get("available"):
            self.cards["git"].set("git не найден", "bad")
        elif not git.get("isRepository"):
            self.cards["git"].set("не репозиторий", "bad")
        else:
            self.cards["git"].set("чисто" if git.get("clean") else "есть изменения", "ok" if git.get("clean") else "warn")
        if latest:
            self.cards["patch"].set(latest.get("title") or latest.get("name") or "найден", "bad" if latest.get("manifestError") else "ok")
        else:
            patches = data.get("patches", {}) if isinstance(data.get("patches"), dict) else {}
            self.cards["patch"].set("нет новых" if patches.get("count") else "нет патчей", "ok")
        self.cards["uts"].set("готов" if workspace.get("userTestSpaceDirExists") else "нужно обновить", "ok" if workspace.get("userTestSpaceDirExists") else "warn")
        self.cards["push"].set("см. план" if latest else "неизвестно", "neutral")
        self._recommend_from_status(data)
        self.set_text(self.plan_text, self._format_status(data))

    def _on_plan(self, result: RunResult) -> None:
        data = result.json_data or {}
        self.last_plan = data if isinstance(data, dict) else None
        self.set_text(self.plan_text, self._format_plan(data if isinstance(data, dict) else {}, result))
        push = data.get("push") if isinstance(data, dict) else None
        if isinstance(push, dict):
            enabled = bool(push.get("enabled"))
            target = f"{push.get('remote')}/{push.get('branch')}" if push.get("remote") and push.get("branch") else "цель неизвестна"
            self.cards["push"].set(("будет выполнен: " if enabled else "отключён: ") + target, "ok" if enabled else "warn")
        self._recommend_from_plan(data if isinstance(data, dict) else {}, result)
        self.notebook.setCurrentIndex(0)

    def _format_status(self, data: dict) -> str:
        workspace = data.get("workspace", {})
        git = data.get("git", {})
        patches = data.get("patches", {})
        lines = ["== статус devctl GUI ==", f"devctl: v{data.get('version')}", f"Рабочая область: {workspace.get('workspaceRoot')}", f"Проект: {workspace.get('projectRoot')}", f"Патчи: {workspace.get('patchesDir')}", f"Архивы: {workspace.get('archivesDir')}", f"UTS: {workspace.get('userTestSpaceDir')} ({'есть' if workspace.get('userTestSpaceDirExists') else 'нет'})", "", "== конфигурация workspace =="]
        workspace_config = data.get("workspaceConfig", {}) if isinstance(data.get("workspaceConfig"), dict) else {}
        lines.extend([f"Требует обновления: {workspace_config.get('upgradeAvailable')}", f"Недостающие поля: {', '.join(str(x) for x in workspace_config.get('missingFields') or []) or 'нет'}", f"Недостающие исключения archive: {', '.join(str(x) for x in workspace_config.get('missingArchiveExcludes') or []) or 'нет'}", f"Недостающие директории: {', '.join(str(x) for x in workspace_config.get('missingDirs') or []) or 'нет'}", "", "== git ==", f"Доступен: {git.get('available')}", f"Репозиторий: {git.get('isRepository')}", f"Ветка: {git.get('branch') or 'неизвестно'}", f"Рабочее дерево: {'чистое' if git.get('clean') else 'есть изменения/неизвестно'}", f"Remote origin: {git.get('remoteUrl') or 'не задан'}", f"Ahead/behind: {((git.get('aheadBehind') or {}).get('ahead'))}/{((git.get('aheadBehind') or {}).get('behind'))}", f"Ошибка: {git.get('error') or 'нет'}", "", "== патчи ==", f"Всего кандидатов: {patches.get('count', 0)}"])
        latest = patches.get("latest")
        if latest:
            lines.extend([f"Последний кандидат: {latest.get('name')}", f"ID: {latest.get('patchId') or 'неизвестно'}", f"Название: {latest.get('title') or 'неизвестно'}", f"Статус: {latest.get('status')}", f"Манифест: {latest.get('manifestError') or 'OK'}"])
        else:
            lines.append("Неприменённых patch.zip не найдено; все кандидаты уже применены или видны в Git-трейлерах." if patches.get("count") else "Zip-файлы патчей не найдены.")
        return "\n".join(lines) + "\n"

    def _format_plan(self, data: dict, result: RunResult) -> str:
        if not isinstance(data, dict) or not data:
            return (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        lines = ["== dry-run план =="]
        if not data.get("ok"):
            lines.append("План содержит ошибку или патч некорректен.")
            validation = data.get("validation") or {}
            if validation.get("error"):
                lines.append(f"Ошибка валидации: {validation.get('error')}")
            elif data.get("error"):
                lines.append(f"Ошибка: {data.get('error')}")
        patch = data.get("patch") or {}
        manifest = data.get("manifest") or {}
        if patch:
            lines.extend(["", "== патч ==", f"Файл: {patch.get('name')}", f"ID: {manifest.get('patchId') or patch.get('patchId') or 'неизвестно'}", f"Название: {manifest.get('title') or patch.get('title') or 'неизвестно'}", f"SHA-256: {patch.get('sha256') or 'неизвестно'}", f"Сводка: {manifest.get('summary') or ''}"])
        apply = data.get("apply") or {}
        lines.extend(["", "== применение ==", f"Корень файлов: {apply.get('filesRoot') or 'files'}", f"Файлов к копированию: {apply.get('copyCount', 0)}"])
        for name in (apply.get("copyFiles") or [])[:120]:
            lines.append(f"  + {name}")
        if len(apply.get("copyFiles") or []) > 120:
            lines.append("  ...")
        lines.append(f"Путей к удалению: {apply.get('deleteCount', 0)}")
        for entry in (apply.get("deletePaths") or [])[:120]:
            lines.append(f"  - {entry.get('path')} recursive={entry.get('recursive', False)} required={entry.get('required', False)}")
        lines.append(""); lines.append("== проверки ==")
        checks = data.get("checks") or []
        if checks:
            for check in checks:
                lines.append(f"  - {check.get('name')}: {check.get('command')} [cwd={check.get('cwd')}]")
        else:
            lines.append("Проверки не объявлены.")
        commit = data.get("commit") or {}
        push = data.get("push") or {}
        lines.extend(["", "== commit / push ==", "Политика: проверки -> commit -> push", f"Сообщение коммита: {commit.get('message') or ''}", f"Push включён: {push.get('enabled')}", f"Цель push: {push.get('remote')}/{push.get('branch')}", f"Примечание: {push.get('note') or ''}", "", "Файлы не изменялись. Для выполнения нажмите «Запустить конвейер»."])
        return "\n".join(lines) + "\n"

    def start_pipeline(self, no_push: bool) -> None:
        self._save_workspace()
        plan = self.last_plan
        if not plan or not plan.get("patch"):
            self.build_plan()
            if not self._confirm("План обновляется. Запустить конвейер после текущей проверки лучше вручную. Всё равно запустить сейчас?"):
                return
        else:
            patch = plan.get("patch", {})
            push = plan.get("push", {})
            target = f"{push.get('remote')}/{push.get('branch')}" if push else "неизвестно"
            message = "\n".join([f"Патч: {patch.get('title') or patch.get('name')}", f"Файл: {patch.get('name')}", f"Push: {'отключён вручную' if no_push else target}", "", "Запустить конвейер сейчас?"])
            if not self._confirm(message, "Подтверждение запуска"):
                return
        self.set_running(True)
        self.set_text(self.run_text, "")
        self.notebook.setCurrentIndex(1)
        args = ["start", "--json"]
        if no_push:
            args.append("--no-push")
        def on_line(line: str) -> None:
            self.events.put(("line", line))
        def on_done(result: RunResult) -> None:
            self.events.put(("done", result))
        self.runner.stream(args, on_line, on_done)

    def set_running(self, running: bool) -> None:
        enabled = not bool(running)
        for widget in (self.main_button, *self.action_buttons, self.init_top_btn, self.settings_top_btn):
            widget.setEnabled(enabled)
        if hasattr(self, "workspace_combo"):
            self.workspace_combo.setEnabled(enabled and bool(registered_workspaces()))

    def _on_start_done(self, result: RunResult) -> None:
        self.set_running(False)
        data = result.json_data or {}
        if isinstance(data, dict):
            self.last_report_path = data.get("reportPath") or self.last_report_path
            self.last_archive_path = data.get("archivePath") or self.last_archive_path
            self.last_uts_path = data.get("utsProjectDir") or self.last_uts_path
            self.set_text(self.report_text, self._format_result(data, result))
            self._recommend_after_result(data, result)
        else:
            self.set_text(self.report_text, result.stdout + ("\n" + result.stderr if result.stderr else ""))
            self._set_next_action("refresh_status", "Проверить состояние после запуска", "Конвейер завершился без машинно-читаемого JSON. Обновите статус и проверьте лог запуска.", "Обновить статус", "warn")
        self.notebook.setCurrentIndex(2)
        self.refresh_status()
        if result.returncode == 0:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Конвейер завершён успешно.")
        else:
            QtWidgets.QMessageBox.critical(self, APP_NAME, "Конвейер завершился с ошибкой. Подробности во вкладке «Отчёт» и в логах запуска.")

    def _format_result(self, data: dict, result: RunResult) -> str:
        lines = ["== результат запуска ==", f"Статус: {data.get('status')}", f"Код возврата: {data.get('returncode', result.returncode)}", f"Отчёт: {data.get('reportPath') or 'нет'}", f"Каталог архива: {data.get('archivePath') or 'нет'}", f"Коммит: {data.get('commitSha') or 'нет'}", f"Push: {data.get('pushResult') or 'нет'}", f"Auto-reset: {data.get('autoResetPerformed')}", f"Удалённый bad patch: {data.get('badPatchDeleted') or 'нет'}", f"UTS project: {data.get('utsProjectDir') or 'нет'}", f"Bytecode очищено: {len(data.get('cleanedBytecodePaths') or [])}", "", "== предупреждения =="]
        lines.extend([f"- {item}" for item in data.get("warnings") or []] or ["нет"])
        lines.append(""); lines.append("== ошибки ==")
        lines.extend([f"- {item}" for item in data.get("errors") or []] or ["нет"])
        return "\n".join(lines) + "\n"

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "result":
                    callback, result = payload  # type: ignore[misc]
                    self._guard(callback, result)
                elif kind == "line":
                    self.append_run(str(payload))
                elif kind == "done":
                    self._guard(self._on_start_done, payload)
        except queue.Empty:
            pass

    def _confirm(self, text: str, title: str = APP_NAME) -> bool:
        answer = QtWidgets.QMessageBox.question(
            self,
            title,
            text,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def open_report(self) -> None:
        open_path(self.last_report_path)

    def open_archives(self) -> None:
        if self.last_archive_path:
            open_path(self.last_archive_path)
            return
        data = self.last_status or {}
        workspace = data.get("workspace", {}) if isinstance(data, dict) else {}
        open_path(workspace.get("archivesDir"))

    def open_patches(self) -> None:
        data = self.last_status or {}
        workspace = data.get("workspace", {}) if isinstance(data, dict) else {}
        open_path(workspace.get("patchesDir"))

    def open_uts(self) -> None:
        if self.last_uts_path:
            open_path(self.last_uts_path)
            return
        data = self.last_status or {}
        workspace = data.get("workspace", {}) if isinstance(data, dict) else {}
        open_path(workspace.get("userTestSpaceDir"))

    def open_project(self) -> None:
        data = self.last_status or {}
        workspace = data.get("workspace", {}) if isinstance(data, dict) else {}
        open_path(workspace.get("projectRoot"))


def run_gui() -> int:
    if QT_IMPORT_ERROR is not None or QtWidgets is None:
        print(
            "[ОШИБКА] devctl GUI v0.5.4 требует PySide6.\n"
            "Установите зависимость: python -m pip install PySide6\n"
            f"Исходная ошибка: {QT_IMPORT_ERROR}",
            file=sys.stderr,
        )
        return 2
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    icon = bundled_root() / "gui" / "assets" / "icon.ico"
    if icon.exists() and QtGui is not None:
        app.setWindowIcon(QtGui.QIcon(str(icon)))
    window = DevctlGui()
    window.show()
    return int(app.exec())


def main() -> int:
    configure_standard_streams()
    if len(sys.argv) > 1 and sys.argv[1] == "--devctl-child":
        return run_devctl_child(sys.argv[2:])
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
