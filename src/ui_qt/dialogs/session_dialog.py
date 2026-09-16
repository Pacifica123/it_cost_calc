from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui_qt.presenters.app_presenter import QtAppPresenter


class SessionDialog(QDialog):
    """Small lifecycle UI for self-contained resumable work sessions."""

    _CONTEXTS = ("защита", "обсуждение", "эксперимент", "работа")

    def __init__(self, presenter: QtAppPresenter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.presenter = presenter
        self.setWindowTitle("Сессии")
        self.resize(620, 420)
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.active_label = QLabel(self)
        self.active_label.setObjectName("pageStatus")
        layout.addWidget(self.active_label)

        self.list_widget = QListWidget(self)
        self.list_widget.itemDoubleClicked.connect(lambda _item: self._switch_selected())
        layout.addWidget(self.list_widget, 1)

        row = QHBoxLayout()
        actions = (
            ("Создать", self._create),
            ("Открыть", self._switch_selected),
            ("Переименовать", self._rename),
            ("Дублировать", self._duplicate),
            ("Удалить", self._delete),
        )
        for text, callback in actions:
            button = QPushButton(text, self)
            button.clicked.connect(callback)
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)

        self.message_label = QLabel("", self)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def refresh(self) -> None:
        active = self.presenter.active_session()
        self.active_label.setText(
            f"Активная: {active.name} · {active.context_type} · revision {active.revision}"
        )
        selected_id = self._selected_id()
        self.list_widget.clear()
        selected_row = -1
        for row, session in enumerate(self.presenter.list_sessions()):
            prefix = "● " if session.active else "   "
            item = QListWidgetItem(
                f"{prefix}{session.name}  [{session.context_type}]  r{session.revision}"
            )
            item.setData(Qt.ItemDataRole.UserRole, session.session_id)
            item.setToolTip(
                f"Создана: {session.created_at}\nОбновлена: {session.updated_at}\n"
                f"Input revision: {session.input_revision}"
            )
            self.list_widget.addItem(item)
            if session.session_id == selected_id or (selected_id is None and session.active):
                selected_row = row
        if selected_row >= 0:
            self.list_widget.setCurrentRow(selected_row)

    def _selected_id(self) -> str | None:
        item = self.list_widget.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return str(value) if value else None

    def _create(self) -> None:
        name, ok = QInputDialog.getText(self, "Новая сессия", "Название:")
        if not ok or not name.strip():
            return
        context, ok = QInputDialog.getItem(
            self,
            "Контекст",
            "Для чего эта сессия:",
            self._CONTEXTS,
            1,
            True,
        )
        if not ok:
            return
        self._run(lambda: self.presenter.create_session(name.strip(), context_type=str(context)))

    def _switch_selected(self) -> None:
        session_id = self._selected_id()
        if session_id:
            self._run(lambda: self.presenter.switch_session(session_id))

    def _rename(self) -> None:
        session_id = self._selected_id()
        if not session_id:
            return
        current = next(
            (item for item in self.presenter.list_sessions() if item.session_id == session_id), None
        )
        if current is None:
            return
        name, ok = QInputDialog.getText(
            self, "Переименовать сессию", "Название:", text=current.name
        )
        if ok and name.strip():
            self._run(lambda: self.presenter.rename_session(session_id, name.strip()))

    def _duplicate(self) -> None:
        session_id = self._selected_id()
        if not session_id:
            return
        current = next(
            (item for item in self.presenter.list_sessions() if item.session_id == session_id), None
        )
        if current is None:
            return
        name, ok = QInputDialog.getText(
            self,
            "Дублировать сессию",
            "Название копии:",
            text=f"{current.name} — копия",
        )
        if ok and name.strip():
            self._run(lambda: self.presenter.duplicate_session(session_id, name=name.strip()))

    def _delete(self) -> None:
        session_id = self._selected_id()
        if not session_id:
            return
        current = next(
            (item for item in self.presenter.list_sessions() if item.session_id == session_id), None
        )
        if current is None:
            return
        answer = QMessageBox.question(
            self,
            "Удалить сессию",
            f"Удалить «{current.name}» вместе с её снимком и историей экспорта?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._run(lambda: self.presenter.delete_session(session_id))

    def _run(self, action) -> None:
        try:
            action()
        except Exception as exc:  # pragma: no cover - GUI status fallback
            self.message_label.setText(str(exc))
            return
        self.message_label.setText("")
        self.refresh()


__all__ = ["SessionDialog"]
