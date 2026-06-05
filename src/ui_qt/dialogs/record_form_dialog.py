from __future__ import annotations

from typing import Any, Mapping, Sequence

from ui_qt.dialogs.form_schema import FieldSpec, normalize_field_specs, payload_from_strings
from ui_qt.text_rules import assert_short_text

try:
    from PySide6.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QLineEdit,
        QMessageBox,
        QVBoxLayout,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    QDialog = object  # type: ignore[assignment,misc]
    QDialogButtonBox = QFormLayout = QLineEdit = QMessageBox = QVBoxLayout = None  # type: ignore[assignment]


class RecordFormDialog(QDialog):  # type: ignore[misc,valid-type]
    """Compact Qt record dialog for future CRUD screens.

    Field labels are expected to be short.  Long field explanations should be
    passed as tooltips via ``help`` in a field descriptor.
    """

    def __init__(
        self,
        parent: Any,
        title: str,
        fields: Sequence[FieldSpec | Mapping[str, Any]],
        initial_values: Mapping[str, Any] | None = None,
    ) -> None:
        if QLineEdit is None:
            raise RuntimeError("PySide6 is required to create RecordFormDialog")
        assert_short_text(title, field="RecordFormDialog.title")
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(420)
        self.result_payload: dict[str, Any] | None = None
        self._fields = normalize_field_specs(fields)
        self._entries: dict[str, QLineEdit] = {}
        self._initial_values = dict(initial_values or {})
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        for field in self._fields:
            entry = QLineEdit(self)
            name = field.name
            entry.setText(str(self._initial_values.get(name, field.default)))
            if field.placeholder:
                entry.setPlaceholderText(field.placeholder)
            if field.help:
                entry.setToolTip(field.help)
            self._entries[name] = entry
            form.addRow(field.label, entry)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)

        layout.addLayout(form)
        layout.addWidget(buttons)

    def _on_save(self) -> None:
        try:
            raw_values = {name: entry.text() for name, entry in self._entries.items()}
            self.result_payload = payload_from_strings(self._fields, raw_values)
            self.accept()
        except ValueError as error:
            QMessageBox.warning(self, "Ошибка ввода", str(error))

    def show_dialog(self) -> dict[str, Any] | None:
        if self.exec() == QDialog.DialogCode.Accepted:
            return self.result_payload
        return None
