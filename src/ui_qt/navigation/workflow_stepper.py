from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ui_qt.widgets.text_rules import assert_short_text

try:
    from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QWidget
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    QButtonGroup = QHBoxLayout = QPushButton = None  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class WorkflowStep:
    step_id: str
    label: str
    enabled: bool = False
    tooltip: str = ""

    def __post_init__(self) -> None:
        assert_short_text(self.label, field="WorkflowStep.label")


DEFAULT_DECISION_STEPS: tuple[WorkflowStep, ...] = (
    WorkflowStep("data", "Данные", enabled=True),
    WorkflowStep("ga", "GA", tooltip="Будет перенесено следующим этапом."),
    WorkflowStep("ahp", "AHP", tooltip="Будет доступно после GA."),
    WorkflowStep("pareto", "Pareto", tooltip="Будет доступно после AHP."),
    WorkflowStep("hybrid", "Гибрид", tooltip="Итог появится после Pareto."),
)


class WorkflowStepper(QWidget):  # type: ignore[misc,valid-type]
    """Compact visual route for ПО/ТО decision workspaces."""

    def __init__(
        self,
        parent: QWidget | None = None,  # type: ignore[valid-type]
        *,
        steps: Iterable[WorkflowStep] = DEFAULT_DECISION_STEPS,
        active_step_id: str = "data",
    ) -> None:
        if QPushButton is None:
            raise RuntimeError("PySide6 is required to create WorkflowStepper")
        super().__init__(parent)
        self.setObjectName("surface")
        self._buttons: dict[str, QPushButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        for step in steps:
            button = QPushButton(step.label, self)
            button.setObjectName("workflowStep")
            button.setCheckable(True)
            button.setEnabled(step.enabled)
            if step.tooltip:
                button.setToolTip(step.tooltip)
            self._group.addButton(button)
            self._buttons[step.step_id] = button
            layout.addWidget(button, 0)
        layout.addStretch(1)
        self.set_active(active_step_id)

    def set_active(self, step_id: str) -> None:
        button = self._buttons.get(step_id)
        if button is None:
            raise ValueError(f"Unknown workflow step: {step_id!r}")
        button.setChecked(True)
