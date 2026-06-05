from __future__ import annotations

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from ui_qt.design import ThemeManager
from ui_qt.widgets import ActionBar, EmptyState, StatusStrip


class QtMainWindow(QMainWindow):
    """Minimal Qt shell with the shared design system attached."""

    def __init__(self) -> None:
        super().__init__()
        self._theme_manager = ThemeManager("light")
        app = QApplication.instance()
        if app is not None:
            self._theme_manager.apply_to_application(app)

        self.setWindowTitle("IT Cost Calc")
        self.resize(1100, 720)
        self.setMinimumSize(860, 560)
        self.setCentralWidget(self._build_placeholder())

    def _build_placeholder(self) -> QWidget:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        action_bar = ActionBar("Qt UI", root, status="Каркас готов")
        action_bar.add_secondary_action("Светлая", lambda: self._apply_theme("light"))
        action_bar.add_secondary_action("Тёмная", lambda: self._apply_theme("dark"))

        placeholder = EmptyState(
            "Qt-интерфейс готов",
            root,
            status="Дизайн-система подключена",
            details=(
                "Этот экран подтверждает, что новый визуальный слой подключён к "
                "параллельному Qt-контуру без переноса расчётной логики."
            ),
        )
        self.status_strip = StatusStrip("Готово", root)

        layout.addWidget(action_bar, 0)
        layout.addWidget(placeholder, 1)
        layout.addWidget(self.status_strip, 0)
        return root

    def _apply_theme(self, theme_name: str) -> None:
        app = QApplication.instance()
        if app is None:
            return
        applied = self._theme_manager.apply_to_application(app, theme_name)
        self.status_strip.set_message("Тема: тёмная" if applied == "dark" else "Тема: светлая")
