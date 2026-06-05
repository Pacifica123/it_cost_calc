from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QSizePolicy, QVBoxLayout, QWidget


class QtMainWindow(QMainWindow):
    """Minimal empty Qt shell for the parallel migration path."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IT Cost Calc")
        self.resize(1100, 720)
        self.setMinimumSize(860, 560)
        self.setCentralWidget(self._build_placeholder())

    def _build_placeholder(self) -> QWidget:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(12)

        title = QLabel("Qt-интерфейс готов")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        status = QLabel("Каркас без переноса логики")
        status.setObjectName("pageStatus")
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        status.setToolTip(
            "Этот экран подтверждает, что параллельный Qt-контур запускается отдельно от Tkinter."
        )

        spacer = QWidget(root)
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout.addWidget(spacer)
        layout.addWidget(title)
        layout.addWidget(status)
        layout.addWidget(spacer)
        return root
