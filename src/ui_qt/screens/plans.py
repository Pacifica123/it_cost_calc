from __future__ import annotations

from dataclasses import asdict

from ui_qt.models import RowTableModel
from ui_qt.presenters import QtAppPresenter
from ui_qt.presenters.implementation_presenter import (
    SHIFT_FIELDS,
    ImplementationPresenter,
    ImplementationSettings,
)
from ui_qt.presenters.rooms_presenter import ROOM_COLUMNS, RoomsPresenter, RoomsSettings, room_from_mapping
from ui_qt.presenters.licenses_presenter import LicenseSettings, LicensesPresenter, parse_date, ru_date
from ui_qt.presenters.financing_presenter import FINANCING_FIELDS, FinancingPresenter, FinancingSettings
from ui_qt.presenters.planning_presenter import PlanningInput, PlanningPresenter
from ui_qt.widgets import CompactLabel, InfoHint
from ui_qt.widgets.analysis_chart import AnalysisChart

try:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import (
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QPushButton,
        QScrollArea,
        QCheckBox,
        QTableView,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    QTimer = None  # type: ignore[assignment]
    QFrame = QGridLayout = QHBoxLayout = QHeaderView = QLabel = QLineEdit = None  # type: ignore[assignment]
    QPushButton = QScrollArea = QTableView = QTabWidget = QVBoxLayout = None  # type: ignore[assignment]
    QCheckBox = QTableWidget = QTableWidgetItem = None  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment,misc]


# (field, label, tooltip) of the shared inputs; money per year, growth and rate in percent.
PLAN_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("one_time", "Разово, ₽", "CAPEX и разовые затраты, платятся в начале (год 0)."),
    ("opex_annual", "OPEX/год, ₽", "Регулярные затраты за год по сегодняшним ценам."),
    ("electricity_annual", "Энергия/год, ₽", "Электроэнергия за год по сегодняшнему тарифу."),
    ("horizon_years", "Горизонт, лет", "Срок планирования, 1–30 лет."),
    ("opex_growth_percent", "Рост OPEX, %/год", "Инфляция регулярных затрат."),
    ("electricity_growth_percent", "Рост тарифа, %/год", "Ежегодная индексация тарифа на электроэнергию."),
    ("discount_rate_percent", "Ставка, %", "Ставка дисконтирования для приведённой стоимости."),
    ("price_growth_percent", "Рост цен техники, %/год", "Рост цены при повторной покупке в плане замены."),
    ("server_months", "Срок сервера/сети, мес", "Срок службы серверов и сетевого оборудования."),
    ("hardware_months", "Срок ПК/прочего, мес", "Срок службы рабочих мест и прочей техники."),
    ("software_months", "Срок лицензий, мес", "Период продления лицензий ПО."),
)


class PlansScreen(QWidget):  # type: ignore[misc,valid-type]
    """«Планы»: TCO по годам, амортизация, замена, финансирование, лицензии, внедрение, риски, помещения."""

    def __init__(self, app_presenter: QtAppPresenter | None = None, parent: QWidget | None = None) -> None:  # type: ignore[valid-type]
        if QLineEdit is None:
            raise RuntimeError("PySide6 is required to create PlansScreen")
        super().__init__(parent)
        self.presenter = PlanningPresenter(app_presenter)
        self.inputs: dict[str, QLineEdit] = {}  # type: ignore[valid-type]
        self.financing_presenter = FinancingPresenter(self.presenter.app_presenter)
        self.financing_inputs: dict[str, QLineEdit] = {}  # type: ignore[valid-type]
        self.lease_model = RowTableModel(
            columns=("title", "upfront", "monthly", "total", "pv"),
            headers={"title": "Вариант", "upfront": "Сразу", "monthly": "В месяц", "total": "Всего", "pv": "Приведённо"},
        )
        self.staff_model = RowTableModel(columns=("name", "value"), headers={"name": "Показатель", "value": "Значение"})
        self.licenses_presenter = LicensesPresenter(self.presenter.app_presenter)
        self.implementation_presenter = ImplementationPresenter(self.presenter.app_presenter)
        self.rooms_presenter = RoomsPresenter(self.presenter.app_presenter)
        self._rooms_custom = False
        self._filling_rooms = False
        self.room_result_model = RowTableModel(
            columns=("name", "area", "lan", "climate", "light", "total"),
            headers={"name": "Помещение", "area": "Площадь", "lan": "Линии / кабель", "climate": "Охлаждение", "light": "Освещение", "total": "Сумма"},
        )
        self.room_lines_model = RowTableModel(
            columns=("group", "name", "quantity", "price", "total"),
            headers={"group": "Раздел", "name": "Позиция", "quantity": "Кол-во", "price": "Цена", "total": "Сумма"},
        )
        self.duration_inputs: dict[str, QLineEdit] = {}  # type: ignore[valid-type]
        self.shift_inputs: dict[str, QLineEdit] = {}  # type: ignore[valid-type]
        self.schedule_model = RowTableModel(
            columns=("title", "section", "duration", "dates", "slack"),
            headers={"title": "Этап", "section": "Раздел", "duration": "Дней", "dates": "Сроки", "slack": "Резерв"},
        )
        self.risk_model = RowTableModel(
            columns=("title", "category", "pi", "level", "response"),
            headers={"title": "Риск", "category": "Область", "pi": "P × I", "level": "Уровень", "response": "Мера"},
        )
        self.scenario_model = RowTableModel(
            columns=("title", "one_time", "running", "tco", "npv", "irr"),
            headers={"title": "Сценарий", "one_time": "Разово", "running": "В год", "tco": "TCO", "npv": "NPV", "irr": "IRR"},
        )
        self.license_model = RowTableModel(
            columns=("name", "kind", "period", "amount", "next"),
            headers={"name": "Позиция", "kind": "Вид", "period": "Период", "amount": "Платёж", "next": "Ближайшие даты"},
        )
        self.license_year_model = RowTableModel(
            columns=("year", "licenses", "subscriptions", "total"),
            headers={"year": "Год", "licenses": "Продления", "subscriptions": "Подписки", "total": "Итого"},
        )
        self.local_cloud_model = RowTableModel(
            columns=("year", "local", "cloud"),
            headers={"year": "Год", "local": "Локально", "cloud": "Облако"},
        )
        self.tco_model = RowTableModel(
            columns=("year", "opex", "electricity", "total", "discounted", "cumulative"),
            headers={
                "year": "Год",
                "opex": "OPEX",
                "electricity": "Энергия",
                "total": "Итого за год",
                "discounted": "Приведённо",
                "cumulative": "Накопленно",
            },
        )
        self.amortization_model = RowTableModel(
            columns=("name", "kind", "quantity", "total", "months", "monthly"),
            headers={
                "name": "Позиция",
                "kind": "Вид",
                "quantity": "Кол-во",
                "total": "Стоимость",
                "months": "Срок, мес",
                "monthly": "В месяц",
            },
        )
        self.replacement_model = RowTableModel(
            columns=("year", "items", "total"),
            headers={"year": "Год", "items": "Что покупать", "total": "Сумма"},
        )
        self._build_ui()
        self._setup_auto_recalculation()
        self.refresh_data()

    # --- layout ----------------------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_header(), 0)

        scroll = QScrollArea(self)
        scroll.setObjectName("contentArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget(scroll)
        content.setObjectName("contentArea")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)
        content_layout.addWidget(self._build_inputs(), 0)
        self.tabs = QTabWidget(content)
        self._build_tabs()
        content_layout.addWidget(self.tabs, 1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        layout.addLayout(self._build_actions(), 0)

    def _build_tabs(self) -> None:
        self.tabs.addTab(self._build_tco_tab(), "TCO по годам")
        self.tabs.addTab(self._build_amortization_tab(), "Амортизация")
        self.tabs.addTab(self._build_replacement_tab(), "План замены")
        self.tabs.addTab(self._build_financing_tab(), "Финансирование")
        self.tabs.addTab(self._build_licenses_tab(), "Лицензии")
        self.tabs.addTab(self._build_implementation_tab(), "Внедрение")
        self.tabs.addTab(self._build_risks_tab(), "Риски и сценарии")
        self.tabs.addTab(self._build_rooms_tab(), "Помещения")

    def _build_header(self) -> QWidget:  # type: ignore[valid-type]
        header = QFrame(self)
        header.setObjectName("surface")
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 8, 12, 8)
        row.addWidget(CompactLabel("Планы", header), 0)
        row.addWidget(
            InfoHint(
                "Многолетний взгляд на выбранную конфигурацию: затраты по годам с ростом цен, "
                "амортизация и бюджет замены оборудования.",
                header,
            ),
            0,
        )
        row.addStretch(1)
        self.status_label = CompactLabel("Нет данных", header)
        row.addWidget(self.status_label, 0)
        return header

    def _build_inputs(self) -> QWidget:  # type: ignore[valid-type]
        panel = QFrame(self)
        panel.setObjectName("surface")
        grid = QGridLayout(panel)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        for index, (key, label, tooltip) in enumerate(PLAN_FIELDS):
            line = QLineEdit(panel)
            line.setMinimumHeight(30)
            line.setToolTip(tooltip)
            self.inputs[key] = line
            row, column = divmod(index, 3)
            caption = CompactLabel(label, panel)
            caption.setToolTip(tooltip)
            grid.addWidget(caption, row, column * 2)
            grid.addWidget(line, row, column * 2 + 1)
        for column in (1, 3, 5):
            grid.setColumnStretch(column, 1)
        return panel

    def _table(self, model: RowTableModel, parent: QWidget, height: int = 220) -> QTableView:  # type: ignore[valid-type]
        table = QTableView(parent)
        table.setModel(model)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setMinimumHeight(height)
        table.setWordWrap(True)
        return table

    def _tab_page(self) -> tuple[QWidget, QVBoxLayout]:  # type: ignore[valid-type]
        page = QFrame(self)
        page.setObjectName("card")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        return page, layout

    def _note(self, parent: QWidget) -> QLabel:  # type: ignore[valid-type]
        label = QLabel("—", parent)
        label.setWordWrap(True)
        return label

    def _build_tco_tab(self) -> QWidget:  # type: ignore[valid-type]
        page, layout = self._tab_page()
        self.tco_note = self._note(page)
        layout.addWidget(self.tco_note)
        self.tco_chart = AnalysisChart(page)
        layout.addWidget(self.tco_chart)
        self.tco_bars = AnalysisChart(page)
        layout.addWidget(self.tco_bars)
        self.tco_table = self._table(self.tco_model, page)
        layout.addWidget(self.tco_table)
        layout.addStretch(1)
        return page

    def _build_amortization_tab(self) -> QWidget:  # type: ignore[valid-type]
        page, layout = self._tab_page()
        self.amortization_note = self._note(page)
        layout.addWidget(self.amortization_note)
        self.amortization_table = self._table(self.amortization_model, page, 260)
        layout.addWidget(self.amortization_table)
        layout.addStretch(1)
        return page

    def _build_replacement_tab(self) -> QWidget:  # type: ignore[valid-type]
        page, layout = self._tab_page()
        self.replacement_note = self._note(page)
        layout.addWidget(self.replacement_note)
        self.replacement_chart = AnalysisChart(page)
        layout.addWidget(self.replacement_chart)
        self.replacement_table = self._table(self.replacement_model, page, 340)
        header = self.replacement_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.replacement_table)
        layout.addStretch(1)
        return page

    def _build_financing_tab(self) -> QWidget:  # type: ignore[valid-type]
        page, layout = self._tab_page()
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        for index, (key, label, tooltip) in enumerate(FINANCING_FIELDS):
            line = QLineEdit(page)
            line.setMinimumHeight(28)
            line.setToolTip(tooltip)
            self.financing_inputs[key] = line
            row, column = divmod(index, 3)
            caption = CompactLabel(label, page)
            caption.setToolTip(tooltip)
            grid.addWidget(caption, row, column * 2)
            grid.addWidget(line, row, column * 2 + 1)
        for column in (1, 3, 5):
            grid.setColumnStretch(column, 1)
        layout.addLayout(grid)

        layout.addWidget(CompactLabel("Покупка, лизинг или аренда техники", page))
        self.lease_note = self._note(page)
        layout.addWidget(self.lease_note)
        self.lease_chart = AnalysisChart(page)
        layout.addWidget(self.lease_chart)
        self.lease_table = self._table(self.lease_model, page, 130)
        layout.addWidget(self.lease_table)

        layout.addWidget(CompactLabel("Свой ИТ-штат или аутсорсинг", page))
        self.staff_note = self._note(page)
        layout.addWidget(self.staff_note)
        self.staff_table = self._table(self.staff_model, page, 170)
        layout.addWidget(self.staff_table)

        layout.addWidget(CompactLabel("Локальная инфраструктура или облако", page))
        self.local_cloud_note = self._note(page)
        layout.addWidget(self.local_cloud_note)
        self.local_cloud_chart = AnalysisChart(page)
        layout.addWidget(self.local_cloud_chart)
        self.local_cloud_table = self._table(self.local_cloud_model, page, 200)
        layout.addWidget(self.local_cloud_table)
        layout.addStretch(1)
        return page

    def _build_licenses_tab(self) -> QWidget:  # type: ignore[valid-type]
        page, layout = self._tab_page()
        row = QHBoxLayout()
        row.addWidget(CompactLabel("Дата покупки", page), 0)
        self.license_start_input = QLineEdit(page)
        self.license_start_input.setPlaceholderText("ДД.ММ.ГГГГ")
        self.license_start_input.setToolTip("От этой даты считаются продления. Срок лицензии берётся из поля «Срок лицензий, мес».")
        row.addWidget(self.license_start_input, 1)
        row.addWidget(CompactLabel("Напомнить за, дней", page), 0)
        self.license_remind_input = QLineEdit(page)
        row.addWidget(self.license_remind_input, 1)
        self.ics_button = QPushButton("Сохранить .ics", page)
        self.ics_button.setToolTip("Календарь продлений с напоминаниями для Outlook, Google Календаря и телефона.")
        self.ics_button.clicked.connect(self.export_ics)
        row.addWidget(self.ics_button, 0)
        layout.addLayout(row)
        self.license_note = self._note(page)
        layout.addWidget(self.license_note)
        self.license_table = self._table(self.license_model, page, 160)
        layout.addWidget(self.license_table)
        self.license_chart = AnalysisChart(page)
        layout.addWidget(self.license_chart)
        self.license_year_table = self._table(self.license_year_model, page, 160)
        layout.addWidget(self.license_year_table)
        layout.addStretch(1)
        return page

    def _build_implementation_tab(self) -> QWidget:  # type: ignore[valid-type]
        page, layout = self._tab_page()
        row = QHBoxLayout()
        row.addWidget(CompactLabel("Начало внедрения", page), 0)
        self.implementation_start_input = QLineEdit(page)
        self.implementation_start_input.setPlaceholderText("ДД.ММ.ГГГГ")
        row.addWidget(self.implementation_start_input, 1)
        row.addStretch(1)
        layout.addLayout(row)
        self.durations_grid = QGridLayout()
        self.durations_grid.setHorizontalSpacing(10)
        layout.addWidget(CompactLabel("Длительность этапов, рабочих дней (можно менять)", page))
        layout.addLayout(self.durations_grid)
        self.schedule_note = self._note(page)
        layout.addWidget(self.schedule_note)
        self.gantt_chart = AnalysisChart(page, height=300)
        layout.addWidget(self.gantt_chart)
        self.schedule_table = self._table(self.schedule_model, page, 300)
        layout.addWidget(self.schedule_table)
        layout.addStretch(1)
        return page

    def _build_risks_tab(self) -> QWidget:  # type: ignore[valid-type]
        page, layout = self._tab_page()
        self.risk_note = self._note(page)
        layout.addWidget(self.risk_note)
        row = QHBoxLayout()
        row.addWidget(CompactLabel("Бюджет проекта, ₽ (0 — не задан)", page), 0)
        self.budget_input = QLineEdit(page)
        row.addWidget(self.budget_input, 1)
        row.addStretch(1)
        layout.addLayout(row)
        self.risk_matrix_chart = AnalysisChart(page, height=260)
        layout.addWidget(self.risk_matrix_chart)
        self.risk_table = self._table(self.risk_model, page, 440)
        header = self.risk_table.horizontalHeader()
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.risk_table)
        layout.addWidget(CompactLabel("Сценарии", page))
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        for index, (key, label) in enumerate(SHIFT_FIELDS):
            line = QLineEdit(page)
            self.shift_inputs[key] = line
            row_index, column = divmod(index, 3)
            grid.addWidget(CompactLabel(label, page), row_index, column * 2)
            grid.addWidget(line, row_index, column * 2 + 1)
        layout.addLayout(grid)
        self.scenario_note = self._note(page)
        layout.addWidget(self.scenario_note)
        self.scenario_table = self._table(self.scenario_model, page, 150)
        layout.addWidget(self.scenario_table)
        layout.addStretch(1)
        return page

    def read_implementation_settings(self) -> ImplementationSettings:
        start = parse_date(self.implementation_start_input.text())
        durations = {}
        for key, line in self.duration_inputs.items():
            text = line.text().strip()
            if text:
                durations[key] = text
        shifts = {}
        for key, line in self.shift_inputs.items():
            text = line.text().replace(" ", "").replace(",", ".")
            if text:
                shifts[key] = text
        return ImplementationSettings.from_mapping(
            {
                "start": start.isoformat() if start else "",
                "durations": durations,
                "shifts": shifts,
                "budget": self.budget_input.text().replace(" ", "").replace(",", ".") or 0,
            }
        )

    def _apply_implementation_settings(self, settings: ImplementationSettings) -> None:
        self.implementation_start_input.setText(ru_date(settings.start))
        self._saved_durations = dict(settings.durations)
        for key, value in settings.shifts.items():
            if key in self.shift_inputs:
                self.shift_inputs[key].setText(f"{value:g}")
        self.budget_input.setText(f"{settings.budget:g}")

    def _sync_duration_inputs(self, view) -> None:
        keys = [task.key for task in view.schedule.tasks]
        if keys == list(self.duration_inputs):
            for task in view.schedule.tasks:
                line = self.duration_inputs[task.key]
                if not line.hasFocus():
                    line.setText(str(task.duration))
            return
        while self.durations_grid.count():
            item = self.durations_grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.duration_inputs.clear()
        for index, task in enumerate(view.schedule.tasks):
            line = QLineEdit(str(task.duration), self)
            line.setMaximumWidth(70)
            line.textEdited.connect(lambda _text: self._recalc_timer.start() if hasattr(self, "_recalc_timer") else None)
            self.duration_inputs[task.key] = line
            row, column = divmod(index, 3)
            self.durations_grid.addWidget(CompactLabel(task.title, self), row, column * 2)
            self.durations_grid.addWidget(line, row, column * 2 + 1)

    def _sync_implementation(self, values: PlanningInput, save: bool) -> None:
        settings = self.read_implementation_settings()
        if not self.duration_inputs and getattr(self, "_saved_durations", None):
            settings = ImplementationSettings(settings.start, dict(self._saved_durations), settings.shifts, settings.budget)
        if save:
            self.implementation_presenter.save_settings(settings)
        view = self.implementation_presenter.build(self.presenter.components, values, settings)
        self.last_implementation = view
        self._sync_duration_inputs(view)
        self.schedule_model.replace_rows(self.implementation_presenter.schedule_rows(view))
        self.schedule_note.setText(self.implementation_presenter.schedule_text(view))
        tasks = view.schedule.tasks
        self.gantt_chart.show_gantt(
            [task.title for task in tasks],
            [task.es for task in tasks],
            [task.duration for task in tasks],
            [task.critical for task in tasks],
            "Диаграмма Ганта",
        )
        self.risk_model.replace_rows(self.implementation_presenter.risk_rows(view))
        self.risk_table.resizeRowsToContents()
        self.risk_note.setText(self.implementation_presenter.risk_text(view))
        self.risk_matrix_chart.show_matrix(self.implementation_presenter.matrix(view), "Матрица рисков")
        self.scenario_model.replace_rows(self.implementation_presenter.scenario_rows(view))
        self.scenario_note.setText(self.implementation_presenter.scenario_text(view))

    def _build_rooms_tab(self) -> QWidget:  # type: ignore[valid-type]
        page, layout = self._tab_page()
        layout.addWidget(CompactLabel("Помещения (ячейки редактируются; «Окна» и «Серверная» — да/нет)", page))
        self.rooms_table = QTableWidget(0, len(ROOM_COLUMNS), page)
        self.rooms_table.setHorizontalHeaderLabels([title for _key, title in ROOM_COLUMNS])
        self.rooms_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.rooms_table.verticalHeader().setVisible(False)
        self.rooms_table.setMinimumHeight(150)
        self.rooms_table.itemChanged.connect(self._room_cell_changed)
        layout.addWidget(self.rooms_table)
        row = QHBoxLayout()
        for text, callback, tooltip in (
            ("Добавить", self.add_room, "Новое помещение 5 × 4 м."),
            ("Удалить", self.remove_room, "Удалить выбранное помещение."),
            ("По выбору", self.reset_rooms, "Офис и серверная по составу итоговой конфигурации."),
        ):
            button = QPushButton(text, page)
            button.setToolTip(tooltip)
            button.clicked.connect(callback)
            row.addWidget(button, 0)
        row.addStretch(1)
        row.addWidget(CompactLabel("ИБП, мин", page), 0)
        self.ups_minutes_input = QLineEdit(page)
        self.ups_minutes_input.setMaximumWidth(70)
        row.addWidget(self.ups_minutes_input, 0)
        self.ups_workstations_check = QCheckBox("ИБП и для ПК", page)
        self.ups_workstations_check.toggled.connect(lambda _checked: self._recalc_timer.start() if hasattr(self, "_recalc_timer") else None)
        row.addWidget(self.ups_workstations_check, 0)
        row.addWidget(CompactLabel("Тариф, ₽/кВт·ч", page), 0)
        self.rooms_tariff_input = QLineEdit(page)
        self.rooms_tariff_input.setMaximumWidth(70)
        row.addWidget(self.rooms_tariff_input, 0)
        layout.addLayout(row)
        self.rooms_note = self._note(page)
        layout.addWidget(self.rooms_note)
        self.rooms_chart = AnalysisChart(page)
        layout.addWidget(self.rooms_chart)
        self.room_result_table = self._table(self.room_result_model, page, 130)
        layout.addWidget(self.room_result_table)
        self.room_lines_table = self._table(self.room_lines_model, page, 380)
        header = self.room_lines_table.horizontalHeader()
        for column in (0, 2, 3, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.room_lines_table)
        layout.addStretch(1)
        return page

    def _fill_rooms_table(self, rooms) -> None:
        self._filling_rooms = True
        try:
            self.rooms_table.setRowCount(0)
            for room in rooms:
                self._append_room_row(room)
        finally:
            self._filling_rooms = False

    def _append_room_row(self, room) -> None:
        index = self.rooms_table.rowCount()
        self.rooms_table.insertRow(index)
        for column, (key, _title) in enumerate(ROOM_COLUMNS):
            value = getattr(room, key)
            if isinstance(value, bool):
                text = "да" if value else "нет"
            elif isinstance(value, float):
                text = f"{value:g}"
            else:
                text = str(value)
            self.rooms_table.setItem(index, column, QTableWidgetItem(text))

    def read_rooms(self) -> list:
        rooms = []
        for row in range(self.rooms_table.rowCount()):
            payload = {}
            for column, (key, _title) in enumerate(ROOM_COLUMNS):
                item = self.rooms_table.item(row, column)
                payload[key] = item.text() if item is not None else ""
            room = room_from_mapping(payload)
            if room is not None:
                rooms.append(room)
        return rooms

    def _room_cell_changed(self, _item) -> None:
        if self._filling_rooms:
            return
        self._rooms_custom = True
        if hasattr(self, "_recalc_timer"):
            self._recalc_timer.start()

    def add_room(self) -> None:
        from domain.planning.rooms import Room

        self._filling_rooms = True
        try:
            self._append_room_row(Room(f"Помещение {self.rooms_table.rowCount() + 1}", 5, 4, workstations=4))
        finally:
            self._filling_rooms = False
        self._rooms_custom = True
        self.calculate()

    def remove_room(self) -> None:
        row = self.rooms_table.currentRow()
        if row < 0:
            row = self.rooms_table.rowCount() - 1
        if row >= 0:
            self.rooms_table.removeRow(row)
            self._rooms_custom = True
            self.calculate()

    def reset_rooms(self) -> None:
        self._rooms_custom = False
        self._fill_rooms_table(self.rooms_presenter.default_rooms(self.presenter.components))
        self.calculate()

    def read_rooms_settings(self) -> RoomsSettings:
        return RoomsSettings.from_mapping(
            {
                "rooms": [],
                "ups_minutes": self.ups_minutes_input.text().replace(",", ".") or 15,
                "ups_for_workstations": self.ups_workstations_check.isChecked(),
                "tariff_per_kwh": self.rooms_tariff_input.text().replace(",", ".") or 8,
                "custom": self._rooms_custom,
            }
        )

    def _apply_rooms_settings(self, settings: RoomsSettings) -> None:
        self._rooms_custom = settings.custom
        rooms = settings.rooms if settings.custom else self.rooms_presenter.default_rooms(self.presenter.components)
        self._fill_rooms_table(rooms)
        self.ups_minutes_input.setText(f"{settings.ups_minutes:g}")
        self.ups_workstations_check.blockSignals(True)
        self.ups_workstations_check.setChecked(settings.ups_for_workstations)
        self.ups_workstations_check.blockSignals(False)
        self.rooms_tariff_input.setText(f"{settings.tariff_per_kwh:g}")

    def _sync_rooms(self, save: bool) -> None:
        base = self.read_rooms_settings()
        settings = RoomsSettings(tuple(self.read_rooms()), base.ups_minutes, base.ups_for_workstations, base.tariff_per_kwh, self._rooms_custom)
        if save:
            self.rooms_presenter.save_settings(settings)
        result = self.rooms_presenter.build(settings)
        self.last_rooms = result
        self.room_result_model.replace_rows(self.rooms_presenter.room_rows(result))
        self.room_lines_model.replace_rows(self.rooms_presenter.line_rows(result))
        self.rooms_note.setText(self.rooms_presenter.summary(result))
        if result.by_group:
            self.rooms_chart.show_stacked_bars(list(result.by_group), [("Сумма", list(result.by_group.values()))], "Инфраструктура помещений по разделам")
        else:
            self.rooms_chart.show_empty("Нет помещений")

    def read_license_settings(self) -> LicenseSettings:
        start = parse_date(self.license_start_input.text())
        payload = {"start": start.isoformat() if start else "", "remind_days": self.license_remind_input.text() or 14}
        return LicenseSettings.from_mapping(payload)

    def _sync_licenses(self, values: PlanningInput, save: bool) -> None:
        settings = self.read_license_settings()
        if save:
            self.licenses_presenter.save_settings(settings)
        view = self.licenses_presenter.build(self.presenter.components, values, settings)
        self.last_licenses = view
        self.license_model.replace_rows(self.licenses_presenter.source_rows(view))
        self.license_year_model.replace_rows(self.licenses_presenter.year_rows(view))
        self.license_note.setText(self.licenses_presenter.summary(view))
        self.ics_button.setEnabled(bool(view.sources))
        years = view.calendar.by_year
        if any(year.total for year in years):
            self.license_chart.show_stacked_bars(
                [str(year.year) for year in years],
                [("Продления", [year.licenses for year in years]), ("Подписки", [year.subscriptions for year in years])],
                "Платежи за лицензии по годам",
            )
        else:
            self.license_chart.show_empty("Нет платежей")

    def export_ics(self) -> None:
        view = getattr(self, "last_licenses", None)
        if view is None or not view.sources:
            return
        try:
            path = self.licenses_presenter.export_ics(view)
        except Exception as exc:  # pragma: no cover - GUI status fallback
            self.status_label.setText("Ошибка .ics")
            self.status_label.setToolTip(str(exc))
            return
        self.last_ics_path = path
        self.status_label.setText("Календарь сохранён")
        self.status_label.setToolTip(str(path))
        self.license_note.setText(self.licenses_presenter.summary(view) + f" Файл: {path}")

    def read_financing_settings(self) -> FinancingSettings:
        payload = {}
        for key, line in self.financing_inputs.items():
            text = line.text().replace(" ", "").replace(",", ".")
            if text:
                payload[key] = text
        return FinancingSettings.from_mapping(payload)

    def _apply_financing_settings(self, settings: FinancingSettings) -> None:
        for key, value in asdict(settings).items():
            if key in self.financing_inputs:
                self.financing_inputs[key].setText(f"{value:g}" if isinstance(value, float) else str(value))

    def _sync_financing(self, values: PlanningInput, save: bool) -> None:
        settings = self.read_financing_settings()
        if save:
            self.financing_presenter.save_settings(settings)
        view = self.financing_presenter.build(self.presenter.components, values, settings)
        self.last_financing = view
        self.lease_model.replace_rows(self.financing_presenter.lease_rows(view))
        self.lease_note.setText(self.financing_presenter.lease_text(view))
        options = view.lease_buy.options
        self.lease_chart.show_lines(
            list(range(len(options[0].curve))),
            [(option.title, option.curve) for option in options],
            "Накопленные затраты (приведённо)",
            x_label="Год",
        )
        self.staff_model.replace_rows(self.financing_presenter.staff_rows(view))
        self.staff_note.setText(self.financing_presenter.staff_text(view))
        self.local_cloud_model.replace_rows(self.financing_presenter.local_cloud_rows(view))
        self.local_cloud_note.setText(self.financing_presenter.local_cloud_text(view))
        result = view.local_cloud
        self.local_cloud_chart.show_lines(
            list(range(len(result.local_curve))),
            [("Локально", result.local_curve), ("Облако", result.cloud_curve)],
            "Локально или облако: накопленно",
            x_label="Год",
        )

    def _build_actions(self) -> QHBoxLayout:  # type: ignore[valid-type]
        row = QHBoxLayout()
        self.from_selection_button = QPushButton("Из выбора", self)
        self.from_selection_button.setToolTip("Взять CAPEX, OPEX, энергию и состав из итогового Hybrid-выбора ТО и ПО.")
        self.from_selection_button.clicked.connect(self.load_from_selection)
        row.addWidget(self.from_selection_button, 0)
        row.addStretch(1)
        self.calculate_button = QPushButton("Рассчитать", self)
        self.calculate_button.setProperty("role", "primary")
        self.calculate_button.clicked.connect(self.calculate)
        row.addWidget(self.calculate_button, 0)
        return row

    # --- behaviour -------------------------------------------------------------------------------

    def _setup_auto_recalculation(self) -> None:
        if QTimer is None:
            return
        self._recalc_timer = QTimer(self)
        self._recalc_timer.setSingleShot(True)
        self._recalc_timer.setInterval(300)
        self._recalc_timer.timeout.connect(self.calculate)
        for line in [
            *self.inputs.values(),
            *self.financing_inputs.values(),
            *self.shift_inputs.values(),
            self.license_start_input,
            self.license_remind_input,
            self.implementation_start_input,
            self.budget_input,
            self.ups_minutes_input,
            self.rooms_tariff_input,
        ]:
            line.textEdited.connect(lambda _text: self._recalc_timer.start())

    def refresh_data(self) -> None:
        values = self.presenter.saved_input()
        self.presenter.components = self.presenter.selected_components()
        if values.one_time == 0 and values.opex_annual == 0 and values.electricity_annual == 0:
            values = self.presenter.load_selection(values)
        self._apply_input(values)
        self._apply_financing_settings(self.financing_presenter.saved_settings())
        license_settings = self.licenses_presenter.saved_settings()
        self.license_start_input.setText(ru_date(license_settings.start))
        self.license_remind_input.setText(str(license_settings.remind_days))
        self._apply_implementation_settings(self.implementation_presenter.saved_settings())
        self._apply_rooms_settings(self.rooms_presenter.saved_settings())
        self.calculate(save=False)

    def load_from_selection(self) -> None:
        values = self.presenter.load_selection(self.read_input())
        self._apply_input(values)
        self.calculate()
        if self.presenter.source_status != "ok":
            self.status_label.setText("Нет Hybrid-выбора")

    def read_input(self) -> PlanningInput:
        payload = {}
        for key, line in self.inputs.items():
            text = line.text().replace(" ", "").replace(",", ".")
            if text:
                payload[key] = text
        return PlanningInput.from_mapping(payload)

    def _apply_input(self, values: PlanningInput) -> None:
        for key, value in asdict(values).items():
            if key in self.inputs:
                self.inputs[key].setText(f"{value:g}" if isinstance(value, float) else str(value))

    def calculate(self, save: bool = True) -> None:
        values = self.read_input()
        if save:
            self.presenter.save_input(values)
        self._sync_tco(values)
        self._sync_amortization(values)
        self._sync_replacement(values)
        self._sync_financing(values, save)
        self._sync_licenses(values, save)
        self._sync_implementation(values, save)
        self._sync_rooms(save)
        self._sync_extra_tabs(values)
        self.status_label.setText("Рассчитано" if values.one_time or values.opex_annual or values.electricity_annual else "Нет данных")

    def _sync_extra_tabs(self, values: PlanningInput) -> None:
        """Hook for tabs added later (financing, licenses)."""

    def _sync_tco(self, values: PlanningInput) -> None:
        result = self.presenter.tco(values)
        self.last_tco = result
        self.tco_model.replace_rows(self.presenter.tco_rows(result))
        self.tco_note.setText(self.presenter.tco_summary(result))
        xs = list(range(len(result.years) + 1))
        self.tco_chart.show_lines(
            xs,
            [("Накопленно", result.cumulative_line), ("Приведённо", result.cumulative_discounted_line)],
            "Накопленная стоимость владения",
            x_label="Год",
        )
        self.tco_bars.show_stacked_bars(
            [str(year.year) for year in result.years],
            [("OPEX", [year.opex for year in result.years]), ("Энергия", [year.electricity for year in result.years])],
            "Затраты по годам с ростом цен",
        )

    def _sync_amortization(self, values: PlanningInput) -> None:
        rows = self.presenter.amortization(values)
        self.amortization_model.replace_rows(self.presenter.amortization_rows(rows))
        self.amortization_note.setText(self.presenter.amortization_text(rows))

    def _sync_replacement(self, values: PlanningInput) -> None:
        plan = self.presenter.replacement(values)
        self.last_replacement = plan
        self.replacement_model.replace_rows(self.presenter.replacement_rows(plan))
        self.replacement_table.resizeRowsToContents()
        self.replacement_note.setText(self.presenter.replacement_text(plan))
        if plan.peak_year is None:
            self.replacement_chart.show_empty("Замен нет")
        else:
            self.replacement_chart.show_stacked_bars(
                [str(year.year) for year in plan.years],
                [("Замена", [year.total for year in plan.years])],
                "Бюджет замены по годам",
            )
