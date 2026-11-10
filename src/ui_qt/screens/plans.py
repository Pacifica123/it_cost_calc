from __future__ import annotations

from dataclasses import asdict

from ui_qt.models import RowTableModel
from ui_qt.presenters import QtAppPresenter
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
        QTableView,
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
    """«Планы»: TCO по годам, амортизация, план замены, финансирование и календарь лицензий."""

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
        for line in [*self.inputs.values(), *self.financing_inputs.values(), self.license_start_input, self.license_remind_input]:
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
