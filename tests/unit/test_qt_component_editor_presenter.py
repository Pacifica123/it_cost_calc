from pathlib import Path

import pytest

from ui_qt.models import SolutionComponentTableModel
from ui_qt.presenters import ComponentEditorPresenter, QtAppPresenter
from ui_qt.presenters.component_editor_presenter import build_solution_component_payload


class DummyIndex:
    def __init__(self, row: int, column: int) -> None:
        self._row = row
        self._column = column

    def isValid(self) -> bool:
        return True

    def row(self) -> int:
        return self._row

    def column(self) -> int:
        return self._column


def _presenter(tmp_path: Path) -> ComponentEditorPresenter:
    app = QtAppPresenter(
        repo_root=Path(__file__).resolve().parents[2],
        runtime_entities_path=tmp_path / "runtime_entities.json",
    )
    return ComponentEditorPresenter(app)


def _server_values(name: str = "Сервер") -> dict[str, object]:
    return {
        "id": "",
        "name": name,
        "scope": "technical",
        "component_type": "server",
        "quantity": "2",
        "purchase_cost": "120000",
        "implementation_cost": "10000",
        "migration_cost": "0",
        "testing_cost": "0",
        "monthly_cost": "1500",
        "annual_cost": "0",
        "energy_cost": "900",
        "description": "узел",
        "strict_analysis_participation": True,
        "metrics": {"max_power": "450", "client_seats": "0"},
        "cost_assumptions": ["цена поставщика"],
    }


def test_component_editor_builds_payload_with_generated_id():
    payload = build_solution_component_payload(_server_values())

    assert payload["id"].startswith("component-")
    assert payload["name"] == "Сервер"
    assert payload["quantity"] == 2.0
    assert payload["metrics"]["max_power"] == 450.0
    assert payload["metadata"]["description"] == "узел"


def test_component_editor_rejects_empty_name_on_save(tmp_path: Path):
    presenter = _presenter(tmp_path)
    values = _server_values("")

    with pytest.raises(ValueError, match="Название"):
        presenter.save(values)


def test_component_editor_creates_edits_and_deletes_component(tmp_path: Path):
    presenter = _presenter(tmp_path)

    created = presenter.save(_server_values("Сервер A"))
    edited_values = presenter.form_values_for_index(created.index)
    edited_values["name"] = "Сервер B"
    updated = presenter.save(edited_values, selected_index=created.index)
    table_rows = presenter.list_table_rows()
    deleted = presenter.delete(updated.index)

    assert created.index == 0
    assert updated.component.name == "Сервер B"
    assert table_rows[0]["name"] == "Сервер B"
    assert table_rows[0]["scope"] == "ТО"
    assert deleted.name == "Сервер B"
    assert presenter.list_rows() == []


def test_solution_component_table_model_uses_short_headers():
    model = SolutionComponentTableModel(
        [{"name": "srv", "scope": "ТО", "component_type": "Сервер"}]
    )

    assert model.columnCount() == 5
    assert model.data(DummyIndex(0, 0)) == "srv"
    assert model.headerData(0, 1) == "Компонент"
