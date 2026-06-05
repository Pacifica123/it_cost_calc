import pytest

from ui_qt.dialogs import FieldSpec, default_entity_fields, payload_from_strings
from ui_qt.models import EntityTableModel, RowTableModel
from ui_qt.widgets import SmartTable


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


def test_field_schema_parses_payload_and_enforces_short_labels():
    fields = [
        FieldSpec("name", "Название"),
        FieldSpec("quantity", "Кол.", kind="int"),
        FieldSpec("price", "Цена", kind="float"),
    ]

    payload = payload_from_strings(
        fields,
        {"name": " Сервер ", "quantity": "2", "price": "10,5"},
    )

    assert payload == {"name": "Сервер", "quantity": 2, "price": 10.5}


def test_field_schema_rejects_long_visible_label():
    with pytest.raises(ValueError):
        FieldSpec("bad", "слишком длинная видимая подпись которая нарушает контракт интерфейса и точно должна быть отклонена")


def test_default_entity_fields_are_dialog_ready():
    fields = default_entity_fields()

    assert [field.name for field in fields] == [
        "name",
        "quantity",
        "price",
        "monthly_cost",
        "one_time_cost",
    ]
    assert all(field.label for field in fields)


def test_row_table_model_supports_crud_helpers_without_qt_runtime():
    model = RowTableModel(
        rows=[{"name": "srv", "price": 100}],
        columns=("name", "price"),
        headers={"name": "Название", "price": "Цена"},
    )

    model.append_row({"name": "pc", "price": 50})
    old = model.update_row(0, {"name": "srv-2", "price": 120})
    removed = model.remove_row(1)

    assert old == {"name": "srv", "price": 100}
    assert removed == {"name": "pc", "price": 50}
    assert model.rowCount() == 1
    assert model.column_index("price") == 1
    assert model.data(DummyIndex(0, 0)) == "srv-2"


def test_entity_table_model_can_be_refreshed_for_smart_table():
    model = EntityTableModel()

    assert model.is_empty()
    model.replace_rows([{"name": "license", "quantity": 1, "price": 20}])

    assert not model.is_empty()
    assert model.as_rows()[0]["name"] == "license"


def test_smart_table_is_lazy_imported_without_qt_runtime():
    assert SmartTable.__name__ == "SmartTable"
