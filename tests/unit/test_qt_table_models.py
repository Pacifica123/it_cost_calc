from ui_qt.models import CandidateTableModel, DecisionTableModel, EntityTableModel


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


def test_entity_table_model_exposes_plain_rows_without_qt_runtime():
    model = EntityTableModel([{"name": "srv", "quantity": 2, "price": 150.0}])

    assert model.rowCount() == 1
    assert model.columnCount() == 5
    assert model.data(DummyIndex(0, 0)) == "srv"
    assert model.data(DummyIndex(0, 1)) == "2"
    assert model.row_dict(0)["price"] == 150.0


def test_candidate_and_decision_models_keep_default_columns():
    candidate = CandidateTableModel([{"rank": 1, "name": "GA-1", "score": 0.91}])
    decision = DecisionTableModel([{"rank": 1, "name": "A", "hybrid_score": 0.8}])

    assert candidate.columns[:3] == ("rank", "name", "score")
    assert decision.columns[-1] == "hybrid_score"


class DummyEnum:
    def __init__(self, value: int) -> None:
        self.value = value


def test_row_table_model_accepts_qt_enum_like_roles_and_orientation():
    model = EntityTableModel([{"name": "srv", "quantity": 1, "price": 10}])

    assert model.headerData(0, DummyEnum(1), DummyEnum(0)) == "Название"
    assert model.data(DummyIndex(0, 0), DummyEnum(0)) == "srv"


def test_row_table_model_exposes_cell_text_as_tooltip():
    model = EntityTableModel([{"name": "очень длинное значение для подсказки", "quantity": 1, "price": 10}])

    assert model.data(DummyIndex(0, 0), DummyEnum(3)) == "очень длинное значение для подсказки"
    assert model.data(DummyIndex(0, 0), 9999) is None
