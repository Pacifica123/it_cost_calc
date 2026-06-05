from pathlib import Path

from infrastructure.storage import JsonFileStorage
from ui_qt.presenters import QtAppPresenter


def test_qt_presenter_uses_json_repository_without_treeview(tmp_path: Path):
    runtime_path = tmp_path / "runtime_entities.json"
    presenter = QtAppPresenter(
        repo_root=Path(__file__).resolve().parents[2],
        runtime_entities_path=runtime_path,
    )

    created = presenter.add_entity("server", {"name": "srv", "quantity": 1, "price": 100.0})
    updated = presenter.update_entity(
        "server",
        0,
        {"name": "srv-2", "quantity": 2, "price": 150.0},
    )
    rows = presenter.table_rows("server")

    assert runtime_path.exists()
    assert created["name"] == "srv"
    assert updated["quantity"] == 2
    assert rows == [{"name": "srv-2", "quantity": 2, "price": 150.0}]
    assert "Treeview" not in type(presenter.repository).__name__


def test_qt_presenter_loads_demo_dataset_to_runtime_storage(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    runtime_path = tmp_path / "runtime_entities.json"
    presenter = QtAppPresenter(repo_root=repo_root, runtime_entities_path=runtime_path)

    dataset = presenter.load_demo_dataset()
    persisted = JsonFileStorage().read(runtime_path)

    assert presenter.has_runtime_data()
    assert dataset["server"][0]["name"] == persisted["entities"]["server"][0]["name"]
    assert presenter.total_rows() == sum(len(rows) for rows in dataset.values())
