from pathlib import Path

from ui_qt.presenters import QtAppPresenter


def test_qt_presenter_uses_sqlite_session_repository_without_treeview(tmp_path: Path):
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

    assert presenter.paths.session_db_path.exists()
    assert not runtime_path.exists(), "legacy JSON must not become a second writable source of truth"
    assert created["name"] == "srv"
    assert updated["quantity"] == 2
    assert rows == [{"name": "srv-2", "quantity": 2, "price": 150.0}]
    assert "Treeview" not in type(presenter.repository).__name__


def test_qt_presenter_loads_demo_dataset_to_sqlite_session_storage(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    runtime_path = tmp_path / "runtime_entities.json"
    presenter = QtAppPresenter(repo_root=repo_root, runtime_entities_path=runtime_path)

    dataset = presenter.load_demo_dataset()
    persisted = presenter.session_store.session_state()["entities"]

    assert presenter.has_runtime_data()
    assert dataset["server"][0]["name"] == persisted["server"][0]["name"]
    assert presenter.total_rows() == sum(len(rows) for rows in dataset.values())
    assert not runtime_path.exists()
