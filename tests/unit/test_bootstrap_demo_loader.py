from pathlib import Path

from bootstrap import load_demo_data
from infrastructure.storage import SessionStore


def test_bootstrap_load_demo_data_writes_runtime_storage(tmp_path: Path):
    fixture_path = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "demo_dataset.json"
    runtime_path = tmp_path / "runtime_entities.json"

    dataset = load_demo_data(
        repo_root=Path(__file__).resolve().parents[2],
        fixture_path=fixture_path,
        runtime_entities_path=runtime_path,
    )

    db_path = runtime_path.with_name("runtime.sqlite3")
    store = SessionStore(db_path, legacy_runtime_path=runtime_path)

    assert db_path.exists()
    assert not runtime_path.exists()
    assert "server" in dataset
    assert store.session_state()["entities"]["server"][0]["name"] == dataset["server"][0]["name"]
