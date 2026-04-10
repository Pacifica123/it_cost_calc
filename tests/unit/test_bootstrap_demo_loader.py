from pathlib import Path

from it_cost_calc.bootstrap import load_demo_data
from it_cost_calc.infrastructure.storage import JsonFileStorage


def test_bootstrap_load_demo_data_writes_runtime_storage(tmp_path: Path):
    fixture_path = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "demo_dataset.json"
    runtime_path = tmp_path / "runtime_entities.json"

    dataset = load_demo_data(
        repo_root=Path(__file__).resolve().parents[2],
        fixture_path=fixture_path,
        runtime_entities_path=runtime_path,
    )

    payload = JsonFileStorage().read(runtime_path)

    assert runtime_path.exists()
    assert "server" in dataset
    assert payload["entities"]["server"][0]["name"] == dataset["server"][0]["name"]
