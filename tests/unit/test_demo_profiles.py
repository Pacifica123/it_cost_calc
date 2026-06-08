from pathlib import Path

from ui_qt.presenters import QtAppPresenter


def test_qt_presenter_lists_demo_profiles():
    presenter = QtAppPresenter(repo_root=Path(__file__).resolve().parents[2])

    profiles = presenter.list_demo_profiles()
    profile_ids = {profile["id"] for profile in profiles}

    assert "default" in profile_ids
    assert "calculation_control" in profile_ids
    assert "technical_load_1000" in profile_ids


def test_qt_presenter_loads_calculation_control_profile(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    runtime_path = tmp_path / "runtime_entities.json"
    presenter = QtAppPresenter(repo_root=repo_root, runtime_entities_path=runtime_path)

    dataset = presenter.load_demo_profile("calculation_control")

    assert dataset["server"][0]["name"] == "T1 Сервер"
    assert dataset["client"][0]["client_seats"] == 3
    assert dataset["licenses"][1]["name"] == "S2 Pro"
    assert presenter.total_rows() == sum(len(rows) for rows in dataset.values())


def test_qt_presenter_loads_technical_load_profile(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    runtime_path = tmp_path / "runtime_entities.json"
    presenter = QtAppPresenter(repo_root=repo_root, runtime_entities_path=runtime_path)

    dataset = presenter.load_demo_profile("technical_load_1000")

    assert len(dataset["server"]) == 334
    assert len(dataset["client"]) == 333
    assert len(dataset["network"]) == 333
    assert sum(len(dataset[name]) for name in ("server", "client", "network")) == 1000
    assert dataset["server"][0]["name"] == "LOAD-SRV-0001"
    assert dataset["client"][0]["client_seats"] >= 3
    assert presenter.total_rows() == sum(len(rows) for rows in dataset.values())
