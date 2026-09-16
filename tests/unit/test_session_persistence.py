from __future__ import annotations

from pathlib import Path

from shared.constants import ANALYSIS_SCOPE_TECHNICAL
from ui_qt.presenters import EnergyPresenter, EnergyProfileInput, NpvInput, NpvPresenter, QtAppPresenter


def _app(tmp_path: Path) -> QtAppPresenter:
    fixtures = tmp_path / "data" / "fixtures"
    generated = tmp_path / "data" / "generated"
    fixtures.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    demo = fixtures / "demo_dataset.json"
    if not demo.exists():
        demo.write_text("{}", encoding="utf-8")
    return QtAppPresenter(
        repo_root=tmp_path,
        runtime_entities_path=generated / "runtime_entities.json",
        demo_dataset_path=demo,
    )


def _persist_pool(app: QtAppPresenter) -> None:
    app.scoped_candidate_pool_service.replace(
        ANALYSIS_SCOPE_TECHNICAL,
        [
            {
                "id": "GA-1",
                "name": "Candidate A",
                "scope": ANALYSIS_SCOPE_TECHNICAL,
                "components": [
                    {
                        "name": "srv",
                        "category": "server",
                        "source_category": "server",
                        "scope": ANALYSIS_SCOPE_TECHNICAL,
                        "component_type": "server",
                        "quantity": 1,
                        "price": 1000,
                        "max_power": 100,
                    }
                ],
                "totals": {"capital_cost": 1000, "ram_gb": 16, "client_seats": 1},
                "metrics": {
                    "ga_score": 1.0,
                    "raw_scores_by_criterion": {
                        "total_ram_gb": 16,
                        "total_power_watts": 100,
                    },
                },
                "metadata": {"rank": 1},
                "source": "ga",
            }
        ],
        source_label="test GA",
        source_method="ga",
    )
    assert app.get_candidate_pool_snapshot(ANALYSIS_SCOPE_TECHNICAL).candidates


def test_session_roundtrip_reconstructs_partial_pipeline_without_recompute(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.replace_entities({"server": [{"name": "source", "quantity": 1, "price": 1000}]})
    _persist_pool(app)
    ahp = app.run_ahp_for_candidate_pool(ANALYSIS_SCOPE_TECHNICAL)
    electricity = app.calculate_electricity_costs(
        hours_per_day=8,
        working_days=20,
        cost_per_kwh=5,
        equipment_rows=[{"name": "srv", "quantity": 1, "max_power": 100}],
    )
    npv = app.build_npv_report(investment=1000, discount_rate=0.1, cash_flows=[500, 500, 500])
    exported = app.export_selected_fragment("cost_summary")
    session_id = app.active_session().session_id

    restarted = _app(tmp_path)

    assert restarted.active_session().session_id == session_id
    assert restarted.table_rows("server")[0]["name"] == "source"
    assert restarted.get_candidate_pool_snapshot(ANALYSIS_SCOPE_TECHNICAL).candidates[0]["id"] == "GA-1"
    assert restarted.get_ahp_result(ANALYSIS_SCOPE_TECHNICAL) == ahp
    assert restarted.get_electricity_report() == electricity
    assert restarted.get_npv_report() == npv
    assert len(restarted.list_export_artifacts()) == len(exported)


def test_session_fork_is_deep_and_future_mutations_are_isolated(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.replace_entities({"server": [{"name": "S1", "quantity": 1, "price": 10}]})
    app.export_selected_fragment("cost_summary")
    s1 = app.active_session()

    s2 = app.duplicate_session(s1.session_id, name="Fork")
    assert s2.session_id != s1.session_id
    assert len(app.list_export_artifacts()) == 1
    app.update_entity("server", 0, {"name": "S2", "quantity": 2, "price": 20})
    app.export_selected_fragment("npv")
    assert app.table_rows("server")[0]["name"] == "S2"
    assert len(app.list_export_artifacts()) == 2

    app.switch_session(s1.session_id)
    assert app.table_rows("server")[0]["name"] == "S1"
    assert len(app.list_export_artifacts()) == 1


def test_global_saved_data_and_sessions_are_explicitly_decoupled(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.replace_entities({"server": [{"name": "global-A", "quantity": 1, "price": 10}]})
    app.save_active_entities_as_global()
    s1 = app.active_session().session_id

    s2 = app.create_session("From global", context_type="эксперимент")
    assert app.table_rows("server")[0]["name"] == "global-A"
    app.update_entity("server", 0, {"name": "session-only", "quantity": 1, "price": 10})
    assert app.session_store.global_entities()["server"][0]["name"] == "global-A"

    app.session_store.replace_global_entities(
        {"server": [{"name": "global-B", "quantity": 1, "price": 20}]}
    )
    assert app.table_rows("server")[0]["name"] == "session-only"
    app.switch_session(s1)
    assert app.table_rows("server")[0]["name"] == "global-A"
    app.switch_session(s2.session_id)
    assert app.table_rows("server")[0]["name"] == "session-only"

    app.create_session("New global snapshot")
    assert app.table_rows("server")[0]["name"] == "global-B"


def test_input_mutation_preserves_old_results_but_marks_them_stale(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.replace_entities({"server": [{"name": "before", "quantity": 1, "price": 10}]})
    _persist_pool(app)
    old_ahp = app.run_ahp_for_candidate_pool(ANALYSIS_SCOPE_TECHNICAL)

    app.update_entity("server", 0, {"name": "after", "quantity": 1, "price": 11})

    assert app.get_candidate_pool_snapshot(ANALYSIS_SCOPE_TECHNICAL).candidates == []
    assert app.get_ahp_result(ANALYSIS_SCOPE_TECHNICAL) == {}
    state = app.session_store.session_state()
    assert state["ahp_results"][ANALYSIS_SCOPE_TECHNICAL] == old_ahp
    assert state["derived_meta"]["ahp:technical"]["validity"] == "stale"
    assert state["derived_meta"]["pool:technical"]["validity"] == "stale"


def test_export_payload_survives_restart_and_can_be_rematerialized(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.replace_entities({"server": [{"name": "srv", "quantity": 1, "price": 10}]})
    path = app.export_selected_fragment("cost_summary")["markdown"]
    original = path.read_bytes()
    artifact = app.list_export_artifacts()[0]
    path.unlink()

    restarted = _app(tmp_path)
    restored = restarted.rematerialize_export(artifact.artifact_id, tmp_path / "restored")

    assert restored.is_file()
    assert restored.read_bytes() == original
    assert restarted.list_export_artifacts()[0].source_hash == artifact.source_hash


def test_legacy_runtime_json_migrates_once_without_becoming_writable_truth(tmp_path: Path) -> None:
    generated = tmp_path / "data" / "generated"
    fixtures = tmp_path / "data" / "fixtures"
    generated.mkdir(parents=True)
    fixtures.mkdir(parents=True)
    runtime = generated / "runtime_entities.json"
    runtime.write_text(
        '{"entities":{"server":[{"name":"legacy","quantity":1,"price":10}]}}',
        encoding="utf-8",
    )
    demo = fixtures / "demo_dataset.json"
    demo.write_text("{}", encoding="utf-8")

    app = QtAppPresenter(repo_root=tmp_path, runtime_entities_path=runtime, demo_dataset_path=demo)
    session_count = len(app.list_sessions())
    assert app.table_rows("server")[0]["name"] == "legacy"
    assert app.session_store.global_entities()["server"][0]["name"] == "legacy"
    assert app.session_store.legacy_import_count() == 1

    runtime.write_text(
        '{"entities":{"server":[{"name":"changed-outside","quantity":1,"price":20}]}}',
        encoding="utf-8",
    )
    restarted = QtAppPresenter(repo_root=tmp_path, runtime_entities_path=runtime, demo_dataset_path=demo)

    assert restarted.session_store.legacy_import_count() == 1
    assert len(restarted.list_sessions()) == session_count
    assert restarted.table_rows("server")[0]["name"] == "legacy"


def test_specialized_presenters_restore_energy_and_npv_state(tmp_path: Path) -> None:
    app = _app(tmp_path)
    energy = EnergyPresenter(app)
    energy_report = energy.calculate(
        EnergyProfileInput(hours_per_day=12, working_days=25, cost_per_kwh=6.5)
    )

    npv = NpvPresenter(app)
    prepared = npv.prepare_from_costs(horizon_years=3, annual_effect=250, discount_rate=0.12)
    npv_report = npv.calculate(prepared)

    restarted = _app(tmp_path)
    restored_energy = EnergyPresenter(restarted)
    restored_npv = NpvPresenter(restarted)

    assert restored_energy.profile == EnergyProfileInput(12.0, 25.0, 6.5)
    assert restored_energy.report() == energy_report
    assert restored_npv.default_input() == prepared
    assert restored_npv.saved_report() == npv_report
    assert restored_npv.has_saved_basis()
