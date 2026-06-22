from __future__ import annotations

from pathlib import Path

from shared.constants import ANALYSIS_SCOPE_TECHNICAL
from ui_qt.presenters import HybridPresenter, QtAppPresenter


def _presenter(tmp_path: Path) -> QtAppPresenter:
    fixtures = tmp_path / "data" / "fixtures"
    generated = tmp_path / "data" / "generated"
    fixtures.mkdir(parents=True)
    generated.mkdir(parents=True)
    (fixtures / "demo_dataset.json").write_text("{}", encoding="utf-8")
    return QtAppPresenter(
        repo_root=tmp_path,
        runtime_entities_path=generated / "runtime_entities.json",
        demo_dataset_path=fixtures / "demo_dataset.json",
    )


def _candidate(candidate_id: str, *, rank: int, score: float, ram: float, power: float, cost: float):
    return {
        "id": candidate_id,
        "name": candidate_id,
        "scope": ANALYSIS_SCOPE_TECHNICAL,
        "components": [],
        "totals": {
            "capital_cost": cost,
            "ram_gb": ram,
            "client_seats": 3,
        },
        "metrics": {
            "ga_score": score,
            "raw_scores_by_criterion": {
                "total_ram_gb": ram,
                "total_power_watts": power,
            },
        },
        "source": "ga",
        "metadata": {"rank": rank},
    }


def _fill_pool(app: QtAppPresenter) -> None:
    app.scoped_candidate_pool_service.replace(
        ANALYSIS_SCOPE_TECHNICAL,
        [
            _candidate("GA-1", rank=1, score=0.9, ram=16, power=80, cost=100),
            _candidate("GA-2", rank=2, score=0.7, ram=32, power=140, cost=120),
        ],
        source_label="test GA",
        source_method="ga",
    )


def test_qt_hybrid_waits_for_ahp(tmp_path: Path):
    app = _presenter(tmp_path)
    _fill_pool(app)
    presenter = HybridPresenter(app, scope=ANALYSIS_SCOPE_TECHNICAL)

    assert presenter.can_run() is False
    result = presenter.run()

    assert result["status"] == "incomplete"
    assert result["missing_methods"] == ["ahp"]
    assert presenter.summary().status == "Не готово"


def test_qt_hybrid_combines_current_pool_after_ahp_and_pareto(tmp_path: Path):
    app = _presenter(tmp_path)
    _fill_pool(app)
    app.run_ahp_for_candidate_pool(ANALYSIS_SCOPE_TECHNICAL)
    app.run_pareto_for_candidate_pool(ANALYSIS_SCOPE_TECHNICAL)
    presenter = HybridPresenter(app, scope=ANALYSIS_SCOPE_TECHNICAL)

    result = presenter.run(lambda_value=0.4)

    assert result["status"] == "ok"
    assert result["lambda"] == 0.4
    assert result["metadata"]["analysis_scope"] == ANALYSIS_SCOPE_TECHNICAL
    assert len(presenter.ranking_rows()) == 2
    assert presenter.ranking_rows()[0]["candidate_id"]
    assert presenter.summary().ranked_count == 2
    assert app.get_hybrid_result(ANALYSIS_SCOPE_TECHNICAL)["status"] == "ok"


def test_qt_hybrid_export_payload_includes_scope_results(tmp_path: Path):
    app = _presenter(tmp_path)
    _fill_pool(app)
    app.run_ahp_for_candidate_pool(ANALYSIS_SCOPE_TECHNICAL)
    app.run_pareto_for_candidate_pool(ANALYSIS_SCOPE_TECHNICAL)
    app.run_hybrid_assessment(ANALYSIS_SCOPE_TECHNICAL)

    scoped = app._scoped_decision_report_inputs()

    assert scoped["candidate_configurations"]
    assert scoped["hybrid_assessment_result"]["by_scope"][ANALYSIS_SCOPE_TECHNICAL]["status"] == "ok"
    assert scoped["criteria_importance_result"]["by_scope"][ANALYSIS_SCOPE_TECHNICAL]["status"] == "ok"


def test_workspace_source_connects_hybrid_step():
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "ui_qt"
        / "screens"
        / "workspace_data.py"
    ).read_text(encoding="utf-8")

    assert "HybridScreen" in source
    assert "self.hybrid_step_page" in source
    assert 'step_id == "hybrid"' in source
