from __future__ import annotations

from pathlib import Path

from shared.constants import ANALYSIS_SCOPE_TECHNICAL
from ui_qt.presenters import AhpPresenter, ParetoPresenter, QtAppPresenter


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


def test_qt_ahp_runs_only_current_candidate_pool(tmp_path: Path):
    app = _presenter(tmp_path)
    app.scoped_candidate_pool_service.replace(
        ANALYSIS_SCOPE_TECHNICAL,
        [
            _candidate("GA-1", rank=1, score=0.9, ram=16, power=80, cost=100),
            _candidate("GA-2", rank=2, score=0.7, ram=32, power=140, cost=120),
        ],
        source_label="test GA",
        source_method="ga",
    )

    report = app.run_ahp_for_candidate_pool(ANALYSIS_SCOPE_TECHNICAL)

    assert report["status"] == "ok"
    assert [row["id"] for row in report["final"]["ranking"]] == ["GA-2", "GA-1"]
    assert {row["id"] for row in report["final"]["ranking"]} == {"GA-1", "GA-2"}
    assert app.get_ahp_result(ANALYSIS_SCOPE_TECHNICAL)["candidate_count"] == 2


def test_qt_pareto_handles_single_candidate_boundary(tmp_path: Path):
    app = _presenter(tmp_path)
    app.scoped_candidate_pool_service.replace(
        ANALYSIS_SCOPE_TECHNICAL,
        [_candidate("GA-1", rank=1, score=1.0, ram=16, power=80, cost=100)],
        source_label="single GA",
        source_method="ga",
    )

    report = app.run_pareto_for_candidate_pool(ANALYSIS_SCOPE_TECHNICAL)

    assert report["status"] == "ok"
    assert report["final_nondominated"] == ["GA-1"]
    assert report["ranking"][0]["nondominated"] is True


def test_qt_decision_presenters_return_compact_rows(tmp_path: Path):
    app = _presenter(tmp_path)
    app.scoped_candidate_pool_service.replace(
        ANALYSIS_SCOPE_TECHNICAL,
        [
            _candidate("GA-1", rank=1, score=0.9, ram=16, power=80, cost=100),
            _candidate("GA-2", rank=2, score=0.7, ram=32, power=140, cost=120),
        ],
        source_label="test GA",
        source_method="ga",
    )
    ahp = AhpPresenter(app, scope=ANALYSIS_SCOPE_TECHNICAL)
    pareto = ParetoPresenter(app, scope=ANALYSIS_SCOPE_TECHNICAL)

    ahp.run()
    pareto.run()

    assert ahp.summary().ranked_count == 2
    assert ahp.ranking_rows()[0]["name"]
    assert pareto.summary().ranked_count == 2
    assert pareto.nondominated_rows()
