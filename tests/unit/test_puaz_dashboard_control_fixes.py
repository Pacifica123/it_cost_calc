"""Регрессии, найденные контрольным примером ПУАЗ для интерактивного дашборда.

Каждый тест фиксирует одно искажение, которое было видно на дашборде:
пропадание альтернатив ПО, отсутствие рангов GA, маскировка неполных метрик
и двойной учёт подписки в TCO.
"""

from __future__ import annotations

from application.services.decision_report_service import DecisionReportService
from application.services.tco_model_service import TCOModelService
from infrastructure.exporters.interactive_dashboard_exporter import (
    build_interactive_dashboard_payload,
)


def _candidate(candidate_id: str, scope: str, *, cost: float) -> dict:
    return {
        "id": candidate_id,
        "name": f"GA-кандидат {candidate_id[-1]}",
        "scope": scope,
        "components": [],
        "totals": {"capital_cost": cost},
        "metadata": {"rank": int(candidate_id[-1])},
        "metrics": {"ga_score": 1.0 / int(candidate_id[-1])},
    }


def test_decision_report_keeps_same_ga_ids_from_software_and_technical_scopes():
    report = DecisionReportService().build_report(
        candidate_configurations=[
            _candidate("GA-1", "software", cost=30000.0),
            _candidate("GA-2", "software", cost=50000.0),
            _candidate("GA-1", "technical", cost=290000.0),
        ]
    )

    keys = {(row["scope"], row["id"]) for row in report["candidate_configurations"]}

    assert keys == {("software", "GA-1"), ("software", "GA-2"), ("technical", "GA-1")}


def test_dashboard_reads_ga_ranks_from_candidate_configurations():
    report = {
        "candidate_configurations": [
            _candidate("GA-1", "technical", cost=100.0),
            _candidate("GA-2", "technical", cost=90.0),
        ],
        "analysis_results": {
            "genetic_optimization": {
                "by_scope": {
                    "technical": {
                        # raw GA rows carry rank/score but no id
                        "candidate_solutions": [{"rank": 1, "score": 0.9}, {"rank": 2, "score": 0.5}],
                        "candidate_configurations": [
                            _candidate("GA-1", "technical", cost=100.0),
                            _candidate("GA-2", "technical", cost=90.0),
                        ],
                    }
                }
            }
        },
    }

    payload = build_interactive_dashboard_payload(report)
    by_id = {row["id"]: row for row in payload["candidates"]}

    assert by_id["GA-1"]["method_evidence"]["GA:technical"]["rank"] == 1
    assert by_id["GA-2"]["method_evidence"]["GA:technical"]["rank"] == 2
    assert by_id["GA-1"]["analysis_support"] == 100.0
    assert by_id["GA-2"]["analysis_support"] == 0.0


def test_missing_catalog_metric_is_not_hidden_by_zero_in_candidate_copy():
    source_row = {
        "name": "К4 Рабочее место производительное",
        "quantity": 1,
        "price": 70000.0,
        "scope": "technical",
        "component_type": "workstation",
        "origin": "catalog",
        "catalog_item_id": "ctl-ws-2",
        "ram_gb": 16,
        "cpu_cores": 8,
        "max_power_watts": 120,
        "metric_warnings": ["compute metric storage_gb is missing"],
    }
    candidate_copy = {**source_row, "storage_gb": 0.0}
    candidate = {
        "id": "GA-1",
        "scope": "technical",
        "components": [candidate_copy],
        "totals": {"capital_cost": 70000.0},
    }

    report = DecisionReportService().build_report(
        entities={"client": [source_row]},
        candidate_configurations=[candidate],
    )
    quality = report["catalog_data_quality"]

    assert quality["summary"]["incomplete_metrics"] == 1
    assert quality["components"][0]["missing_metrics"] == ["storage_gb"]


def test_explicit_zero_purchase_cost_of_subscription_is_not_counted_twice():
    subscription = {
        "name": "П3 Облачный офис (подписка)",
        "category": "subscription_licenses",
        "source_category": "subscription_licenses",
        "component_type": "software_subscription",
        "scope": "software",
        "purchase_cost": 0.0,
        # analytics cost of the first year, used by GA ranking
        "total_cost": 24000.0,
        "price": 24000.0,
        "monthly_cost": 2000.0,
        "quantity": 1,
    }

    model = TCOModelService().build_component_cost_model(subscription)

    assert model.purchase_cost == 0.0
    assert model.subscription_cost == 2000.0


def test_dashboard_capex_comes_from_tco_model_not_first_year_analytics_cost():
    candidate = {
        "id": "GA-6",
        "scope": "technical",
        "components": [],
        "totals": {
            # first-year analytics cost: purchase 250 000 + contract 36 000
            "capital_cost": 286000.0,
            "initial_investment": 250000.0,
            "tco": {
                "total_capex": 250000.0,
                "annual_opex": 36000.0,
                "total_ownership_cost": 286000.0,
            },
        },
    }

    payload = build_interactive_dashboard_payload({"candidate_configurations": [candidate]})
    row = payload["candidates"][0]

    assert row["capex"] == 250000.0
    assert row["annual_opex"] == 36000.0
    assert row["capex"] + row["annual_opex"] == row["tco"]
