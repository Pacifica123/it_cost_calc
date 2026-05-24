from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from application.services.solution_component_runtime_service import SolutionComponentRuntimeService
from infrastructure.repositories.in_memory_entity_repository import InMemoryEntityRepository
from ui.tabs.solution_component_editor_tab import (
    build_solution_component_payload,
    format_normalization_preview,
    readiness_label,
)


def main() -> int:
    repository = InMemoryEntityRepository()
    runtime_service = SolutionComponentRuntimeService(repository)

    technical_payload = build_solution_component_payload(
        {
            "name": "Сервер приложений для MVP",
            "scope": "technical",
            "component_type": "server",
            "strict_analysis_participation": True,
            "quantity": "1",
            "purchase_cost": "185000",
            "monthly_cost": "9000",
            "metrics": {
                "max_power": "420",
                "client_seats": "0",
                "performance_score": "8",
                "reliability_score": "8",
                "hours_per_day": "12",
                "working_days": "22",
            },
            "description": "Проверочный технический компонент редактора.",
            "cost_assumptions": ["Цена указана для MVP-сценария C4."],
        }
    )
    technical = runtime_service.add_component(technical_payload)

    software_draft_payload = build_solution_component_payload(
        {
            "name": "Черновик подписки Service Desk",
            "scope": "software",
            "component_type": "software_subscription",
            "strict_analysis_participation": False,
            "quantity": "10",
            "metrics": {
                "license_units": "10",
                "functionality_score": "7",
                "compatibility_score": "6",
                "support_score": "8",
            },
            "description": "Неполная программная строка, сохранённая как черновик.",
            "cost_assumptions": ["Стоимость будет уточнена после выбора тарифа."],
        }
    )
    software_draft = runtime_service.add_component(software_draft_payload)

    snapshot = runtime_service.export_snapshot()
    assert snapshot["component_count"] == 2, snapshot
    assert snapshot["strict_component_count"] == 1, snapshot
    assert snapshot["draft_component_count"] == 1, snapshot
    assert technical.analysis_ready, technical
    assert technical.energy_eligible, technical
    assert not software_draft.candidate_eligible, software_draft
    assert readiness_label(technical) == "готов к аналитике"
    assert "CandidateConfiguration: да" in format_normalization_preview(technical)
    assert "CandidateConfiguration: нет" in format_normalization_preview(software_draft)
    print("solution_component_editor_mvp_demo: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
