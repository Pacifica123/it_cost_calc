"""Smoke scenario for C8 manual conversion of legacy sandbox rows.

The script intentionally does not use pytest and does not create bytecode when
called with PYTHONDONTWRITEBYTECODE=1.  It verifies the conversion service as a
plain application contract: legacy rows stay untouched, converted components keep
origin metadata, incomplete conversions remain drafts, and explicit strict
classification can become analysis-ready.
"""

from __future__ import annotations

import json

from application.services.solution_component_sandbox_conversion_service import (
    SolutionComponentSandboxConversionService,
)
from application.services.solution_component_runtime_service import SolutionComponentRuntimeService
from infrastructure.repositories.in_memory_entity_repository import InMemoryEntityRepository
from shared.constants import LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX, SOLUTION_COMPONENT_ENTITY


def main() -> None:
    sandbox_entity = f"{LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX}printer_support"
    legacy_row = {
        "name": "Поддержка МФУ",
        "quantity": 2,
        "monthly_cost": 1800,
        "_legacy_sandbox": True,
        "_legacy_article_title": "Черновые сервисные статьи",
        "_legacy_expense_type": "periodic",
        "_legacy_note": "Свободная запись старой ИТ-песочницы.",
    }
    repository = InMemoryEntityRepository({sandbox_entity: [legacy_row]})
    conversion_service = SolutionComponentSandboxConversionService()
    runtime_service = SolutionComponentRuntimeService(
        repository,
        normalization_service=conversion_service.normalization_service,
    )

    draft_payload = conversion_service.build_conversion_draft(
        legacy_row,
        sandbox_entity=sandbox_entity,
        row_index=0,
    )
    draft = runtime_service.add_component(draft_payload)

    assert repository.list(sandbox_entity)[0] == legacy_row, "legacy sandbox row must remain unchanged"
    assert draft.origin.value == "converted_sandbox"
    assert draft.normalization_state.value == "draft"
    assert not draft.candidate_eligible
    assert draft.metadata["legacy_sandbox_origin"]["entity_key"] == sandbox_entity
    assert draft.metadata["legacy_sandbox_origin"]["original_row"]["_legacy_sandbox"] is True

    strict_component = runtime_service.add_component(
        conversion_service.build_conversion_draft(
            legacy_row,
            sandbox_entity=sandbox_entity,
            row_index=0,
            overrides={
                "id": "converted-printer-support-strict",
                "scope": "technical",
                "component_type": "support_service",
                "strict_analysis_participation": True,
                "monthly_cost": 1800,
                "metrics": {"client_seats": 0, "support_score": 4},
            },
        )
    )

    assert strict_component.origin.value == "converted_sandbox"
    assert strict_component.candidate_eligible, strict_component.blocking_errors
    assert strict_component.tco_eligible
    assert strict_component.metadata["source_category"] == "Черновые сервисные статьи"
    assert len(repository.list(SOLUTION_COMPONENT_ENTITY)) == 2
    assert len(repository.list(sandbox_entity)) == 1, "conversion must not archive legacy row implicitly"

    snapshot = runtime_service.export_snapshot()
    assert snapshot["component_count"] == 2
    assert snapshot["strict_component_count"] == 1
    assert snapshot["draft_component_count"] == 1

    print("SolutionComponent C8 sandbox conversion smoke: OK")
    print(
        json.dumps(
            {
                "legacy_entity": sandbox_entity,
                "legacy_rows_after_conversion": len(repository.list(sandbox_entity)),
                "solution_components": len(repository.list(SOLUTION_COMPONENT_ENTITY)),
                "strict_component_id": strict_component.id,
                "draft_state": draft.normalization_state.value,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
