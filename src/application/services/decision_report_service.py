"""Application service for assembling the unified DecisionReport.

The service is deliberately tolerant to partial inputs.  A user can export a
report after only entering costs, after running GA, or after completing the full
GA + AHP + NPV chain.  Missing sections are reported as warnings instead of
breaking the export flow.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence
from uuid import uuid4

from application.services.candidate_configuration_service import CandidateConfigurationService
from application.services.solution_component_normalization_service import (
    SolutionComponentNormalizationService,
)
from domain import DecisionReport, to_plain_data
from shared.constants import LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX, SOLUTION_COMPONENT_ENTITY


class DecisionReportService:
    """Builds a single explainable decision artifact from runtime and analyses."""

    REPORT_TYPE = "it_solution_decision_report"
    SCHEMA_VERSION = 1

    def __init__(
        self,
        candidate_configuration_service: CandidateConfigurationService | None = None,
        solution_component_service: SolutionComponentNormalizationService | None = None,
    ):
        self.candidate_configuration_service = (
            candidate_configuration_service or CandidateConfigurationService()
        )
        self.solution_component_service = (
            solution_component_service or SolutionComponentNormalizationService()
        )

    def build_report(
        self,
        *,
        project: Mapping[str, Any] | None = None,
        entities: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        cost_totals: Mapping[str, Any] | None = None,
        candidate_configurations: Sequence[Mapping[str, Any]] | None = None,
        ahp_result: Mapping[str, Any] | None = None,
        criteria_importance_result: Mapping[str, Any] | None = None,
        genetic_result: Mapping[str, Any] | None = None,
        genetic_ahp_result: Mapping[str, Any] | None = None,
        npv_report: Mapping[str, Any] | None = None,
        assumptions: Sequence[str] | None = None,
        risks: Sequence[Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble a reproducible report payload.

        The returned dictionary is safe for JSON export and can also be passed to
        the markdown exporter.  All heavy/debug method payloads stay under
        ``analysis_results`` while the top-level fields contain the user-facing
        interpretation.
        """
        generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        project_payload = self._project_payload(project, generated_at=generated_at)
        components = self._components_from_entities(entities or {})
        cost_model = self._cost_model(cost_totals or {})
        candidates = self._candidate_pool(
            explicit=candidate_configurations,
            entities=entities,
            genetic_result=genetic_result,
            genetic_ahp_result=genetic_ahp_result,
            criteria_importance_result=criteria_importance_result,
        )
        analysis_results = self._analysis_results(
            ahp_result=ahp_result,
            criteria_importance_result=criteria_importance_result,
            genetic_result=genetic_result,
            genetic_ahp_result=genetic_ahp_result,
        )
        npv_interpretation = self._npv_interpretation(npv_report)
        constraints = self._constraints(analysis_results)
        warnings = self._warnings(
            components=components,
            candidates=candidates,
            analysis_results=analysis_results,
            npv_interpretation=npv_interpretation,
        )
        risk_payload = self._risks(risks, warnings=warnings)
        winner_explanation = self._winner_explanation(
            candidates=candidates,
            analysis_results=analysis_results,
            cost_model=cost_model,
            npv_interpretation=npv_interpretation,
        )

        report = DecisionReport(
            id=str(project_payload.get("id") or f"decision-report-{uuid4().hex[:8]}"),
            title=str(project_payload.get("title") or "Итоговый отчёт выбора ИТ-решения"),
            project=project_payload,
            components=components,
            cost_model=cost_model,
            assumptions=self._assumptions(assumptions, cost_model=cost_model),
            constraints=constraints,
            candidate_configurations=candidates,
            analysis_results=analysis_results,
            npv_interpretation=npv_interpretation,
            winner_explanation=winner_explanation,
            risks=risk_payload,
            warnings=warnings,
            metadata={
                "schema_version": self.SCHEMA_VERSION,
                "report_type": self.REPORT_TYPE,
                "generated_at": generated_at,
                **{str(key): deepcopy(value) for key, value in dict(metadata or {}).items()},
            },
        )
        return report.to_dict()

    def _project_payload(self, project: Mapping[str, Any] | None, *, generated_at: str) -> dict[str, Any]:
        payload = {str(key): deepcopy(value) for key, value in dict(project or {}).items()}
        payload.setdefault("id", f"decision-report-{generated_at[:10]}")
        payload.setdefault("title", "Итоговый отчёт выбора ИТ-решения")
        payload.setdefault("goal", "Свести компоненты, стоимость, альтернативы и методы анализа в единое объяснение выбора.")
        payload.setdefault("generated_at", generated_at)
        return payload

    def _components_from_entities(
        self,
        entities: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> list[dict[str, Any]]:
        components: list[dict[str, Any]] = []
        for category, rows in dict(entities).items():
            category_name = str(category)
            if category_name == SOLUTION_COMPONENT_ENTITY:
                for index, row in enumerate(rows):
                    components.append(self._solution_component_snapshot(row, index=index))
                continue

            is_legacy_sandbox = category_name.startswith(LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX)
            for index, row in enumerate(rows):
                component = {str(key): deepcopy(value) for key, value in dict(row).items()}
                component.setdefault("id", f"{category_name}:{index}")
                component.setdefault("source_category", category_name)
                component.setdefault("name", component.get("Наименование") or component.get("name") or component["id"])
                if is_legacy_sandbox:
                    component.setdefault("role", "legacy_infrastructure_sandbox")
                    component.setdefault("strict_analysis_participation", False)
                components.append(component)
        return components

    def _solution_component_snapshot(
        self,
        row: Mapping[str, Any],
        *,
        index: int,
    ) -> dict[str, Any]:
        try:
            snapshot = self.solution_component_service.to_decision_report_component_snapshot(row)
        except Exception as error:  # defensive: report export should not fail on one draft
            snapshot = {
                "id": str(row.get("id") or f"{SOLUTION_COMPONENT_ENTITY}:{index}"),
                "name": str(row.get("name") or row.get("title") or "Компонент редактора"),
                "source_format": "solution_component",
                "normalization_state": "blocked",
                "editor_status": "draft",
                "analysis_ready": False,
                "candidate_eligible": False,
                "validation_warnings": [],
                "blocking_errors": [f"SolutionComponent export failed: {error}"],
            }
        snapshot.setdefault("id", f"{SOLUTION_COMPONENT_ENTITY}:{index}")
        snapshot.setdefault("name", snapshot["id"])
        snapshot["source_category"] = SOLUTION_COMPONENT_ENTITY
        snapshot["role"] = "solution_component_editor"
        snapshot["schema_version"] = row.get("schema_version")
        snapshot.setdefault("strict_analysis_participation", bool(row.get("strict_analysis_participation", False)))
        return snapshot

    def _cost_model(self, totals: Mapping[str, Any]) -> dict[str, Any]:
        payload = {str(key): deepcopy(value) for key, value in dict(totals).items()}
        if "tco" not in payload:
            try:
                payload["tco"] = self.candidate_configuration_service.tco_model_service.from_cost_totals(
                    payload
                )
            except Exception as error:  # defensive: report export must not break the UI
                payload["tco_warning"] = f"TCO summary was not derived: {error}"
        return payload

    def _candidate_pool(
        self,
        *,
        explicit: Sequence[Mapping[str, Any]] | None,
        entities: Mapping[str, Sequence[Mapping[str, Any]]] | None,
        genetic_result: Mapping[str, Any] | None,
        genetic_ahp_result: Mapping[str, Any] | None,
        criteria_importance_result: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        pools: list[Sequence[Mapping[str, Any]] | None] = [explicit]
        if genetic_ahp_result:
            pools.append(self._nested_list(genetic_ahp_result, "export_payload", "candidate_configurations"))
            pools.append(self._nested_list(genetic_ahp_result, "ahp_report", "candidate_configurations"))
        if genetic_result:
            pools.append(self._nested_list(genetic_result, "candidate_configurations"))
        if criteria_importance_result:
            pools.append(self._nested_list(criteria_importance_result, "candidate_configurations"))

        by_id: dict[str, dict[str, Any]] = {}
        for pool in pools:
            for candidate in pool or []:
                if not isinstance(candidate, Mapping):
                    continue
                normalized = to_plain_data(candidate)
                candidate_id = str(normalized.get("id") or normalized.get("name") or len(by_id) + 1)
                normalized.setdefault("id", candidate_id)
                by_id[candidate_id] = normalized

        if not by_id and entities:
            try:
                runtime_candidate = self.candidate_configuration_service.from_runtime_entities(
                    entities,
                    candidate_id="runtime_current",
                    name="Текущий набор компонентов",
                )
                by_id[runtime_candidate.id] = runtime_candidate.to_dict()
            except Exception as error:
                by_id["runtime_current"] = {
                    "id": "runtime_current",
                    "name": "Текущий набор компонентов",
                    "metadata": {"warning": f"Runtime candidate was not derived: {error}"},
                }

        return list(by_id.values())

    def _analysis_results(
        self,
        *,
        ahp_result: Mapping[str, Any] | None,
        criteria_importance_result: Mapping[str, Any] | None,
        genetic_result: Mapping[str, Any] | None,
        genetic_ahp_result: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        if ahp_result:
            results["ahp"] = deepcopy(dict(ahp_result))
        if criteria_importance_result:
            results["criteria_importance"] = deepcopy(dict(criteria_importance_result))
        if genetic_result:
            results["genetic_optimization"] = deepcopy(dict(genetic_result))
        if genetic_ahp_result:
            results["genetic_ahp"] = deepcopy(dict(genetic_ahp_result))
            ahp_report = genetic_ahp_result.get("ahp_report")
            if isinstance(ahp_report, Mapping):
                results.setdefault("ahp", deepcopy(dict(ahp_report)))
            genetic_summary = genetic_ahp_result.get("genetic_optimization")
            if isinstance(genetic_summary, Mapping):
                results.setdefault("genetic_optimization", deepcopy(dict(genetic_summary)))
            export_payload = genetic_ahp_result.get("export_payload")
            if isinstance(export_payload, Mapping) and export_payload.get("hybrid_assessment"):
                results["hybrid_assessment"] = deepcopy(export_payload.get("hybrid_assessment"))
        return results

    def _npv_interpretation(self, npv_report: Mapping[str, Any] | None) -> dict[str, Any]:
        if not npv_report:
            return {
                "status": "not_provided",
                "explanation": "NPV-сценарий ещё не рассчитан или не передан в DecisionReport.",
            }
        report = deepcopy(dict(npv_report))
        npv_value = self._number(report.get("npv"), default=0.0)
        status = "positive" if npv_value > 0 else "negative_or_zero"
        report.setdefault("status", status)
        report.setdefault(
            "interpretation",
            "Положительный NPV поддерживает финансовую целесообразность сценария; "
            "нулевой или отрицательный NPV требует пересмотра эффекта, горизонта или затрат.",
        )
        return report

    def _constraints(self, analysis_results: Mapping[str, Any]) -> list[dict[str, Any]]:
        constraints: list[dict[str, Any]] = []
        ga = analysis_results.get("genetic_optimization")
        if isinstance(ga, Mapping):
            for item in ga.get("constraints", []):
                if isinstance(item, Mapping):
                    constraints.append({str(key): deepcopy(value) for key, value in item.items()})
        ahp = analysis_results.get("ahp")
        if isinstance(ahp, Mapping):
            criteria = ahp.get("criteria")
            if isinstance(criteria, Mapping):
                constraints.append(
                    {
                        "name": "ahp_criteria_weights",
                        "type": "methodological",
                        "value": deepcopy(criteria.get("weights", [])),
                    }
                )
        return constraints

    def _assumptions(self, assumptions: Sequence[str] | None, *, cost_model: Mapping[str, Any]) -> list[str]:
        result = [str(item) for item in assumptions or []]
        if cost_model.get("tco"):
            result.append("TCO рассчитан на основе текущих CAPEX/OPEX/электроэнергии и принятого горизонта.")
        result.append("DecisionReport фиксирует состояние данных на момент экспорта и не запускает методы заново.")
        return list(dict.fromkeys(result))

    def _warnings(
        self,
        *,
        components: Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        analysis_results: Mapping[str, Any],
        npv_interpretation: Mapping[str, Any],
    ) -> list[str]:
        warnings: list[str] = []
        if not candidates:
            warnings.append("Пул альтернатив пуст: отчёт не может полноценно объяснить выбор победителя.")
        if not analysis_results:
            warnings.append("Нет результатов аналитических методов: отчёт содержит только исходные данные и стоимость.")
        if npv_interpretation.get("status") == "not_provided":
            warnings.append("NPV-интерпретация отсутствует: финансовый вывод неполный.")
        for component in components:
            if component.get("source_format") != "solution_component":
                continue
            if not component.get("candidate_eligible", False):
                name = component.get("name") or component.get("id") or "компонент"
                state = component.get("normalization_state") or "draft"
                warnings.append(
                    f"Компонент редактора '{name}' сохранён со статусом {state} "
                    "и не участвует в строгой аналитике."
                )
            for warning in component.get("blocking_errors", []) or []:
                warnings.append(str(warning))
        for warning in npv_interpretation.get("warnings", []) or []:
            warnings.append(str(warning))
        return list(dict.fromkeys(warnings))

    def _risks(
        self,
        risks: Sequence[Mapping[str, Any]] | None,
        *,
        warnings: Sequence[str],
    ) -> list[dict[str, Any]]:
        result = [
            {str(key): deepcopy(value) for key, value in dict(item).items()}
            for item in risks or []
            if isinstance(item, Mapping)
        ]
        for index, warning in enumerate(warnings, start=1):
            result.append(
                {
                    "id": f"warning-{index}",
                    "level": "medium",
                    "description": warning,
                    "mitigation": "Уточнить исходные данные или запустить недостающий аналитический метод перед защитой вывода.",
                }
            )
        return result

    def _winner_explanation(
        self,
        *,
        candidates: Sequence[Mapping[str, Any]],
        analysis_results: Mapping[str, Any],
        cost_model: Mapping[str, Any],
        npv_interpretation: Mapping[str, Any],
    ) -> dict[str, Any]:
        leaders = {
            "ahp": self._ahp_leader(analysis_results.get("ahp")),
            "ga": self._ga_leader(analysis_results.get("genetic_optimization")),
            "hybrid": self._hybrid_leader(analysis_results.get("hybrid_assessment")),
            "criteria_importance": self._criteria_importance_leader(
                analysis_results.get("criteria_importance")
            ),
        }
        leaders = {method: leader for method, leader in leaders.items() if leader}
        leader_names = {self._leader_key(leader) for leader in leaders.values()}
        agreement = "matched" if len(leader_names) == 1 and leaders else "diverged"
        recommended = self._recommended_leader(leaders, candidates)
        reasons = self._winner_reasons(
            recommended,
            leaders=leaders,
            agreement=agreement,
            cost_model=cost_model,
            npv_interpretation=npv_interpretation,
        )
        return {
            "status": "ok" if recommended else "not_enough_data",
            "agreement": agreement if leaders else "not_enough_data",
            "leaders": leaders,
            "recommended": recommended,
            "reasons": reasons,
            "explanation": self._winner_text(recommended, leaders=leaders, agreement=agreement),
        }

    def _ahp_leader(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        final = value.get("final") if isinstance(value.get("final"), Mapping) else {}
        winner = final.get("winner") if isinstance(final.get("winner"), Mapping) else None
        if winner:
            return self._leader("AHP", winner)
        ranking = final.get("ranking") if isinstance(final.get("ranking"), list) else []
        return self._leader("AHP", ranking[0]) if ranking and isinstance(ranking[0], Mapping) else None

    def _ga_leader(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        solutions = value.get("candidate_solutions") if isinstance(value.get("candidate_solutions"), list) else []
        if solutions and isinstance(solutions[0], Mapping):
            return self._leader("GA", solutions[0])
        if value.get("selected_items"):
            return {
                "method": "GA",
                "id": "ga_best",
                "name": "Лучшее решение GA",
                "score": self._nested_get(value, "ga_result", "best_agg"),
            }
        return None

    def _hybrid_leader(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        winner = value.get("winner") if isinstance(value.get("winner"), Mapping) else None
        if winner:
            return self._leader("GA + AHP", winner)
        ranking = value.get("ranking") if isinstance(value.get("ranking"), list) else []
        return self._leader("GA + AHP", ranking[0]) if ranking and isinstance(ranking[0], Mapping) else None

    def _criteria_importance_leader(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        ranking = value.get("ranking") if isinstance(value.get("ranking"), list) else []
        if ranking and isinstance(ranking[0], Mapping):
            return self._leader("Критериальный анализ", ranking[0])
        final = value.get("final_nondominated") if isinstance(value.get("final_nondominated"), list) else []
        if final:
            return {"method": "Критериальный анализ", "id": str(final[0]), "name": str(final[0])}
        return None

    def _leader(self, method: str, item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "method": method,
            "id": str(item.get("id") or item.get("candidate_id") or item.get("rank") or method),
            "name": str(item.get("name") or item.get("id") or method),
            "rank": item.get("rank"),
            "score": item.get("score", item.get("hybrid_score", item.get("ga_score"))),
            "totals": deepcopy(item.get("totals", {})) if isinstance(item.get("totals"), Mapping) else {},
        }

    def _recommended_leader(
        self,
        leaders: Mapping[str, Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        for key in ("hybrid", "ahp", "ga", "criteria_importance"):
            if key in leaders:
                return deepcopy(dict(leaders[key]))
        return self._leader("CandidateConfiguration", candidates[0]) if candidates else None

    def _winner_reasons(
        self,
        recommended: Mapping[str, Any] | None,
        *,
        leaders: Mapping[str, Mapping[str, Any]],
        agreement: str,
        cost_model: Mapping[str, Any],
        npv_interpretation: Mapping[str, Any],
    ) -> list[str]:
        if not recommended:
            return ["Недостаточно данных для выбора победителя."]
        reasons = []
        if agreement == "matched" and leaders:
            reasons.append("Основные методы указывают на одного лидера, поэтому вывод считается согласованным.")
        elif leaders:
            reasons.append("Лидеры методов различаются, поэтому рекомендуемый вариант выбран по приоритету гибридной/AHP-оценки.")
        if cost_model.get("tco"):
            reasons.append("Стоимость владения включена в отчёт как отдельный финансовый блок.")
        if npv_interpretation.get("status") == "positive":
            reasons.append("NPV положительный при заданных денежных потоках и ставке дисконтирования.")
        elif npv_interpretation.get("status") == "negative_or_zero":
            reasons.append("NPV не подтверждает финансовую привлекательность без дополнительных эффектов или экономии.")
        return reasons

    def _winner_text(
        self,
        recommended: Mapping[str, Any] | None,
        *,
        leaders: Mapping[str, Mapping[str, Any]],
        agreement: str,
    ) -> str:
        if not recommended:
            return "Победитель не определён: нет достаточного набора альтернатив и результатов методов."
        name = recommended.get("name") or recommended.get("id")
        if agreement == "matched" and leaders:
            return f"Рекомендуемая альтернатива — {name}: методы дают согласованный лидер."
        return (
            f"Рекомендуемая альтернатива — {name}. При расхождении методов отчёт показывает лидеров "
            "по каждому методу и причины расхождения, а не скрывает неопределённость."
        )

    def _nested_list(self, value: Mapping[str, Any], *path: str) -> list[Mapping[str, Any]]:
        current: Any = value
        for key in path:
            if not isinstance(current, Mapping):
                return []
            current = current.get(key)
        return [item for item in current if isinstance(item, Mapping)] if isinstance(current, list) else []

    def _nested_get(self, value: Mapping[str, Any], *path: str) -> Any:
        current: Any = value
        for key in path:
            if not isinstance(current, Mapping):
                return None
            current = current.get(key)
        return current

    def _leader_key(self, leader: Mapping[str, Any]) -> str:
        return str(leader.get("id") or leader.get("name") or "").strip().lower()

    def _number(self, value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)


__all__ = ["DecisionReportService"]
