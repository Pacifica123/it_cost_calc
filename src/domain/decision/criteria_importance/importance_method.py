"""Устаревший совместимый импорт. Используйте ``analysis``."""

from .analysis import (
    case_to_json,
    dominance_relation,
    load_default_budgeting_case,
    run_importance_pipeline,
)

__all__ = [
    "load_default_budgeting_case",
    "run_importance_pipeline",
    "dominance_relation",
    "case_to_json",
]
