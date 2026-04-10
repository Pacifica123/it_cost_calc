"""Compatibility wrapper for the renamed cost summary service."""

from application.services.cost_summary_service import CostSummaryService

TotalCosts = CostSummaryService
