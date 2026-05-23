TECHNICAL_CAPITAL_CATEGORIES = (
    "server",
    "client",
    "network",
)

SOFTWARE_CAPITAL_CATEGORIES = (
    "licenses",
)

CAPITAL_COST_CATEGORIES = TECHNICAL_CAPITAL_CATEGORIES + SOFTWARE_CAPITAL_CATEGORIES

OPERATIONAL_COST_CATEGORIES = (
    "subscription_licenses",
    "server_rental",
    "migration",
    "testing",
    "backup",
    "labor_costs",
    "server_administration",
)

ANALYSIS_SCOPE_TECHNICAL = "technical"
ANALYSIS_SCOPE_SOFTWARE = "software"
ANALYSIS_SCOPE_LABELS = {
    ANALYSIS_SCOPE_TECHNICAL: "ТО",
    ANALYSIS_SCOPE_SOFTWARE: "ПО",
}
ANALYSIS_SCOPE_TITLES = {
    ANALYSIS_SCOPE_TECHNICAL: "Техническое обеспечение",
    ANALYSIS_SCOPE_SOFTWARE: "Программное обеспечение",
}
ANALYSIS_SCOPE_CAPITAL_CATEGORIES = {
    ANALYSIS_SCOPE_TECHNICAL: TECHNICAL_CAPITAL_CATEGORIES,
    ANALYSIS_SCOPE_SOFTWARE: SOFTWARE_CAPITAL_CATEGORIES,
}
# Namespaced storage for the legacy free-form infrastructure tab.
# These rows are auxiliary sandbox data and should not be treated as strict
# CAPEX/OPEX/analysis entities unless a dedicated normalizer explicitly opts in.
LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX = "legacy_infrastructure:"
