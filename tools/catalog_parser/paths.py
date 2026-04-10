from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "data" / "examples" / "parser"
GENERATED_DIR = REPO_ROOT / "data" / "generated" / "catalog"
DEFAULT_OUTPUT_PATH = GENERATED_DIR / "equipment_catalog.json"
DEFAULT_EXAMPLE_SNAPSHOT = EXAMPLES_DIR / "dns_data_playwright.json"
DEFAULT_ROUTERS_SNAPSHOT = EXAMPLES_DIR / "dns_routers.json"
