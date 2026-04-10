import json
from pathlib import Path

from tools.catalog_parser.cli import main



def test_build_equipment_catalog_from_examples(tmp_path: Path) -> None:
    output_path = tmp_path / "equipment_catalog.json"
    exit_code = main(["--mode", "examples", "--output", str(output_path)])
    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["stats"]["items_total"] > 0
    assert "cpu" in payload["stats"]["by_category"]
