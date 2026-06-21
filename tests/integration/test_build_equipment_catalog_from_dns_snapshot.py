import json
from pathlib import Path

from tools.catalog_parser.cli import main


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "data" / "examples" / "parser" / "dns_snapshot"


def test_build_equipment_catalog_from_dns_snapshot(tmp_path: Path) -> None:
    output = tmp_path / "catalog.json"
    assert main(["--mode", "dns-snapshot", "--input", str(SNAPSHOT), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["stats"]["items_total"] == 2
    assert payload["sources"][0]["mode"] == "offline-html-jsonld"

    items = {item["category"]: item for item in payload["items"]}
    assert items["router"]["attributes"]["lan_speed_mbps"] == 1000
    assert items["router"]["attributes"]["max_power_watts"] == 18
    assert items["prebuilt_pc"]["attributes"]["ram_gb"] == 32
    assert items["prebuilt_pc"]["attributes"]["storage_gb"] == 3072
    assert items["prebuilt_pc"]["offer"]["observed_at"] == "2026-06-21T12:00:00Z"
