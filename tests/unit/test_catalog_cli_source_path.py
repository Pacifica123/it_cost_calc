from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_feed_download_prefers_current_src_tree_over_stale_installed_service(tmp_path: Path):
    source = tmp_path / "catalog.csv"
    source.write_text(
        "name;category;price\nRouter;router;1000\n",
        encoding="utf-8",
    )
    output = tmp_path / "downloaded.csv"
    staging = tmp_path / "staging.json"

    stale_root = tmp_path / "stale_site"
    service_module = stale_root / "application" / "services" / "catalog_staging_service.py"
    service_module.parent.mkdir(parents=True)
    (stale_root / "application" / "__init__.py").write_text("", encoding="utf-8")
    (stale_root / "application" / "services" / "__init__.py").write_text("", encoding="utf-8")
    service_module.write_text(
        "class CatalogStagingService:\n"
        "    def __init__(self, path): self.path = path\n"
        "    def stage_file(self, source_path): return []\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(stale_root)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "update_equipment_catalog.py"),
            "--mode",
            "feed-download",
            "--input",
            str(source),
            "--output",
            str(output),
            "--feed-format",
            "csv",
            "--feed-source-id",
            "regression-source",
            "--staging-path",
            str(staging),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Staging обновлён: 1 позиций." in completed.stdout
    assert output.is_file()
    assert staging.is_file()
