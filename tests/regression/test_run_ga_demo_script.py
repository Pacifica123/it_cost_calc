from __future__ import annotations

import json

import pytest


pytest.importorskip("numpy")


def test_run_ga_demo_script_writes_result_snapshot(tmp_path, monkeypatch):
    import scripts.run_ga_demo as run_ga_demo

    output_path = tmp_path / "ga_demo_output.json"
    monkeypatch.setattr(run_ga_demo, "OUTPUT_PATH", output_path)

    result = run_ga_demo.main()

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["scenario"] == "ga_mvp_weight_performance_energy"
    assert payload["result"]["error"] is None
    assert payload["result"]["fitness_scale"] == "normalized_0_1"
    assert result["best_agg"] is not None
    assert len(result["best_items"]) >= 1
