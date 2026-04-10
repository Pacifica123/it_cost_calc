import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from it_cost_calc.domain.decision.ahp import (
        DEFAULT_CONSTRAINTS,
        DEFAULT_EXPERT_MATRICES,
        DEFAULT_SOFT_CRITERIA,
        run_ahp_pipeline,
    )
    from it_cost_calc.infrastructure.logging import configure_logging

    configure_logging(repo_root=ROOT)
    sample_path = ROOT / "data" / "examples" / "ahp" / "test_confs.json"
    with sample_path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    configurations = payload.get("configurations", payload)
    result = run_ahp_pipeline(
        configurations=configurations,
        soft_criteria=DEFAULT_SOFT_CRITERIA,
        experts_criteria_matrices=DEFAULT_EXPERT_MATRICES,
        constraints=DEFAULT_CONSTRAINTS,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
