import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from it_cost_calc.domain.decision.criteria_importance.analysis import (
        load_default_budgeting_case,
        run_importance_pipeline,
    )
    from it_cost_calc.infrastructure.logging import configure_logging

    configure_logging(repo_root=ROOT)
    case = load_default_budgeting_case()
    report = run_importance_pipeline(case)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
