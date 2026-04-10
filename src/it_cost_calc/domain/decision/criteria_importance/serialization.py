from __future__ import annotations

import json
from typing import Any, Dict

from it_cost_calc.domain import to_plain_data


def case_to_json(case_data: Dict[str, Any]) -> str:
    return json.dumps(to_plain_data(case_data), ensure_ascii=False, indent=2)
