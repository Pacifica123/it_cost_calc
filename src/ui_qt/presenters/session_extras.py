"""Small helpers to keep screen settings of the planning tabs in the active session.

The values live next to the other session state under their own key, so they survive
a restart and follow «Сессии» like the NPV draft does. Nothing here marks analysis
results as stale: these are inputs of independent calculations.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ui_qt.presenters.app_presenter import QtAppPresenter


def load_extra(app: QtAppPresenter, key: str) -> dict[str, Any]:
    try:
        state = app.session_store.session_state()
    except Exception:  # pragma: no cover - storage fallback
        return {}
    payload = state.get(key, {})
    return deepcopy(dict(payload)) if isinstance(payload, Mapping) else {}


def save_extra(app: QtAppPresenter, key: str, payload: Mapping[str, Any]) -> None:
    data = deepcopy(dict(payload))

    def persist(state: dict[str, Any]) -> None:
        state[key] = data

    app.session_store.mutate_active_state(persist)
