from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from ui_qt.navigation.routes import DEFAULT_ROOT_ROUTE_ID, ROOT_ROUTE_BY_ID

_THEME_VALUES = {"light", "dark"}
_SCALE_VALUES = (0.9, 1.0, 1.1, 1.25)


@dataclass(frozen=True)
class QtUiSettings:
    """Small persisted settings model for the Qt shell."""

    theme: str = "light"
    ui_scale: float = 1.0
    start_route: str = DEFAULT_ROOT_ROUTE_ID
    show_advanced: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "QtUiSettings":
        data = dict(payload or {})
        return cls(
            theme=_normalize_theme(data.get("theme")),
            ui_scale=_normalize_scale(data.get("ui_scale")),
            start_route=_normalize_route(data.get("start_route")),
            show_advanced=bool(data.get("show_advanced", False)),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def with_updates(self, **changes: Any) -> "QtUiSettings":
        payload = {**self.to_mapping(), **changes}
        return QtUiSettings.from_mapping(payload)


class QtSettingsPresenter:
    """Load and save lightweight Qt UI preferences."""

    def __init__(self, *, repo_root: str | Path | None = None, path: str | Path | None = None) -> None:
        root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
        self.repo_root = root.resolve()
        self.path = Path(path) if path is not None else self.repo_root / "data" / "generated" / "qt_ui_settings.json"
        self._settings = self.load()

    @property
    def settings(self) -> QtUiSettings:
        return replace(self._settings)

    def load(self) -> QtUiSettings:
        if not self.path.exists():
            return QtUiSettings()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return QtUiSettings()
        if not isinstance(payload, Mapping):
            return QtUiSettings()
        return QtUiSettings.from_mapping(payload)

    def save(self, settings: QtUiSettings | None = None) -> QtUiSettings:
        self._settings = settings or self._settings
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._settings.to_mapping(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.settings

    def update(self, **changes: Any) -> QtUiSettings:
        self._settings = self._settings.with_updates(**changes)
        return self.save()


def allowed_scales() -> tuple[float, ...]:
    return _SCALE_VALUES


def _normalize_theme(value: Any) -> str:
    return str(value) if str(value) in _THEME_VALUES else "light"


def _normalize_scale(value: Any) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 1.0
    return min(_SCALE_VALUES, key=lambda item: abs(item - raw))


def _normalize_route(value: Any) -> str:
    route_id = str(value or "")
    return route_id if route_id in ROOT_ROUTE_BY_ID else DEFAULT_ROOT_ROUTE_ID
