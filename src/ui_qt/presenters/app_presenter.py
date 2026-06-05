from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from application.services.entity_catalog_service import EntityCatalogService
from application.services.equipment_service import EquipmentService
from application.use_cases.load_demo_dataset import LoadDemoDatasetUseCase
from infrastructure.repositories.json_entity_repository import JsonEntityRepository
from infrastructure.storage import JsonFileStorage


@dataclass(frozen=True)
class QtRuntimePaths:
    """Filesystem paths used by the Qt adapter layer."""

    repo_root: Path
    runtime_entities_path: Path
    demo_dataset_path: Path

    @classmethod
    def from_repo_root(
        cls,
        repo_root: str | Path | None = None,
        *,
        runtime_entities_path: str | Path | None = None,
        demo_dataset_path: str | Path | None = None,
    ) -> "QtRuntimePaths":
        root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
        root = root.resolve()
        data_root = root / "data"
        return cls(
            repo_root=root,
            runtime_entities_path=Path(
                runtime_entities_path or data_root / "generated" / "runtime_entities.json"
            ),
            demo_dataset_path=Path(demo_dataset_path or data_root / "fixtures" / "demo_dataset.json"),
        )


class QtAppPresenter:
    """Boundary between future Qt screens and existing application services.

    The presenter is deliberately free from QWidget, Treeview and tkinter types.
    It owns the JSON-backed runtime repository and exposes small operations that
    future screens can call without knowing storage details.
    """

    def __init__(
        self,
        *,
        repo_root: str | Path | None = None,
        runtime_entities_path: str | Path | None = None,
        demo_dataset_path: str | Path | None = None,
        storage: JsonFileStorage | None = None,
        repository: JsonEntityRepository | None = None,
    ) -> None:
        self.paths = QtRuntimePaths.from_repo_root(
            repo_root,
            runtime_entities_path=runtime_entities_path,
            demo_dataset_path=demo_dataset_path,
        )
        self.storage = storage or JsonFileStorage()
        self.repository = repository or JsonEntityRepository(
            self.paths.runtime_entities_path,
            self.storage,
        )
        self.catalog_service = EntityCatalogService(self.repository)
        self.equipment_service = EquipmentService(self.repository)
        self.demo_loader = LoadDemoDatasetUseCase(self.storage, self.equipment_service)

    @property
    def entities(self) -> dict[str, list[dict[str, Any]]]:
        return self.repository.entities

    def entities_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return deepcopy(self.repository.entities)

    def list_entities(self, entity_name: str) -> list[dict[str, Any]]:
        return deepcopy(self.catalog_service.list_entities(entity_name))

    def get_entity(self, entity_name: str, row_index: int) -> dict[str, Any]:
        return deepcopy(self.catalog_service.get_entity(entity_name, row_index))

    def add_entity(self, entity_name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return deepcopy(self.catalog_service.add_entity(entity_name, dict(payload)))

    def update_entity(
        self,
        entity_name: str,
        row_index: int,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return deepcopy(self.catalog_service.update_entity(entity_name, row_index, dict(payload)))

    def delete_entity(self, entity_name: str, row_index: int) -> dict[str, Any]:
        return deepcopy(self.catalog_service.delete_entity(entity_name, row_index))

    def replace_entities(self, payload: Mapping[str, list[Mapping[str, Any]]]) -> None:
        normalized_payload = {
            str(entity_name): [dict(row) for row in rows]
            for entity_name, rows in dict(payload).items()
        }
        self.equipment_service.replace_all(normalized_payload)

    def load_demo_dataset(self, path: str | Path | None = None) -> dict[str, list[dict[str, Any]]]:
        dataset = self.demo_loader.execute(path or self.paths.demo_dataset_path)
        return deepcopy(dataset)

    def save(self) -> Path:
        return self.repository.save()

    def total_rows(self) -> int:
        return sum(len(rows) for rows in self.repository.entities.values())

    def has_runtime_data(self) -> bool:
        return self.total_rows() > 0

    def table_rows(self, entity_name: str) -> list[dict[str, Any]]:
        """Return plain rows ready for Qt table models."""

        return self.list_entities(entity_name)
