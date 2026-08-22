from __future__ import annotations

from ui_qt.dialogs.dns_catalog_import_dialog import DnsCatalogImportDialog
from ui_qt.presenters.catalog_staging_presenter import CatalogStagingPresenter

try:
    from PySide6.QtWidgets import QWidget
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    QWidget = object  # type: ignore[assignment,misc]


class YandexMarketCatalogImportDialog(DnsCatalogImportDialog):
    """Run the Yandex Market collector using the shared cancellable UI."""

    def __init__(
        self,
        presenter: CatalogStagingPresenter,
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        super().__init__(presenter, parent, source="yandex_market")


__all__ = ["YandexMarketCatalogImportDialog"]
