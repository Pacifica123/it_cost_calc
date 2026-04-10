"""Compatibility adapter exposing the legacy Treeview-based CRUD facade."""

from infrastructure.repositories.treeview_crud_repository import TreeviewCrudRepository

CRUD = TreeviewCrudRepository

__all__ = ["CRUD", "TreeviewCrudRepository"]
