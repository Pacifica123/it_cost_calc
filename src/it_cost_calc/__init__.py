"""Compatibility namespace for historical ``it_cost_calc.*`` imports.

The canonical implementation lives in the top-level packages under ``src``:
``application``, ``domain``, ``infrastructure``, ``shared`` and ``ui``.
Files inside this package are intentionally thin import wrappers, not a second
copy of the implementation.
"""

from __future__ import annotations

__all__ = ["application", "domain", "infrastructure", "shared", "ui"]
