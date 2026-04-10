"""Совместимость со старым пространством имён пакета."""
from importlib import import_module

_real_package = import_module("infrastructure")

__path__ = _real_package.__path__
__all__ = getattr(_real_package, "__all__", [])

for _name in dir(_real_package):
    if _name.startswith("__") and _name not in {"__doc__", "__all__"}:
        continue
    globals()[_name] = getattr(_real_package, _name)
