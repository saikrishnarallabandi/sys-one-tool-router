"""Model registry.

Every module in this package may define `VARIANTS: dict[str, (factory, kwargs)]`. The registry
collects them, so adding an adapter is one new file here: subclass `DecisionModel`, export
`VARIANTS`. Modules must import heavy dependencies (torch, SDKs) lazily inside the adapter so that
listing the registry stays cheap.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable

from decisionbench.models.base import DecisionModel

_REGISTRY: dict[str, tuple[Callable[..., DecisionModel], dict[str, Any]]] | None = None


def registry() -> dict[str, tuple[Callable[..., DecisionModel], dict[str, Any]]]:
    global _REGISTRY
    if _REGISTRY is None:
        found: dict[str, tuple[Callable[..., DecisionModel], dict[str, Any]]] = {}
        for mod in pkgutil.iter_modules(__path__):
            if mod.name == "base" or mod.name.startswith("_"):
                continue
            module = importlib.import_module(f"{__name__}.{mod.name}")
            for name, entry in getattr(module, "VARIANTS", {}).items():
                if name in found:
                    raise RuntimeError(f"model name {name!r} is registered twice (second in {module.__name__})")
                found[name] = entry
        _REGISTRY = found
    return _REGISTRY


def create(name: str, **overrides: Any) -> DecisionModel:
    """Instantiate a registered model; `overrides` are merged over the variant's kwargs."""
    reg = registry()
    if name not in reg:
        raise KeyError(f"unknown model {name!r}; registered: {', '.join(sorted(reg))}")
    factory, kwargs = reg[name]
    model = factory(**{**kwargs, **overrides})
    model.name = name
    return model


__all__ = ["DecisionModel", "create", "registry"]
