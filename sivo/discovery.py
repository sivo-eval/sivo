"""Eval function discovery for sivo.

Finds ``eval_*.py`` files, imports them, and returns the callable
``eval_*`` functions inside. Also detects companion ``eval_*_cases()``
functions for data-driven evals (Phase 9).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Callable


@dataclass
class DiscoveredEval:
    """A single discovered eval function with its metadata."""

    name: str
    """Function name, e.g. ``eval_tone``."""

    func: Callable
    """The eval function object."""

    source_file: Path
    """Absolute path to the ``.py`` file it was loaded from."""

    cases_func: Callable | None = None
    """Companion ``eval_<name>_cases()`` generator, if present (Phase 9)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def discover_eval_files(path: Path) -> list[Path]:
    """Return all ``eval_*.py`` files reachable from *path*.

    If *path* is a file it is returned directly (if it matches the naming
    convention). If *path* is a directory the tree is searched recursively.

    Args:
        path: A file or directory to search.

    Returns:
        Sorted list of absolute ``Path`` objects.
    """
    path = path.resolve()
    if path.is_file():
        if _is_eval_file(path):
            return [path]
        return []
    return sorted(path.rglob("eval_*.py"))


def load_eval_functions(source_file: Path) -> list[DiscoveredEval]:
    """Import *source_file* and return all ``eval_*`` functions it defines.

    Functions whose names end in ``_cases`` are treated as data-driven
    case generators and paired with their sibling eval function rather than
    returned as standalone evals.

    Args:
        source_file: Path to a Python file to load.

    Returns:
        ``DiscoveredEval`` objects sorted by function name.
    """
    source_file = source_file.resolve()
    module = _load_module(source_file)

    # Collect all callables defined in this module (not imported).
    local_callables: dict[str, Callable] = {}
    for name, obj in vars(module).items():
        if not callable(obj):
            continue
        obj_module = getattr(obj, "__module__", None)
        if obj_module != module.__name__:
            continue
        local_callables[name] = obj

    # Pass 1: standalone eval functions — start with "eval_" but do NOT end
    # with "_cases", OR end with "_cases" only if no matching "eval_X" sibling
    # exists (handles names like "eval_edge_cases" that look like cases funcs).
    eval_names: list[str] = []
    potential_cases: dict[str, Callable] = {}  # base → cases func

    for name, obj in local_callables.items():
        if not name.startswith("eval_"):
            continue
        if name.endswith("_cases"):
            base = name[: -len("_cases")]
            potential_cases[base] = obj
        else:
            eval_names.append(name)

    # Pass 2: resolve which potential_cases entries are genuine companions
    # (the sibling eval function exists) vs standalone eval functions.
    cases_map: dict[str, Callable] = {}
    for base, cases_func in potential_cases.items():
        if base in eval_names:
            # Genuine cases generator — paired with its sibling
            cases_map[base] = cases_func
        else:
            # No sibling found — treat as a standalone eval function
            eval_names.append(f"{base}_cases")

    return [
        DiscoveredEval(
            name=name,
            func=local_callables[name],
            source_file=source_file,
            cases_func=cases_map.get(name),
        )
        for name in sorted(eval_names)
    ]


def discover(
    path: Path,
    *,
    eval_filter: str | None = None,
) -> list[DiscoveredEval]:
    """Discover all eval functions under *path*, with optional name filter.

    Args:
        path: File or directory to search.
        eval_filter: If given, only return evals whose ``name`` matches
                     exactly (e.g. ``"eval_tone"``).

    Returns:
        List of :class:`DiscoveredEval` objects, ordered by source file
        then function name within each file.
    """
    evals: list[DiscoveredEval] = []
    for f in discover_eval_files(path):
        evals.extend(load_eval_functions(f))

    if eval_filter is not None:
        evals = [e for e in evals if e.name == eval_filter]

    return evals


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_eval_file(path: Path) -> bool:
    return path.name.startswith("eval_") and path.suffix == ".py"


def _module_name(source_file: Path) -> str:
    """Return the deterministic module name used for *source_file*."""
    return f"_sivo_eval_{source_file.stem}_{abs(hash(str(source_file)))}"


def get_loaded_module(source_file: Path) -> ModuleType | None:
    """Return the already-loaded module for *source_file*, or ``None``.

    After :func:`load_eval_functions` has been called for *source_file* the
    module will be in ``sys.modules`` under a deterministic name.  This
    function looks it up without re-loading.

    Args:
        source_file: Absolute path to the eval source file.

    Returns:
        The loaded :class:`~types.ModuleType`, or ``None`` if not yet loaded.
    """
    return sys.modules.get(_module_name(source_file.resolve()))


def _load_module(source_file: Path) -> ModuleType:
    """Import *source_file* as an isolated module and return it."""
    # Use a deterministic but collision-free name so repeated discovery
    # across test runs does not share module state.
    module_name = _module_name(source_file)

    # Re-use a cached module if it's already loaded (idempotent per file)
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {source_file}")

    module = importlib.util.module_from_spec(spec)
    # Set __file__ so relative imports inside the eval file work as expected
    module.__file__ = str(source_file)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module
