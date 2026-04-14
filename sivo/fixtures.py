"""Fixture system for sivo.

Provides the :func:`fixture` decorator (exposed as ``sivo.fixture``) and the
:class:`FixtureRegistry` that manages fixture lifecycle during a run.

Fixture scopes
--------------
``"session"`` (default)
    The fixture factory is called **once per run**. The same value is shared
    across all eval functions and all records. Teardown runs after the run
    completes.

``"eval"``
    The fixture factory is called **once per eval function**. The value is
    shared across all records processed by that eval. Teardown runs before
    moving to the next eval function.

Usage example::

    import sivo

    @sivo.fixture(scope="session")
    def shared_client():
        client = MyAPIClient()
        yield client          # teardown via generator
        client.close()

    @sivo.fixture(scope="eval")
    def eval_counter():
        return {"count": 0}   # simple return (no teardown)

    def eval_something(case, shared_client, eval_counter):
        response = case.output
        assert_contains(response, "expected")
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Generator, Literal

# Attribute name stamped on fixture-decorated functions.
_SCOPE_ATTR = "__sivo_fixture_scope__"


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def fixture(
    *,
    scope: Literal["session", "eval"] = "session",
) -> Callable:
    """Mark a function as an sivo fixture.

    The decorated function becomes a *factory* that the runner calls at the
    appropriate scope boundary. The return value (or the first ``yield`` value
    for generator factories) is injected into eval functions that declare a
    parameter with the same name.

    Generator fixtures support teardown::

        @sivo.fixture(scope="session")
        def my_db():
            db = connect()
            yield db
            db.close()  # runs after the session ends

    Args:
        scope: ``"session"`` (default) initialises the fixture once per run;
               ``"eval"`` initialises it once per eval function.

    Returns:
        A decorator that stamps ``__sivo_fixture_scope__`` on the function
        and returns it unchanged.
    """
    def decorator(fn: Callable) -> Callable:
        setattr(fn, _SCOPE_ATTR, scope)
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class FixtureRegistry:
    """Manages fixture factories and their per-scope cached values.

    Instantiate via :func:`collect_fixtures` rather than directly.

    The lifecycle methods must be called in order:

    1. :meth:`initialize_session` — once before the eval loop.
    2. :meth:`initialize_eval` — once before each eval function's records.
    3. :meth:`teardown_eval` — once after each eval function's records.
    4. :meth:`teardown_session` — once after all evals complete.
    """

    def __init__(self, factories: dict[str, Callable]) -> None:
        self._factories: dict[str, Callable] = factories
        self._session_values: dict[str, Any] = {}
        self._eval_values: dict[str, Any] = {}
        self._session_generators: dict[str, Generator] = {}
        self._eval_generators: dict[str, Generator] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize_session(self) -> None:
        """Call all session-scoped factories and cache their values."""
        for name, fn in self._factories.items():
            if getattr(fn, _SCOPE_ATTR, None) == "session":
                self._session_values[name] = self._call_factory(
                    fn, self._session_generators, name
                )

    def initialize_eval(self) -> None:
        """Call all eval-scoped factories and cache their values."""
        for name, fn in self._factories.items():
            if getattr(fn, _SCOPE_ATTR, None) == "eval":
                self._eval_values[name] = self._call_factory(
                    fn, self._eval_generators, name
                )

    def teardown_eval(self) -> None:
        """Run teardown for eval-scoped generator fixtures."""
        _run_generators(self._eval_generators)
        self._eval_generators.clear()
        self._eval_values.clear()

    def teardown_session(self) -> None:
        """Run teardown for session-scoped generator fixtures."""
        _run_generators(self._session_generators)
        self._session_generators.clear()
        self._session_values.clear()

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    def resolve(self, func: Callable) -> dict[str, Any]:
        """Return a ``kwargs`` dict for all fixture parameters of *func*.

        The first parameter of every eval function is ``case`` (the
        :class:`~sivo.models.EvalCase`). All subsequent parameters are
        treated as fixture requests.

        Args:
            func: The eval function whose non-``case`` parameters to resolve.

        Returns:
            Dict mapping parameter names to their resolved fixture values.

        Raises:
            ValueError: If a requested fixture is not registered.
        """
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        kwargs: dict[str, Any] = {}
        for name in params[1:]:  # skip the first param ('case')
            if name in self._session_values:
                kwargs[name] = self._session_values[name]
            elif name in self._eval_values:
                kwargs[name] = self._eval_values[name]
            else:
                raise ValueError(
                    f"Fixture {name!r} not found in the registry. "
                    "Make sure it is decorated with @sivo.fixture and is "
                    "importable from the eval file."
                )
        return kwargs

    def is_empty(self) -> bool:
        """Return ``True`` when no fixtures are registered."""
        return not self._factories

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _call_factory(
        fn: Callable,
        generators: dict[str, Generator],
        name: str,
    ) -> Any:
        if inspect.isgeneratorfunction(fn):
            gen = fn()
            generators[name] = gen
            return next(gen)
        return fn()


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------


def collect_fixtures(evals: list) -> FixtureRegistry:
    """Build a :class:`FixtureRegistry` by scanning the modules of *evals*.

    Scans the namespace of each discovered eval module (already in
    ``sys.modules``) for callables decorated with :func:`fixture`.  Both
    fixtures defined directly in the eval file and fixtures *imported* into
    it are collected.

    Args:
        evals: List of :class:`~sivo.discovery.DiscoveredEval` objects
               returned by :func:`~sivo.discovery.discover`.

    Returns:
        A :class:`FixtureRegistry` populated with all discovered fixtures.
    """
    from sivo.discovery import get_loaded_module

    factories: dict[str, Callable] = {}
    seen_files: set[Path] = set()

    for discovered in evals:
        if discovered.source_file in seen_files:
            continue
        seen_files.add(discovered.source_file)

        module = get_loaded_module(discovered.source_file)
        if module is None:
            continue

        for attr_name, obj in vars(module).items():
            if callable(obj) and hasattr(obj, _SCOPE_ATTR):
                # First definition wins (avoids duplicate fixture names from
                # multiple files that both import the same fixture).
                if attr_name not in factories:
                    factories[attr_name] = obj

    return FixtureRegistry(factories)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _run_generators(generators: dict[str, Generator]) -> None:
    """Advance each generator once to execute its teardown code."""
    for gen in generators.values():
        try:
            next(gen)
        except StopIteration:
            pass
