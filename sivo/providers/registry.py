"""Provider registry for sivo.

Maps provider names to concrete provider classes and resolves custom
providers from import paths (``"my_package.module:ClassName"``).
"""

from __future__ import annotations

import importlib

from sivo.providers import Provider
from sivo.providers.anthropic import AnthropicProvider

# Built-in provider names — resolved lazily so optional deps (openai) don't
# break on import when not installed.
_BUILTIN_PROVIDER_NAMES: frozenset[str] = frozenset({"anthropic", "openai"})

# Eagerly-available built-ins (no optional deps required).
_BUILTIN_PROVIDERS: dict[str, type] = {
    "anthropic": AnthropicProvider,
}


def _get_openai_provider_class() -> type:
    """Lazily import and return OpenAIProvider, raising a clear error if missing."""
    try:
        from sivo.providers.openai import OpenAIProvider  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "OpenAI provider requires the 'openai' package. "
            "Install it with: pip install sivo[openai]"
        ) from exc
    return OpenAIProvider


def get_provider(name: str, *, api_key: str | None = None) -> Provider:
    """Resolve a provider by name or dotted import path and return an instance.

    Args:
        name:    Built-in name (``"anthropic"`` or ``"openai"``) or a fully-qualified
                 import path (``"my_package.module:ClassName"``).
        api_key: API key forwarded to the provider constructor.

    Returns:
        A provider instance that satisfies the :class:`~sivo.providers.Provider`
        protocol.

    Raises:
        ValueError:  If *name* is not a known built-in and has no ``:`` separator.
        TypeError:   If the resolved class does not implement :class:`~sivo.providers.Provider`.
        ImportError: If the module in a custom import path cannot be imported, or
                     if an optional dependency (e.g. ``openai``) is not installed.
        AttributeError: If the class name is not found in the module.
    """
    if name in _BUILTIN_PROVIDERS:
        return _BUILTIN_PROVIDERS[name](api_key=api_key)

    if name == "openai":
        cls = _get_openai_provider_class()
        return cls(api_key=api_key)

    # Custom provider: "my_package.module:ClassName"
    if ":" not in name:
        known = ", ".join(f'"{k}"' for k in sorted(_BUILTIN_PROVIDER_NAMES))
        raise ValueError(
            f"Unknown provider {name!r}. "
            f"Built-in providers: [{known}]. "
            f"For a custom provider use 'my_package.module:ClassName'."
        )

    module_path, class_name = name.rsplit(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if not isinstance(cls, type):
        raise TypeError(f"{name!r} resolves to a non-class object: {cls!r}")

    instance = cls(api_key=api_key)
    if not isinstance(instance, Provider):
        raise TypeError(
            f"{name!r} does not implement the Provider protocol. "
            f"Ensure it has 'name', 'complete()', and 'judge()' attributes."
        )
    return instance
