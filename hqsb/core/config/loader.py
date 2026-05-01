"""Unified configuration loading with layered precedence and hashing.

The loader merges configuration from four sources, in ascending precedence:

1. programmatic defaults,
2. a YAML file,
3. environment variables (``HQSB_`` prefix),
4. explicit CLI overrides (a flat/nested dict).

After merging, the result is validated against a Pydantic schema model with
``extra="forbid"`` semantics: unknown fields are rejected, and values are
type-checked. A deterministic SHA256 ``config_hash`` is derived from the
final, validated configuration so benchmark results can bind to an exact
configuration (top-level architecture §3.3, S01 §4).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypeVar

import yaml
from pydantic import BaseModel

from hqsb.core.errors import ConfigError

_T = TypeVar("_T", bound=BaseModel)

DEFAULT_ENV_PREFIX = "HQSB_"
DEFAULT_ENV_SEPARATOR = "__"


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Nested mappings are merged recursively; non-mapping values in
    ``override`` replace the corresponding value in ``base``.

    Args:
        base: The lower-precedence mapping (not mutated).
        override: The higher-precedence mapping.

    Returns:
        A new dict with ``override`` applied on top of ``base``.
    """
    result: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _coerce_scalar(value: str) -> Any:
    """Best-effort coercion of an environment-variable string.

    Prefers JSON parsing (so booleans, numbers, lists, and nested objects
    round-trip), falling back to the raw string when the value is not valid
    JSON.
    """
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def _set_nested(target: Dict[str, Any], path: str, value: Any, separator: str) -> None:
    """Set ``value`` at a nested ``path`` (e.g. ``a__b`` -> ``target[a][b]``)."""
    parts = path.split(separator)
    cursor = target
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _env_to_dict(environ: Mapping[str, str], prefix: str, separator: str) -> Dict[str, Any]:
    """Convert ``PREFIX``-prefixed environment variables into a nested dict."""
    result: Dict[str, Any] = {}
    for key, value in environ.items():
        if not key.startswith(prefix):
            continue
        # Environment variables are conventionally upper-case; lower-case
        # them so ``HQSB_BATCH_SIZE`` maps to the ``batch_size`` field.
        stripped = key[len(prefix):].lower()
        if not stripped:
            continue
        _set_nested(result, stripped, _coerce_scalar(value), separator)
    return result


def _read_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file into a dict, raising :class:`ConfigError` on failure."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a mapping at top level")
    return data


def config_hash(config: BaseModel) -> str:
    """Compute a deterministic SHA256 hash of a validated config model.

    The hash is stable across process runs on the same Python version
    because it serializes with sorted keys, no whitespace, and no locale
    dependence.
    """
    canonical = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ConfigLoader:
    """Layered configuration loader producing a validated schema model."""

    def __init__(
        self,
        model: Type[_T],
        *,
        env_prefix: str = DEFAULT_ENV_PREFIX,
        env_separator: str = DEFAULT_ENV_SEPARATOR,
    ) -> None:
        self.model = model
        self.env_prefix = env_prefix
        self.env_separator = env_separator

    def load(
        self,
        *,
        defaults: Optional[Mapping[str, Any]] = None,
        path: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
        cli: Optional[Mapping[str, Any]] = None,
    ) -> _T:
        """Load, merge, validate, and return a configuration model.

        Args:
            defaults: Lowest-precedence programmatic defaults.
            path: Optional YAML config file path.
            environ: Environment mapping (defaults to ``os.environ``).
            cli: Highest-precedence CLI overrides (nested dict).

        Returns:
            An instance of the schema model, fully validated.

        Raises:
            ConfigError: If a file is missing/malformed or if the merged
                configuration fails schema validation (unknown field, wrong
                type, missing required field).
        """
        merged: Dict[str, Any] = {}
        if defaults:
            merged = deep_merge(merged, defaults)
        if path:
            merged = deep_merge(merged, _read_yaml(path))
        env = environ if environ is not None else os.environ
        merged = deep_merge(
            merged, _env_to_dict(env, self.env_prefix, self.env_separator)
        )
        if cli:
            merged = deep_merge(merged, dict(cli))

        try:
            return self.model.model_validate(merged)
        except Exception as exc:  # pydantic ValidationError
            raise ConfigError(
                f"configuration validation failed: {exc}"
            ) from exc

    def load_with_hash(
        self,
        *,
        defaults: Optional[Mapping[str, Any]] = None,
        path: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
        cli: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[_T, str]:
        """Like :meth:`load`, but also returns the config hash.

        Returns:
            A ``(config, hash)`` tuple.
        """
        config = self.load(
            defaults=defaults, path=path, environ=environ, cli=cli
        )
        return config, config_hash(config)


__all__ = [
    "ConfigLoader",
    "config_hash",
    "deep_merge",
]
