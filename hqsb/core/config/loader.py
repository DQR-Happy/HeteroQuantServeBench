"""Unified configuration loading with layered precedence and hashing.

The loader merges configuration from four sources, in ascending precedence:

1. programmatic defaults,
2. a YAML file,
3. environment variables (``HQSB_`` prefix),
4. explicit CLI overrides (a flat/nested dict).

After merging, the result is validated against a Pydantic schema model with
``extra="forbid"`` semantics: unknown fields are rejected, and values are
type-checked.  A deterministic SHA256 ``config_hash`` is derived from the
final, validated configuration so benchmark results can bind to an exact
configuration (top-level architecture §3.3, S01 §4).

E01-02 additions (config precedence & identity) are layered on top without
changing the original public API:

* ``_read_yaml_strict`` — reject duplicate YAML keys before they are silently
  overwritten during merge;
* ``load_resolved`` — returns a :class:`ConfigResolution` with a per-field
  source map, per-layer sanitized inputs, redacted public view, source file
  SHA256, and a *semantic* identity hash that excludes operational/secret
  fields;
* per-layer structural validation — each provided source (file/env/CLI) is
  checked for unknown fields and type/constraint errors *before* merging, so
  a lower-precedence invalid value cannot hide behind a higher one.
"""

from __future__ import annotations

import hashlib
import json
import os
import types
from dataclasses import dataclass, field as dataclass_field
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError

from hqsb.core.errors import ConfigError, HqsbError

_T = TypeVar("_T", bound=BaseModel)

DEFAULT_ENV_PREFIX = "HQSB_"
DEFAULT_ENV_SEPARATOR = "__"

#: Source-layer labels used in the source map, lowest → highest precedence.
_LAYER_ORDER: Tuple[str, ...] = ("defaults", "file", "env", "cli")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Nested mappings are merged recursively; non-mapping values in
    ``override`` replace the corresponding value in ``base``.  Lists are
    replaced whole (never implicitly concatenated), per E01-02 frozen rules.

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
    JSON.  A bare ``"false"``/``"null"`` is valid JSON and parses to the
    right Python scalar; a free-text ``"abc"`` stays a string so the schema
    can reject it instead of mis-typing it.
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


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys at the file location.

    A duplicate key would otherwise be silently overwritten by
    :func:`yaml.safe_load` *before* the four-layer merge, hiding a broken
    file (E01-02 step 5: "YAML duplicate key, reject with location").
    """


def _construct_mapping_unique(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> Dict[Any, Any]:
    mapping: Dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            mark = key_node.start_mark
            raise ConfigError(
                f"duplicate YAML key {key!r} at line {mark.line + 1}, "
                f"column {mark.column + 1}",
                details={
                    "error_code": "duplicate_yaml_key",
                    "source": "file",
                    "field_path": str(key),
                    "line": mark.line + 1,
                    "column": mark.column + 1,
                },
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping_unique
)


def _read_yaml_strict(path: str) -> Dict[str, Any]:
    """Load YAML, rejecting duplicate keys and malformed syntax (step 5)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.load(fh, Loader=_UniqueKeyLoader)
    except ConfigError:
        raise
    except FileNotFoundError:
        raise ConfigError(
            f"config file not found: {path}",
            details={"error_code": "file_not_found", "source": "file"},
        ) from None
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        details = {"error_code": "invalid_yaml", "source": "file"}
        if mark is not None:
            details["line"] = mark.line + 1
            details["column"] = mark.column + 1
        raise ConfigError(f"invalid YAML in {path}: {exc}", details=details) from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"config file {path} must contain a mapping at top level",
            details={"error_code": "invalid_yaml", "source": "file"},
        )
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


def sha256_hex(text: str) -> str:
    """Return the lowercase SHA256 hex digest of ``text`` (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _leaf_paths(mapping: Mapping[str, Any], prefix: str = "") -> List[str]:
    """Return dotted leaf paths of a mapping, treating lists as single leaves.

    Nested dicts are flattened into ``a.b.c`` leaves; a list value (and
    everything under it) is one leaf at its containing key, because lists are
    replaced whole and never merged per-element (E01-02 frozen rule).
    """
    paths: List[str] = []
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            paths.extend(_leaf_paths(value, path))
        else:
            paths.append(path)
    return paths


def _redact_by_paths(obj: Any, secret_paths: Tuple[str, ...], prefix: str = "") -> Any:
    """Return a deep copy of ``obj`` with secret leaf values redacted."""
    if isinstance(obj, dict):
        return {
            k: (
                "<redacted>"
                if (f"{prefix}.{k}" if prefix else k) in secret_paths
                else _redact_by_paths(v, secret_paths, f"{prefix}.{k}" if prefix else k)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact_by_paths(v, secret_paths, prefix) for v in obj]
    return obj


def redact_text(text: str, secret_values: List[str]) -> str:
    """Replace any known secret plaintext occurrences in ``text``."""
    for value in secret_values:
        if value:
            text = text.replace(value, "<redacted>")
    return text


def _collect_secret_values(mapping: Mapping[str, Any], secret_paths: Tuple[str, ...]) -> List[str]:
    """Collect the plaintext values currently at the secret field paths."""
    values: List[str] = []

    def walk(mapping: Mapping[str, Any], prefix: str) -> None:
        for key, value in mapping.items():
            path = f"{prefix}.{key}" if prefix else key
            if path in secret_paths:
                if isinstance(value, str):
                    values.append(value)
            elif isinstance(value, dict):
                walk(value, path)
    walk(mapping, "")
    return values


def _strip_annotated(ann: Any) -> Tuple[Any, List[Any]]:
    """Remove outer ``Annotated`` wrappers, returning ``(base, metadata)``."""
    extra: List[Any] = []
    while get_origin(ann) is Annotated:
        args = get_args(ann)
        ann = args[0]
        extra = list(args[1:]) + extra
    return ann, extra


def _is_optional(ann: Any) -> bool:
    origin = get_origin(ann)
    if origin in (Union, types.UnionType):
        return type(None) in get_args(ann)
    return False


def _strip_optional(ann: Any) -> Any:
    origin = get_origin(ann)
    if origin in (Union, types.UnionType):
        non_none = [a for a in get_args(ann) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return ann


def _field_adapter(annotation: Any, metadata: List[Any]) -> TypeAdapter:
    """Build a :class:`TypeAdapter` honoring annotation + field constraints."""
    base, extra = _strip_annotated(annotation)
    all_meta = extra + list(metadata)
    if all_meta:
        return TypeAdapter(Annotated.__class_getitem__((base, *all_meta)))
    return TypeAdapter(base)


def _check_layer_shape(
    model_cls: Type[BaseModel],
    value: Any,
    *,
    source: str,
    path: str,
    secret_values: List[str],
) -> None:
    """Validate one provided layer's *shape* (unknown keys, types, constraints).

    This is a missing-tolerant structural check: only keys actually present in
    the layer are validated, so a partial overlay is legal, but a wrong-typed
    or unknown key is rejected *at its own source* even when a later layer
    would have overwritten it (E01-02 frozen strict policy).

    ``path`` is the dotted path used for error localization.
    """
    if not isinstance(value, Mapping):
        raise ConfigError(
            f"{source}: expected a mapping at {path or '<root>'}, "
            f"got {type(value).__name__}"
        )
    for key, item in value.items():
        field_path = f"{path}.{key}" if path else key
        if key not in model_cls.model_fields:
            raise ConfigError(
                f"unknown field {field_path!r} (source: {source})",
                details={"error_code": "unknown_field", "source": source,
                         "field_path": field_path},
            )
        field = model_cls.model_fields[key]

        if item is None:
            if _is_optional(field.annotation):
                continue
            raise ConfigError(
                f"field {field_path!r} cannot be null (source: {source})",
                details={"error_code": "null_not_allowed", "source": source,
                         "field_path": field_path},
            )

        inner = _strip_optional(_strip_annotated(field.annotation)[0])
        origin = get_origin(inner)

        # Nested model object.
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            _check_layer_shape(
                inner, item, source=source, path=field_path,
                secret_values=secret_values,
            )
            continue

        # List of model objects (e.g. workloads).
        if origin is list:
            (item_ann,) = get_args(inner)
            if isinstance(item_ann, type) and issubclass(item_ann, BaseModel):
                if not isinstance(item, list):
                    raise ConfigError(
                        f"field {field_path!r} must be a list (source: {source})",
                        details={"error_code": "type_error", "source": source,
                                 "field_path": field_path},
                    )
                for idx, entry in enumerate(item):
                    _check_layer_shape(
                        item_ann, entry, source=source,
                        path=f"{field_path}[{idx}]", secret_values=secret_values,
                    )
                continue

        # Scalar / list-of-scalar / literal leaf.
        adapter = _field_adapter(field.annotation, list(field.metadata))
        try:
            adapter.validate_python(item)
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else {}
            raise ConfigError(
                f"field {field_path!r} has invalid value "
                f"{redact_text(str(item), secret_values)!r} "
                f"(source: {source}; {first.get('type', 'value_error')})",
                details={"error_code": "type_error", "source": source,
                         "field_path": field_path, "pydantic_type": first.get("type", "")},
            ) from exc


@dataclass
class ConfigResolution:
    """Full resolution record for one config load (E01-02 evidence unit)."""

    config: BaseModel
    resolved_dump: Dict[str, Any]
    public_view: Dict[str, Any]
    source_map: Dict[str, str]
    layer_inputs: Dict[str, Any]
    source_file_sha256: Optional[str]
    semantic_payload: Dict[str, Any]
    config_hash: str
    full_config_hash: str
    env_keys_used: List[str] = dataclass_field(default_factory=list)


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

    # ── Identity rule discovery ─────────────────────────────────────────

    def _secret_paths(self) -> Tuple[str, ...]:
        return tuple(getattr(self.model, "SECRET_FIELD_PATHS", ()) or ())

    def _operational_paths(self) -> Tuple[str, ...]:
        return tuple(getattr(self.model, "OPERATIONAL_FIELD_PATHS", ()) or ())

    # ── Core load ───────────────────────────────────────────────────────

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
        return self.load_resolved(
            defaults=defaults, path=path, environ=environ, cli=cli
        ).config

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
        resolution = self.load_resolved(
            defaults=defaults, path=path, environ=environ, cli=cli
        )
        return resolution.config, resolution.config_hash

    def load_resolved(
        self,
        *,
        defaults: Optional[Mapping[str, Any]] = None,
        path: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
        cli: Optional[Mapping[str, Any]] = None,
    ) -> ConfigResolution:
        """Load and resolve configuration, returning full provenance metadata.

        This is the E01-02 primary entry point: it merges the four layers,
        records a per-field source map, redacts secret values, and computes
        both the full and the semantic identity hash.
        """
        secret_paths = self._secret_paths()
        operational_paths = self._operational_paths()

        # 1. Assemble layers.
        layers: Dict[str, Any] = {}
        if defaults:
            layers["defaults"] = dict(defaults)

        file_data: Optional[Dict[str, Any]] = None
        source_file_sha256: Optional[str] = None
        if path:
            try:
                raw = open(path, "rb").read()
            except FileNotFoundError:
                raise ConfigError(
                    f"config file not found: {path}",
                    details={"error_code": "file_not_found", "source": "file"},
                ) from None
            source_file_sha256 = hashlib.sha256(raw).hexdigest()
            file_data = _read_yaml_strict(path)
            layers["file"] = file_data

        env = environ if environ is not None else os.environ
        env_data = _env_to_dict(env, self.env_prefix, self.env_separator)
        env_keys_used = [
            k for k in env if k.startswith(self.env_prefix)
        ]
        if env_data:
            layers["env"] = env_data

        if cli:
            layers["cli"] = dict(cli)

        # 2. Per-layer structural validation (unknown/type/constraints at source).
        secret_values = _collect_secret_values(
            {k: v for k, v in layers.items() if k != "defaults"}, secret_paths
        )
        for source in ("file", "env", "cli"):
            if source in layers and layers[source]:
                _check_layer_shape(
                    self.model,
                    layers[source],
                    source=source,
                    path="",
                    secret_values=secret_values,
                )

        # 3. Merge in fixed precedence.
        merged: Dict[str, Any] = {}
        for layer in _LAYER_ORDER:
            if layer in layers:
                merged = deep_merge(merged, layers[layer])

        # 4. Validate the full merged document.
        try:
            config = self.model.model_validate(merged)
        except HqsbError:
            # Preserve the stable taxonomy: version gate → SchemaError (4),
            # cross-field conflict → ConfigError (3), etc.
            raise
        except ValidationError as exc:
            raise ConfigError(
                redact_text(f"configuration validation failed: {exc}", secret_values),
                details={"error_code": "schema_validation", "source": "merged"},
            ) from exc
        except Exception as exc:  # pydantic custom errors surfaced directly
            raise ConfigError(
                redact_text(str(exc), secret_values),
                details={"error_code": "schema_validation", "source": "merged"},
            ) from exc

        # 5. Provenance + identity.
        resolved_dump = config.model_dump(mode="json")
        source_map = self._build_source_map(layers, resolved_dump)

        public_view = _redact_by_paths(resolved_dump, secret_paths)
        layer_inputs_public = {
            layer: _redact_by_paths(data, secret_paths)
            for layer, data in layers.items()
        }

        semantic_payload = self._semantic_payload(resolved_dump, operational_paths)
        semantic_payload = _redact_by_paths(semantic_payload, secret_paths)

        return ConfigResolution(
            config=config,
            resolved_dump=resolved_dump,
            public_view=public_view,
            source_map=source_map,
            layer_inputs=layer_inputs_public,
            source_file_sha256=source_file_sha256,
            semantic_payload=semantic_payload,
            config_hash=self._payload_hash(semantic_payload),
            full_config_hash=config_hash(config),
            env_keys_used=env_keys_used,
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    def _build_source_map(
        self, layers: Dict[str, Any], resolved_dump: Dict[str, Any]
    ) -> Dict[str, str]:
        """Attribute each resolved leaf path to its highest-precedence source."""
        declared: Dict[str, str] = {}
        for layer in _LAYER_ORDER:
            if layer not in layers:
                continue
            for leaf in _leaf_paths(layers[layer]):
                declared[leaf] = layer

        resolved_leaves = _leaf_paths(resolved_dump)
        source_map: Dict[str, str] = {}
        for leaf in resolved_leaves:
            # Fall back through precedence to find the declaring layer.
            source_map[leaf] = declared.get(leaf, "schema")
        return source_map

    def _semantic_payload(
        self, resolved_dump: Dict[str, Any], operational_paths: Tuple[str, ...]
    ) -> Dict[str, Any]:
        """Drop operational/secret fields from the identity payload."""

        def keep(path: str) -> bool:
            for op in operational_paths:
                if path == op or path.startswith(op + "."):
                    return False
            return True

        def filter_obj(obj: Any, prefix: str) -> Any:
            if isinstance(obj, dict):
                out: Dict[str, Any] = {}
                for k, v in obj.items():
                    path = f"{prefix}.{k}" if prefix else k
                    if not keep(path):
                        continue
                    out[k] = filter_obj(v, path)
                return out
            if isinstance(obj, list):
                return [filter_obj(v, prefix) for v in obj]
            return obj

        return filter_obj(resolved_dump, "")

    @staticmethod
    def _payload_hash(payload: Dict[str, Any]) -> str:
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "ConfigLoader",
    "ConfigResolution",
    "config_hash",
    "deep_merge",
    "sha256_hex",
]
