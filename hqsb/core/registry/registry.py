"""Generic plugin registry for HQSB extension points.

Backends, Operators, Quantizers, Monitors, and Reporters all plug into the
system through a registry. A registry enforces:

* **unique names** — registering a second entry under an existing name is
  rejected unless it is the *same* entry (idempotent re-register),
* **optional versioning** — a caller may supply a version tag used for
  conflict detection,
* **explicit lookup failures** — requesting a missing entry raises
  :class:`RegistryLookupError` rather than returning ``None``,
* **explicit unload** — entries can be removed and reported as missing.

This is the single mechanism through which ``benchmark`` discovers a
``Backend`` without importing any concrete backend module (top-level
architecture §3.1/§3.2, S01 §5).
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterator, Optional, TypeVar

from hqsb.core.errors import (
    DuplicateRegistrationError,
    RegistryLookupError,
)

# The registered object type.
_T = TypeVar("_T")

# A factory callable returning (or being) the registered object.
Factory = Callable[[], _T]


class Registry(Generic[_T]):
    """A name → object registry with conflict and lookup enforcement."""

    def __init__(self, kind: str = "entry") -> None:
        self.kind = kind
        self._entries: Dict[str, _T] = {}
        self._versions: Dict[str, Optional[str]] = {}

    def register(
        self,
        name: str,
        obj: _T,
        *,
        version: Optional[str] = None,
        replace: bool = False,
    ) -> None:
        """Register ``obj`` under ``name``.

        Args:
            name: Unique entry name.
            obj: The object to register.
            version: Optional version tag recorded for diagnostics.
            replace: If True, allow replacing an existing entry; otherwise
                re-registering a *different* object under the same name
                raises :class:`DuplicateRegistrationError`.

        Raises:
            DuplicateRegistrationError: If ``name`` is taken by a different
                object and ``replace`` is False.
        """
        if name in self._entries and not replace:
            existing_version = self._versions.get(name)
            if existing_version == version:
                if self._entries[name] is obj:
                    return  # idempotent re-register of the same object+version
                raise DuplicateRegistrationError(
                    f"duplicate {self.kind} registration for name {name!r}"
                    + (f" (version {version})" if version else "")
                )
            raise DuplicateRegistrationError(
                f"duplicate {self.kind} registration for name {name!r}: "
                f"version {version!r} conflicts with registered "
                f"version {existing_version!r}"
            )
        self._entries[name] = obj
        self._versions[name] = version

    def get(self, name: str) -> _T:
        """Return the entry registered under ``name``.

        Raises:
            RegistryLookupError: If ``name`` is not registered.
        """
        try:
            return self._entries[name]
        except KeyError:
            raise RegistryLookupError(
                f"{self.kind} {name!r} is not registered"
            ) from None

    def unregister(self, name: str) -> None:
        """Remove the entry registered under ``name``.

        Raises:
            RegistryLookupError: If ``name`` is not registered.
        """
        if name not in self._entries:
            raise RegistryLookupError(
                f"cannot unregister {self.kind} {name!r}: not registered"
            )
        del self._entries[name]
        self._versions.pop(name, None)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def names(self) -> Iterator[str]:
        """Yield all registered names (insertion order)."""
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


class RegistryHub:
    """Aggregates the standard HQSB registries under one object.

    Attributes:
        backends: :class:`Registry` for backend classes/factories.
        operators: :class:`Registry` for operator implementations.
        quantizers: :class:`Registry` for quantizer implementations.
        monitors: :class:`Registry` for resource monitors.
        reporters: :class:`Registry` for result reporters.
    """

    def __init__(self) -> None:
        self.backends: Registry = Registry(kind="backend")
        self.operators: Registry = Registry(kind="operator")
        self.quantizers: Registry = Registry(kind="quantizer")
        self.monitors: Registry = Registry(kind="monitor")
        self.reporters: Registry = Registry(kind="reporter")


__all__ = [
    "Factory",
    "Registry",
    "RegistryHub",
]
