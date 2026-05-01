"""Unit tests for the plugin registry and RegistryHub."""

from __future__ import annotations

import pytest

from hqsb.core.errors import (
    DuplicateRegistrationError,
    RegistryLookupError,
)
from hqsb.core.registry import Registry, RegistryHub


class _Plugin:
    def __init__(self, value):
        self.value = value


@pytest.mark.unit
class TestRegistry:
    def test_register_and_get(self):
        reg = Registry(kind="backend")
        obj = _Plugin(1)
        reg.register("a", obj)
        assert reg.get("a") is obj
        assert "a" in reg
        assert len(reg) == 1

    def test_lookup_missing_raises(self):
        reg = Registry(kind="backend")
        with pytest.raises(RegistryLookupError):
            reg.get("missing")

    def test_duplicate_different_object_raises(self):
        reg = Registry(kind="backend")
        reg.register("a", _Plugin(1))
        with pytest.raises(DuplicateRegistrationError):
            reg.register("a", _Plugin(2))

    def test_idempotent_register_same_object(self):
        reg = Registry(kind="backend")
        obj = _Plugin(1)
        reg.register("a", obj)
        reg.register("a", obj)  # no raise
        assert len(reg) == 1

    def test_replace_flag(self):
        reg = Registry(kind="backend")
        reg.register("a", _Plugin(1))
        new = _Plugin(2)
        reg.register("a", new, replace=True)
        assert reg.get("a") is new

    def test_unregister(self):
        reg = Registry(kind="backend")
        reg.register("a", _Plugin(1))
        reg.unregister("a")
        assert "a" not in reg
        with pytest.raises(RegistryLookupError):
            reg.unregister("a")

    def test_names_iteration(self):
        reg = Registry(kind="backend")
        reg.register("a", _Plugin(1))
        reg.register("b", _Plugin(2))
        assert list(reg.names()) == ["a", "b"]


@pytest.mark.unit
class TestRegistryHub:
    def test_has_standard_registries(self):
        hub = RegistryHub()
        for name in ("backends", "operators", "quantizers", "monitors", "reporters"):
            assert isinstance(getattr(hub, name), Registry)

    def test_registries_are_independent(self):
        hub = RegistryHub()
        hub.backends.register("b", _Plugin(1))
        hub.operators.register("b", _Plugin(2))
        assert hub.backends.get("b").value == 1
        assert hub.operators.get("b").value == 2
