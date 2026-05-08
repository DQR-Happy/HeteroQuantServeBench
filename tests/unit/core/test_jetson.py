"""Unit tests for the Jetson experiment protocol helpers.

Only tests the pure/defensive surface; sysfs probing is exercised via the
public functions and asserted to be well-typed (list/None/bool), never
raising on non-Jetson hosts.
"""

from __future__ import annotations

import pytest

from hqsb.hardware.jetson import (
    active_power_mode,
    is_jetson,
    max_temperature_c,
    read_thermal_zones,
)


@pytest.mark.unit
class TestProbeSurface:
    def test_is_jetson_returns_bool(self):
        assert isinstance(is_jetson(), bool)

    def test_read_thermal_zones_returns_list(self):
        result = read_thermal_zones()
        assert isinstance(result, list)
        for t in result:
            assert isinstance(t, float)

    def test_max_temperature_is_none_or_float(self):
        result = max_temperature_c()
        assert result is None or isinstance(result, float)

    def test_active_power_mode_is_none_or_int(self):
        result = active_power_mode()
        assert result is None or isinstance(result, int)
