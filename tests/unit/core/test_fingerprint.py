"""Unit tests for run / environment fingerprinting (E00-03)."""

from __future__ import annotations

import pytest

from hqsb.core.fingerprint import (
    CommitSection,
    ConfigSection,
    DeviceSection,
    ENVIRONMENT_SECTION_NAMES,
    FingerprintSections,
    ModelSection,
    OsSection,
    PowerSection,
    PythonSection,
    RunFingerprint,
    VolatileObservations,
    canonical_json,
    collect_packages,
    compute_run_fingerprint,
    diff_sections,
    sha256_hex,
)


def _sections(**overrides) -> FingerprintSections:
    sections = FingerprintSections(
        os=OsSection(
            system="Linux", release="5.15.0", machine="aarch64",
            platform="Linux-test", libc="glibc 2.35",
        ),
        device=DeviceSection(
            cuda_available=True, device_count=1, device_names=["Orin"],
            compute_capabilities=[[8, 7]], cuda_runtime_version="12.6",
            cuda_driver_version="540.4.0", nvcc_version="12.6",
            board_compatible="tegra234", l4t_release="R36",
        ),
        python=PythonSection(
            version="3.10.12", implementation="CPython",
            executable="/usr/bin/python3",
        ),
        packages={"torch": "2.5.0", "numpy": "1.23.5"},
        power=PowerSection(
            nvpmodel_mode=2, nvpmodel_name="MAXN_SUPER",
            jetson_clocks_active=False, cpu_governor="schedutil",
            gpu_governor="nvhost_podgov",
        ),
        config=ConfigSection(
            config_path="configs/benchmarks/x.yaml",
            config_sha256="a" * 64, config_hash="b" * 64,
        ),
        model=ModelSection(
            model_id="Qwen/Qwen3-1.7B",
            manifest_path="docs/benchmark/manifest.txt",
            manifest_sha256="c" * 64, model_hash="d" * 64,
        ),
        commit=CommitSection(commit="e" * 40, commit_short="e" * 7, dirty=False),
    )
    for key, value in overrides.items():
        setattr(sections, key, value)
    return sections


def _volatile(**overrides) -> VolatileObservations:
    values = {
        "observed_at_utc": "2026-09-03T00:00:00Z",
        "cpu_cur_freq_khz": 1190400,
        "gpu_cur_freq_hz": 306000000,
        "max_temperature_c": 48.0,
    }
    values.update(overrides)
    return VolatileObservations(**values)


@pytest.mark.unit
class TestCanonicalHash:
    def test_sorted_keys_deterministic(self):
        assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'

    def test_sha256_hex(self):
        assert sha256_hex("abc") == hashlib_sha256("abc")


def hashlib_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.unit
class TestDeterminism:
    def test_identical_sections_identical_fingerprint(self):
        fp1 = compute_run_fingerprint(_sections(), _volatile())
        fp2 = compute_run_fingerprint(_sections(), _volatile())
        assert fp1.run_fingerprint == fp2.run_fingerprint
        assert fp1.environment_fingerprint == fp2.environment_fingerprint
        assert fp1.volatile_digest == fp2.volatile_digest

    def test_packages_insertion_order_does_not_matter(self):
        a = _sections(packages={"torch": "2.5.0", "numpy": "1.23.5"})
        b = _sections(packages={"numpy": "1.23.5", "torch": "2.5.0"})
        assert compute_run_fingerprint(a, _volatile()).run_fingerprint == (
            compute_run_fingerprint(b, _volatile()).run_fingerprint
        )


@pytest.mark.unit
class TestSectionIsolation:
    def test_environment_change_moves_env_and_run(self):
        base = compute_run_fingerprint(_sections(), _volatile())
        changed = compute_run_fingerprint(
            _sections(power=PowerSection(nvpmodel_mode=3, nvpmodel_name="IDLE_3")),
            _volatile(),
        )
        assert changed.environment_fingerprint != base.environment_fingerprint
        assert changed.run_fingerprint != base.run_fingerprint
        assert diff_sections(base, changed) == ["power"]

    def test_input_change_moves_run_only(self):
        base = compute_run_fingerprint(_sections(), _volatile())
        changed = compute_run_fingerprint(
            _sections(model=ModelSection(model_hash="f" * 64)), _volatile()
        )
        assert changed.environment_fingerprint == base.environment_fingerprint
        assert changed.run_fingerprint != base.run_fingerprint
        assert diff_sections(base, changed) == ["model"]

    def test_commit_change_moves_run_only(self):
        base = compute_run_fingerprint(_sections(), _volatile())
        changed = compute_run_fingerprint(
            _sections(commit=CommitSection(commit="0" * 40)), _volatile()
        )
        assert changed.environment_fingerprint == base.environment_fingerprint
        assert changed.run_fingerprint != base.run_fingerprint
        assert diff_sections(base, changed) == ["commit"]

    def test_volatile_change_moves_volatile_only(self):
        base = compute_run_fingerprint(_sections(), _volatile())
        changed = compute_run_fingerprint(
            _sections(), _volatile(max_temperature_c=99.0)
        )
        assert changed.volatile_digest != base.volatile_digest
        assert changed.run_fingerprint == base.run_fingerprint
        assert changed.environment_fingerprint == base.environment_fingerprint
        assert diff_sections(base, changed) == []


@pytest.mark.unit
class TestEnvironmentFingerprintScope:
    def test_environment_excludes_input_sections(self):
        sections = _sections()
        fp = compute_run_fingerprint(sections, _volatile())
        # Mutating input sections must leave environment fingerprint intact.
        env_only = compute_run_fingerprint(
            _sections(model=ModelSection(model_hash="f" * 64)),
            _volatile(),
        )
        assert env_only.environment_fingerprint == fp.environment_fingerprint


@pytest.mark.unit
class TestVerify:
    def test_verify_passes_for_consistent_fingerprint(self):
        fp = compute_run_fingerprint(_sections(), _volatile())
        fp.verify()  # must not raise

    def test_verify_detects_drift(self):
        fp = compute_run_fingerprint(_sections(), _volatile())
        fp.section_digests["os"] = "0" * 64
        with pytest.raises(ValueError):
            fp.verify()


@pytest.mark.unit
class TestCollectors:
    def test_collect_packages_reports_missing(self):
        packages = collect_packages(watchlist=["definitely_not_a_real_pkg_xyz"])
        assert packages["definitely_not_a_real_pkg_xyz"] == "<missing>"

    def test_environment_section_names(self):
        assert ENVIRONMENT_SECTION_NAMES == ("os", "device", "python", "packages", "power")
