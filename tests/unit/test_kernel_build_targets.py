"""Build metadata must describe the actual selected target, without GPU imports."""
import pytest

from scripts.observation.build_model_kernels import architectures


def test_native_target_and_explicit_cross_build():
    assert architectures(None, (8, 7)) == [87]
    assert architectures(None, (8, 9)) == [89]
    assert architectures("89;87;89") == [87, 89]
    with pytest.raises(ValueError, match="REQUIRES_CUDA"):
        architectures(None)


@pytest.mark.parametrize("value", ["native", "89-real", "", "89;bad", "0", "89;", "89 -DOTHER=1"])
def test_invalid_targets_fail_before_cmake(value):
    with pytest.raises(ValueError, match="ARCHITECTURES_INVALID"):
        architectures(value)
