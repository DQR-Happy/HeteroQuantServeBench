import json

import pytest

from hqsb.observation.lease import DeviceLease
from hqsb.observation.procfs import process_key


def test_mutual_exclusion_and_release(tmp_path):
    path = tmp_path / "device.lock"
    first, second = DeviceLease(path), DeviceLease(path)
    first.acquire()
    try:
        with pytest.raises(BlockingIOError):
            second.acquire()
        assert second.fd is None
    finally:
        first.release()
    second.acquire()
    second.release()
    second.release()


def test_unlocked_but_still_living_holder_requires_cleanup(tmp_path):
    path = tmp_path / "device.lock"
    path.write_text(json.dumps({"state": "active", "process": process_key()}))
    with pytest.raises(RuntimeError, match="CLEANUP_UNKNOWN"):
        DeviceLease(path).acquire()


def test_pid_reuse_is_not_old_holder(tmp_path):
    path = tmp_path / "device.lock"
    key = process_key()
    key["start_ticks"] -= 1
    path.write_text(json.dumps({"state": "active", "process": key}))
    lease = DeviceLease(path)
    lease.acquire()
    lease.release()
