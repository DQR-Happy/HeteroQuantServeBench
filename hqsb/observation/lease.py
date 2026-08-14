"""Cross-process model ownership; a stale living holder is never assumed idle."""

import json
import os
from pathlib import Path

from hqsb.observation.procfs import process_key


class DeviceLease:
    def __init__(self, path):
        self.path, self.fd = Path(path), None

    def acquire(self):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raw = os.read(fd, 8192)
            if raw:
                previous = json.loads(raw)
                if previous.get("state") == "active":
                    try:
                        live = process_key(previous["process"]["pid"])
                    except FileNotFoundError:
                        live = None
                    if live == previous["process"]:
                        raise RuntimeError("DEVICE_LEASE_CLEANUP_UNKNOWN")
            body = json.dumps({"state": "active", "process": process_key()}).encode()
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, body)
            os.fsync(fd)
            self.fd = fd
        except Exception:
            os.close(fd)
            raise

    def release(self):
        if self.fd is not None:
            os.lseek(self.fd, 0, os.SEEK_SET)
            os.ftruncate(self.fd, 0)
            os.write(self.fd, b'{"state":"closed"}')
            os.fsync(self.fd)
            os.close(self.fd)
            self.fd = None
