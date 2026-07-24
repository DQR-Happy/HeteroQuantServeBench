"""Run with the project's remote execution proxy; never on the editing Mac."""

from __future__ import annotations

import argparse
import fcntl
import os
import secrets
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="HQSB Console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    from hqsb.console.config import load_settings
    from hqsb.console.app import create_app
    import uvicorn

    settings = load_settings(args.config)
    settings.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(settings.data_dir, 0o700)
    # A second API using this store must not create another model worker.
    lock_file = (settings.data_dir / "service.lock").open("a")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another Console process owns this data directory") from exc
    token_file = settings.data_dir / "access-token"
    if not token_file.exists():
        fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(secrets.token_urlsafe(32))
    token = token_file.read_text().strip()
    os.chmod(token_file, 0o600)
    if len(token) < 24:
        raise ValueError("Console access token must contain at least 24 characters")
    print(
        f"HQSB Console: http://{args.host}:{args.port}; access token file: {token_file}",
        flush=True,
    )
    uvicorn.run(create_app(settings, token), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
