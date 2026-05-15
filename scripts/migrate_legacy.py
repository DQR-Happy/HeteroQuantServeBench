#!/usr/bin/env python3
"""Migrate a legacy benchmark document to the current schema.

Converts a legacy golden or legacy result JSON file into a
:class:`BenchmarkResult` document conforming to the C6 contract.

Migration follows the E01-07 contract:
* source version is identified and gated (future/older versions rejected);
* field-level losses are attached to the C6 summary for auditability;
* writes are transactional (temp file + atomic rename) so an interrupted
  write never leaves a half-written migrated file.

Usage:
    python scripts/migrate_legacy.py <input.json> <output.json>
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

from hqsb.core.errors import HqsbError
from hqsb.core.schema.migrate import migrate_any


def _write_atomic(text: str, output_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".hqsb_migrate_", suffix=".tmp", dir=out_dir
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_path, output_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def main() -> int:
    """Migrate the input document and write the result."""
    if len(sys.argv) != 3:
        print("Usage: python scripts/migrate_legacy.py <input.json> <output.json>",
              file=sys.stderr)
        return 2

    input_path, output_path = sys.argv[1], sys.argv[2]

    try:
        with open(input_path, encoding="utf-8") as fh:
            document = json.load(fh)
        result = migrate_any(document)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except HqsbError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON: {exc}", file=sys.stderr)
        return 1

    _write_atomic(result.model_dump_json(indent=2), output_path)

    migration = result.summary.get("migration", {})
    print(f"Migrated {input_path} -> {output_path}")
    print(f"  run_id: {result.run_id}")
    print(f"  source_family: {migration.get('source_family')}")
    print(f"  source_version: {migration.get('source_version')} -> "
          f"target {migration.get('target_schema')} v{migration.get('target_version')}")
    print(f"  loss_summary: {migration.get('loss_summary')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
