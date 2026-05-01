#!/usr/bin/env python3
"""Migrate a legacy benchmark document to the current schema.

Converts a legacy golden or legacy result JSON file into a
:class:`BenchmarkResult` document conforming to the C6 contract.

Usage:
    python scripts/migrate_legacy.py <input.json> <output.json>
"""

from __future__ import annotations

import json
import sys

from hqsb.core.errors import HqsbError
from hqsb.core.schema.migrate import migrate_any


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

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(result.model_dump_json(indent=2))

    print(f"Migrated {input_path} -> {output_path}")
    print(f"  run_id: {result.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
