"""Export the actual API contract without a device or a running web server."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hqsb.console.app import create_app
from hqsb.console.config import load_settings


def main():
    with TemporaryDirectory() as temporary:
        settings = load_settings()
        settings.data_dir = Path(temporary)
        app = create_app(settings, "schema-export-no-listening-socket", monitor=False)
        destination = Path("contracts/console/openapi.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n"
        )
        app.state.service.close()
        print(destination)


if __name__ == "__main__":
    main()
