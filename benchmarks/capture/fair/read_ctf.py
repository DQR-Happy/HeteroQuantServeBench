"""Run using /usr/bin/python3 with the original Babeltrace2 package."""
import json
import sys

import bt2

for message in bt2.TraceCollectionMessageIterator(sys.argv[1]):
    if isinstance(message, bt2._EventMessageConst) and message.event.name == "hqsb_capture:event":
        p = message.event.payload_field
        print(json.dumps({"run_id": str(p["run_id"]), "producer_id": int(p["producer_id"]),
                          "source_seq": int(p["source_seq"]), "source_timestamp": int(p["source_timestamp"]),
                          "body": str(p["body"])}, ensure_ascii=False))
