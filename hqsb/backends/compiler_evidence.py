"""Owns temporary Inductor debug output and bounded publication of real files."""
from pathlib import Path

from hqsb.backends.graph_formats import ir_graph

FILES = {"inductor_pre": "ir_pre_fusion.txt", "inductor_post": "ir_post_fusion.txt",
         "generated_code": "output_code.py"}


def collect(directory, target, graph_id, phase, first_run_id):
    entries = []
    for kind, filename in FILES.items():
        paths = sorted(Path(directory).rglob(filename))[:4]
        if not paths:
            entries.append({"kind": kind, "id": f"{graph_id}-{kind}", "status": "not_produced",
                            "graph_id": graph_id, "first_run_id": first_run_id,
                            "reason": "compiler did not emit this diagnostic; no reconstruction from FX"})
        for index, path in enumerate(paths):
            if path.is_symlink() or not path.is_file():
                continue
            # Publish only a bounded text prefix, never tensor data/debug archives.
            with path.open("rb") as stream:
                raw = stream.read(1024**2 + 1)
            clipped = len(raw) > 1024**2
            raw = raw[:1024**2]
            text = raw.decode("utf-8", errors="replace")
            name = f"graph-debug-{graph_id}-{index}-{filename}"
            (Path(target) / name).write_text(text)
            entry = {"kind": kind, "id": f"{graph_id}-{index}-{kind}", "graph_id": graph_id,
                     "phase": phase, "status": "available", "first_run_id": first_run_id,
                     "artifact": name, "text": text[:48000], "text_truncated": len(text) > 48000 or clipped,
                     "artifact_truncated": clipped, "scope": "actual compiler diagnostic; not executed kernel trace"}
            if kind != "generated_code":
                entry.update(ir_graph(text))
            entries.append(entry)
    return entries
