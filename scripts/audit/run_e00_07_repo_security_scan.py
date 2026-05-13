#!/usr/bin/env python3
"""E00-07 — repository public-boundary security scan.

Question
--------
On a frozen commit, does the *tracked* file set (the content distributed with the
repository) contain secrets/credentials, personal or machine-specific absolute
paths, large model weights, build artifacts, or raw reports that were tracked by
mistake? Are the .gitignore rules consistent with that boundary, and can an
automated tool actually catch these problems before publish?

Pre-registered design (docs/stage_experiments/details/S00/E00-07_*.md)
----------------------------------------------------------------------
* Scan scope : files listed by `git ls-files` (tracked), content read from the
               working tree (== next commit to be published; HEAD is frozen for
               Git facts).
* Categories : SECRET / MACHINE_PATH / WEIGHTS / BUILD_ARTIFACT / RAW_TRACKED
               plus an ignore-rule audit (tracked-despite-ignored, high-risk
               extensions) and an untracked reference summary.
* Rules      : pre-registered regex/ext rules written to raw/scan_rules.json
               (see module RULES). Text rules scan text files line by line;
               binary files only go through extension/size rules.
* Two-phase  : `--phase pre`  runs before any in-experiment disposition and
               writes raw/hits_pre.jsonl;
               `--phase post` runs after disposition, writes raw/hits_post.jsonl,
               and produces the verdict by cross-checking dispositions.json.
* Tool self-test: five fixtures in a throwaway git repo (private key block,
               /home/<user> path, a >50 MiB weight, a build artifact, and a
               benign source file) must be detected / not mis-fired.

Raw output (docs/stage_experiments/S00/E00-07/raw/)
----------------------------------------------------
environment.json, git_facts.json, scan_rules.json, gitignore_rules.json,
ignore_audit.json, untracked_summary.json, hits_pre.jsonl, hits_post.jsonl,
hits_summary.json, dispositions.json, self_test.json, verdict.json,
run_records/*.log, command.txt, file_manifest.json and the aggregated run record
e00_07_run_run_<ts>.json.

Usage
-----
    python3 scripts/audit/run_e00_07_repo_security_scan.py \
        --output-dir docs/stage_experiments/S00/E00-07/raw --phase pre
    # ... apply dispositions (raw/dispositions.json) and source fixes ...
    python3 scripts/audit/run_e00_07_repo_security_scan.py \
        --output-dir docs/stage_experiments/S00/E00-07/raw --phase post
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_ID = "E00-07"
STAGE = "S00"

# ── constants ─────────────────────────────────────────────────────────────────

WEIGHT_EXTS = {
    ".safetensors", ".gguf", ".pt", ".pth", ".ckpt", ".onnx", ".engine",
    ".plan", ".bin", ".h5", ".hdf5", ".npz",
}
BUILD_EXTS = {
    ".o", ".a", ".so", ".dll", ".dylib", ".whl", ".pyc", ".pyo", ".class",
    ".jar", ".so.1", ".so.2",
}
RAW_EXTS = {".log", ".stdout", ".stderr", ".jsonl", ".ncu-rep", ".nsys-rep", ".qdrep"}
BUILD_DIR_MARKS = ("build/", "cmake-build-", "out/", "dist/", "__pycache__/")

BIG_FILE_BYTES = 50 * 1024 * 1024  # 50 MiB

# ── rule set (pre-registered) ─────────────────────────────────────────────────

# Each text rule returns (rule_id, category, severity, match_text) per line, or None.
# A rule may yield multiple matches per line (kept simple: first match per rule/line).
TEXT_RULES: List[Dict[str, Any]] = [
    {
        "id": "SECRET-PRIVKEY",
        "category": "SECRET",
        "severity": "high",
        "regex": re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP |ENCRYPTED )?PRIVATE KEY-----",
            re.IGNORECASE,
        ),
        "reason": "private-key block present in a tracked text file",
    },
    {
        "id": "SECRET-TOKPREFIX",
        "category": "SECRET",
        "severity": "high",
        "regex": re.compile(
            r"(?i)\b(?:AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|github_pat_"
            r"[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|xox[baprs]-[0-9A-Za-z-]{10,}"
            r"|AIza[0-9A-Za-z_\-]{35})\b",
        ),
        "reason": "known credential/API-key prefix",
    },
    {
        "id": "SECRET-KEYASSIGN",
        "category": "SECRET",
        "severity": "medium",
        "regex": re.compile(
            r"(?i)\b(?:api[_-]?key|apikey|access[_-]?key|auth[_-]?token|"
            r"client[_-]?secret|secret|password|passwd|token)\b\s*[:=]\s*"
            r"[\"']?([A-Za-z0-9_\-/+=]{16,})[\"']?",
        ),
        "reason": "generic key/password assignment with a long, token-like value "
                  "(needs human review)",
    },
    {
        "id": "PATH-HOME",
        "category": "MACHINE_PATH",
        "severity": "high",
        "regex": re.compile(
            # branch 1: file:///home/<user>/... and file:///Users/<user>/...
            r"file:///(?:home|Users)/[A-Za-z0-9_.-]+(?:/[^\s\"',;)\]}]*)?"
            # branch 2: /home/<user>/... preceded by a boundary char (start/space/quote/=/:/(/[/,)
            r"|(?:^|[\s\"'=:([])/(?:home|Users)/"
            r"[A-Za-z0-9_.-]+(?:/[^\s\"',;)\]}]*)?",
        ),
        "reason": "machine-specific absolute user path in a tracked file",
    },
    {
        "id": "PATH-ROOT",
        "category": "MACHINE_PATH",
        "severity": "medium",
        "regex": re.compile(r"(?:^|[\s\"'=:([])/root/[A-Za-z0-9_.-]+(?:/[^\s\"',;)\]}]*)?"),
        "reason": "root user path reference (review context)",
    },
]


def text_hits(rule: Dict[str, Any], rel_path: str, text: str) -> List[Dict[str, Any]]:
    rx = rule["regex"]
    hits: List[Dict[str, Any]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        m = rx.search(raw)
        if not m:
            continue
        match_text = m.group(0)
        hits.append({
            "rule": rule["id"],
            "category": rule["category"],
            "severity": rule["severity"],
            "line": lineno,
            "match": match_text[:400],
            "match_sha256": hashlib.sha256(match_text.encode("utf-8", "replace")).hexdigest(),
            "excerpt": raw.strip()[:240],
            "reason": rule["reason"],
        })
    return hits


# ── tiny helpers ──────────────────────────────────────────────────────────────


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], capture_output=True, cwd=str(repo_root),
    )


def _git(repo_root: Path, *args: str) -> str:
    proc = _run_git(repo_root, *args)
    return proc.stdout.decode("utf-8", "replace").strip() if proc.returncode == 0 else ""


def _git_ok(repo_root: Path, *args: str) -> bool:
    return _run_git(repo_root, *args).returncode == 0


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _human_size(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


# ── git / environment facts ───────────────────────────────────────────────────


def collect_git_facts(repo_root: Path) -> Dict[str, Any]:
    head_full = _git(repo_root, "rev-parse", "HEAD")
    dirty = bool(_git(repo_root, "status", "--porcelain"))
    tracked = _git(repo_root, "ls-files", "-z")
    tracked_files = [f for f in tracked.split("\0") if f]
    return {
        "head_full": head_full,
        "head_short": _git(repo_root, "rev-parse", "--short", "HEAD"),
        "branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": dirty,
        "head_subject": _git(repo_root, "log", "-1", "--format=%s"),
        "tracked_file_count": len(tracked_files),
        "commits": _git(repo_root, "rev-list", "--all").split(),
    }


def collect_environment_block() -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "experiment_id": EXPERIMENT_ID,
        "git_commit": _git(_REPO_ROOT, "rev-parse", "HEAD"),
        "git_commit_short": _git(_REPO_ROOT, "rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git(_REPO_ROOT, "status", "--porcelain")),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "started_at_utc": _now_utc(),
    }


def is_binary(path: Path, head: bytes) -> bool:
    if b"\0" in head:
        return True
    return Path(path).suffix.lower() in (WEIGHT_EXTS | BUILD_EXTS)


def ext_of(rel: str) -> str:
    dot = rel.rfind(".")
    if dot < 0:
        return ""
    return rel[dot:].lower()


# ── scan ──────────────────────────────────────────────────────────────────────


def _hit(rule: str, category: str, severity: str, rel: str, line: int,
         reason: str, size: int, sha256: str, excerpt: str) -> Dict[str, Any]:
    return {
        "rule": rule,
        "category": category,
        "severity": severity,
        "file": rel,
        "line": line,
        "match": "",
        "match_sha256": "",
        "excerpt": excerpt,
        "reason": reason,
        "size_bytes": size,
        "file_sha256": sha256,
        "verdict": "pending",
        "phase": "",
    }


def scan_tracked(repo_root: Path, tracked_files: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Scan tracked files. Returns (hits, file_stats)."""
    hits: List[Dict[str, Any]] = []
    sizes: Dict[str, int] = {}
    per_file_rules: Dict[str, int] = {}

    for rel in tracked_files:
        full = repo_root / rel
        if not full.is_file():
            continue
        try:
            data = full.read_bytes()
        except OSError:
            continue
        size = len(data)
        sizes[rel] = size
        sha256 = _sha256_bytes(data)
        rules_here: Set[str] = set()
        ext = ext_of(rel)
        low_rel = rel.lower()

        # extension / path rules apply to any file (binary included)
        if ext in WEIGHT_EXTS:
            hits.append(_hit("WEIGHT-EXT", "WEIGHTS", "high", rel, 0,
                             f"tracked model-weight extension: {ext}", size, sha256, f"{size}"))
            rules_here.add("WEIGHT-EXT")
        if size >= BIG_FILE_BYTES:
            hits.append(_hit("WEIGHT-SIZE", "WEIGHTS", "high", rel, 0,
                             f"tracked file >= 50 MiB: {_human_size(size)}", size, sha256, f"{size}"))
            rules_here.add("WEIGHT-SIZE")
        if ext in BUILD_EXTS or ext.startswith(".so."):
            hits.append(_hit("BUILD-EXT", "BUILD_ARTIFACT", "high", rel, 0,
                             f"tracked build-artifact extension: {ext}", size, sha256, f"{size}"))
            rules_here.add("BUILD-EXT")
        if any(mark in low_rel for mark in BUILD_DIR_MARKS):
            hits.append(_hit("BUILD-DIR", "BUILD_ARTIFACT", "high", rel, 0,
                             "tracked file under a generated/build directory", size, sha256, f"{size}"))
            rules_here.add("BUILD-DIR")
        is_raw_data = ext in RAW_EXTS or (
            Path(rel).name != ".gitkeep"
            and any(c == "raw" for c in (p.lower() for p in Path(rel).parts[:-1]))
        )
        if is_raw_data:
            hits.append(_hit("RAW-EXT", "RAW_TRACKED", "medium", rel, 0,
                             f"tracked raw/log style data file (ext={ext}, in raw dir)",
                             size, sha256, f"{size}"))
            rules_here.add("RAW-EXT")

        # text rules only on text files
        binary = is_binary(full, data[:8192])
        if not binary and data:
            text = data.decode("utf-8", "replace")
            for rule in TEXT_RULES:
                for th in text_hits(rule, rel, text):
                    th["file"] = rel
                    th["size_bytes"] = size
                    th["file_sha256"] = sha256
                    th["verdict"] = "pending"
                    th["phase"] = ""
                    hits.append(th)
                    rules_here.add(rule["id"])

        per_file_rules[rel] = len(rules_here)

    return hits, {"sizes": sizes, "per_file_rule_count": per_file_rules}


# ── ignore-rule audit ─────────────────────────────────────────────────────────


def load_gitignore_rules(repo_root: Path) -> List[Dict[str, Any]]:
    gi = repo_root / ".gitignore"
    rules: List[Dict[str, Any]] = []
    if gi.is_file():
        for i, raw in enumerate(_read_text(gi).splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rules.append({"line": i, "pattern": line, "text": raw})
    return rules


def tracked_despite_ignored(repo_root: Path, tracked_files: List[str]) -> List[Dict[str, Any]]:
    """Run `git check-ignore -v -z --stdin` over tracked paths.

    Note: tracked files are never reported by check-ignore (the index overrides
    ignore rules); this is therefore expected to be empty and kept only to prove
    that statement. Static classification is done by
    tracked_under_ignored_dir_roots().
    """
    if not tracked_files:
        return []
    payload = "\0".join(tracked_files).encode() + b"\0"
    proc = subprocess.run(
        ["git", "check-ignore", "-v", "-z", "--stdin"],
        input=payload, capture_output=True, cwd=str(repo_root),
    )
    out = proc.stdout.decode("utf-8", "replace")
    result: List[Dict[str, Any]] = []
    for chunk in [c for c in out.split("\0") if c]:
        # format: <src>:<lineno>:<pattern>\t<path>
        if "\t" not in chunk:
            continue
        rule, path = chunk.split("\t", 1)
        result.append({"file": path, "rule": rule})
    return result


def tracked_under_ignored_dir_roots(repo_root: Path,
                                    tracked_files: List[str]) -> List[Dict[str, Any]]:
    """Static: which tracked files live under a directory that .gitignore ignores?

    Used to classify *historical* tracked content that was added before an ignore
    rule existed (e.g. docs/ tree) - these are not "leaks" but need to be audited.
    """
    anchors: List[Tuple[str, Dict[str, Any]]] = []
    for r in load_gitignore_rules(repo_root):
        p = r["pattern"]
        if not p.startswith("/"):
            continue
        pp = p[1:]
        if pp.endswith("/"):
            pp = pp[:-1]
        if pp and not any(ch in pp for ch in "*?[!"):
            anchors.append((pp, r))
    result: List[Dict[str, Any]] = []
    for prefix, rule in anchors:
        files = [f for f in tracked_files
                 if f == prefix or f.startswith(prefix + "/")]
        if files:
            result.append({
                "ignored_dir_root": prefix,
                "rule_line": rule["line"],
                "rule_text": rule["text"],
                "tracked_count": len(files),
                "tracked_files": files,
            })
    return result


def untracked_summary(repo_root: Path) -> Dict[str, Any]:
    untracked_visible = [f for f in _git(repo_root, "ls-files", "--others",
                                         "--exclude-standard", "-z").split("\0") if f]
    untracked_ignored = [f for f in _git(repo_root, "ls-files", "--others", "-i",
                                         "--exclude-standard", "-z").split("\0") if f]
    def count_by_ext(paths: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p in paths:
            e = ext_of(p)
            counts[e] = counts.get(e, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:15])

    def risk_ext(paths: List[str], exts: Set[str]) -> int:
        return sum(1 for p in paths if ext_of(p) in exts or any(
            m in p.lower() for m in ("raw/", "/raw", "run_2", ".stdout", ".stderr")))

    return {
        "untracked_not_ignored_count": len(untracked_visible),
        "untracked_not_ignored": untracked_visible[:200],
        "untracked_ignored_count": len(untracked_ignored),
        "untracked_ignored_ext_top": count_by_ext(untracked_ignored),
        "untracked_not_ignored_risk_like": risk_ext(untracked_visible, RAW_EXTS),
    }


# ── fixtures self-test ────────────────────────────────────────────────────────


def _write(repo: Path, rel: str, data: bytes) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def run_self_test() -> Dict[str, Any]:
    """Build a throwaway repo with five fixtures and verify detector behaviour."""
    results: Dict[str, Any] = {}
    tmp = Path(tempfile.mkdtemp(prefix="hqsb_e0007_selftest_"))
    try:
        subprocess.run(["git", "init", "-q"], cwd=str(tmp), check=True)

        # fixture A: private key block
        _write(tmp, "keys/sample.pem",
               b"-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAfake\n-----END RSA PRIVATE KEY-----\n")
        # fixture B: machine path in config
        _write(tmp, "configs/model.yaml",
               b"model:\n  local_path: /home/alice/models/Qwen\n")
        # fixture C: >50 MiB weight
        _write(tmp, "weights/w.bin", b"\0" * (BIG_FILE_BYTES + 1))
        # fixture D: build artifact
        _write(tmp, "build/lib.o", b"\x7fELF not real but binary-ish\n")
        # fixture E: benign source with look-alike keywords
        _write(tmp, "app/main.py", (
            b"import os\n"
            b"# password=for the server (short, comment only)\n"
            b"tokenizer = None  # AutoTokenizer placeholder\n"
            b"local_path = \"~/models/hqsb/Qwen3-1.7B\"\n"
            b"# onnxruntime==1.22.1\n"
            b"parser = None  # --password arg lives here\n"
        ))
        subprocess.run(["git", "-c", "user.name=hqsb-selftest",
                        "-c", "user.email=hqsb-selftest@example.invalid",
                        "add", "-A"], cwd=str(tmp), check=True)
        subprocess.run(["git", "-c", "user.name=hqsb-selftest",
                        "-c", "user.email=hqsb-selftest@example.invalid",
                        "commit", "-q", "-m", "fixtures"], cwd=str(tmp), check=True)

        tracked = [f for f in _git(tmp, "ls-files", "-z").split("\0") if f]
        hits, _ = scan_tracked(tmp, tracked)

        def has(rule: str) -> bool:
            return any(h["rule"] == rule for h in hits)

        results["fixture_A_secret_pem"] = {
            "detected": has("SECRET-PRIVKEY"), "expect": True,
        }
        results["fixture_B_machine_path"] = {
            "detected": has("PATH-HOME"), "expect": True,
        }
        results["fixture_C_weight_size"] = {
            "detected": has("WEIGHT-SIZE"), "expect": True,
        }
        results["fixture_D_build_artifact"] = {
            "detected": has("BUILD-DIR"), "expect": True,
        }
        benign_high = [h for h in hits if h["file"].endswith("main.py")
                       and h["severity"] == "high"]
        results["fixture_E_benign_no_high"] = {
            "detected": len(benign_high) == 0, "expect": True,
            "high_hits": [h["rule"] for h in benign_high],
        }
        results["all_pass"] = all(
            v["detected"] == v["expect"] for v in results.values() if isinstance(v, dict)
        )
        results["fixtures"] = sorted({h["file"] for h in hits})
        results["total_hits"] = len(hits)
        return results
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── output helpers ────────────────────────────────────────────────────────────


def _jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def _dump_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)


def save_manifest(out_dir: Path) -> None:
    files = sorted(p.name for p in out_dir.rglob("*") if p.is_file())
    _dump_json(out_dir / "file_manifest.json", {
        "count": len(files),
        "files": files,
        "hashes": {
            f.relative_to(out_dir).as_posix(): _sha256_bytes(f.read_bytes())
            for f in out_dir.rglob("*") if f.is_file()
        },
    })


def build_verdict(out_dir: Path, post_hits: List[Dict[str, Any]],
                  pre_hits: List[Dict[str, Any]], self_test: Dict[str, Any]) -> Dict[str, Any]:
    disp_path = out_dir / "dispositions.json"
    dispositions: Dict[str, Any] = {}
    if disp_path.is_file():
        raw = json.loads(disp_path.read_text(encoding="utf-8"))
        dispositions = raw if isinstance(raw, dict) else {"items": raw}

    def key(h: Dict[str, Any]) -> Tuple[str, str, int]:
        return (h["rule"], h["file"], h.get("line", 0))

    candidate_keys: Set[Tuple[str, str, int]] = set()
    for h in pre_hits + post_hits:
        if h["severity"] in ("medium", "high"):
            candidate_keys.add(key(h))

    items = dispositions.get("items", [])
    decided: Set[Tuple[str, str, int]] = set()
    for it in items:
        decided.add((it.get("rule", ""), it.get("file", ""), int(it.get("line", 0))))
    pending = sorted(candidate_keys - decided)

    high_post = [h for h in post_hits if h["severity"] == "high"]
    risky_post = [h for h in high_post if h["category"] in ("SECRET", "WEIGHTS", "MACHINE_PATH")]

    c1 = len(risky_post) == 0
    c2 = len(pending) == 0
    c3 = bool(self_test.get("all_pass"))
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"run_{int(time.time() * 1000)}",
        "head": _git(_REPO_ROOT, "rev-parse", "HEAD"),
        "criterion_1_no_real_secret_weight_machine_path_tracked": {
            "pass": c1,
            "high_risky_post_count": len(risky_post),
            "details": risky_post,
        },
        "criterion_2_all_false_positives_audited": {
            "pass": c2,
            "pending_candidates": sorted([list(k) for k in pending]),
            "candidate_count": len(candidate_keys),
        },
        "criterion_3_tool_self_test": {
            "pass": c3,
            "self_test": self_test,
        },
        "overall": "PASS" if (c1 and c2 and c3) else "FAIL",
        "generated_at_utc": _now_utc(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str,
                        default="docs/stage_experiments/S00/E00-07/raw")
    parser.add_argument("--phase", choices=["pre", "post"], default="pre")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()

    out_dir = (_REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "run_records"
    run_dir.mkdir(parents=True, exist_ok=True)

    # always dump environment + git facts + rules
    _dump_json(out_dir / "environment.json", collect_environment_block())
    _dump_json(out_dir / "git_facts.json", collect_git_facts(_REPO_ROOT))
    _dump_json(out_dir / "scan_rules.json", {
        "text_rules": [{"id": r["id"], "category": r["category"],
                        "severity": r["severity"], "pattern": r["regex"].pattern,
                        "reason": r["reason"]} for r in TEXT_RULES],
        "weight_exts": sorted(WEIGHT_EXTS),
        "build_exts": sorted(BUILD_EXTS),
        "raw_exts": sorted(RAW_EXTS),
        "big_file_bytes": BIG_FILE_BYTES,
    })

    self_test = run_self_test()
    _dump_json(out_dir / "self_test.json", self_test)

    if args.self_test_only:
        print(f"self-test: all_pass={self_test.get('all_pass')}")
        return 0

    # ignore-rule audit (same for both phases)
    if args.phase == "pre":
        gi_rules = load_gitignore_rules(_REPO_ROOT)
        _dump_json(out_dir / "gitignore_rules.json", gi_rules)
        tracked = [f for f in _git(_REPO_ROOT, "ls-files", "-z").split("\0") if f]
        tdi = tracked_despite_ignored(_REPO_ROOT, tracked)
        under_ignored = tracked_under_ignored_dir_roots(_REPO_ROOT, tracked)
        un_sum = untracked_summary(_REPO_ROOT)
        _dump_json(out_dir / "untracked_summary.json", un_sum)
        _dump_json(out_dir / "ignore_audit.json", {
            "tracked_file_count": len(tracked),
            "check_ignore_note": ("git check-ignore never reports tracked files "
                                  "(index overrides ignore rules); empty result "
                                  "is expected and proves no rule silently removes "
                                  "tracked content."),
            "tracked_despite_ignored": tdi,
            "tracked_despite_ignored_count": len(tdi),
            "tracked_under_ignored_dir_roots": under_ignored,
            "tracked_under_ignored_dir_roots_count": len(under_ignored),
            "notes": ("files under ignored dir roots are historical tracked content "
                      "added before the ignore rule; each is classified in the "
                      "report (keep-exempt historical docs) or disposed."),
        })

    # scan
    tracked = [f for f in _git(_REPO_ROOT, "ls-files", "-z").split("\0") if f]
    hits, stats = scan_tracked(_REPO_ROOT, tracked)
    for h in hits:
        h["phase"] = args.phase
    hits_path = out_dir / f"hits_{args.phase}.jsonl"
    _jsonl(hits_path, hits)

    summary = {
        "phase": args.phase,
        "tracked_file_count": len(tracked),
        "hit_count": len(hits),
        "by_severity": {
            s: len([h for h in hits if h["severity"] == s]) for s in ("high", "medium", "low")
        },
        "by_category": {c: len([h for h in hits if h["category"] == c])
                        for c in sorted({h["category"] for h in hits})},
        "by_rule": {r: len([h for h in hits if h["rule"] == r])
                    for r in sorted({h["rule"] for h in hits})},
        "largest_tracked": sorted(
            ((sz, p) for p, sz in stats["sizes"].items()),
            reverse=True)[:10],
    }
    _dump_json(out_dir / "hits_summary.json", summary)

    # candidate template at pre-phase (for the auditor to fill in)
    if args.phase == "pre":
        candidates = [h for h in hits if h["severity"] in ("medium", "high")]
        templ = out_dir / "dispositions.template.json"
        _dump_json(templ, {
            "note": ("One entry per medium/high candidate. decision in "
                     "{false_positive, real-fixed, real-removed, keep-exempt}; "
                     "copy to dispositions.json and fill reason."),
            "items": [{"rule": h["rule"], "file": h["file"], "line": h.get("line", 0),
                       "severity": h["severity"], "decision": "pending",
                       "reason": ""} for h in sorted(
                candidates, key=lambda x: (x["rule"], x["file"], x.get("line", 0)))],
        })

    # post-phase: verdict + run record
    if args.phase == "post":
        pre_path = out_dir / "hits_pre.jsonl"
        pre_hits: List[Dict[str, Any]] = []
        if pre_path.is_file():
            pre_hits = [json.loads(line) for line in
                        pre_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        verdict = build_verdict(out_dir, hits, pre_hits, self_test)
        _dump_json(out_dir / "verdict.json", verdict)
        run_id = f"run_{int(time.time() * 1000)}"
        record = {
            "experiment_id": EXPERIMENT_ID,
            "run_id": run_id,
            "phase": args.phase,
            "git_commit": _git(_REPO_ROOT, "rev-parse", "HEAD"),
            "git_dirty": bool(_git(_REPO_ROOT, "status", "--porcelain")),
            "summary": summary,
            "verdict": verdict,
        }
        _dump_json(out_dir / f"e00_07_run_{run_id}.json", record)

    # command.txt + file manifest + stdout log
    cmd = "python3 scripts/audit/run_e00_07_repo_security_scan.py " \
          f"--output-dir {args.output_dir} --phase {args.phase}"
    (out_dir / "command.txt").write_text(cmd + "\n", encoding="utf-8")
    log_path = run_dir / f"e00_07_{args.phase}_{_now_utc().replace(':', '-')}.log"
    log_path.write_text(
        json.dumps({"command": cmd, "phase": args.phase,
                    "summary": summary, "self_test": self_test},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_manifest(out_dir)

    print(json.dumps({"phase": args.phase, "hits": len(hits), **summary.get("by_severity", {})},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
