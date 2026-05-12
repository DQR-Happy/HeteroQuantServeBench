#!/usr/bin/env python3
"""E00-06 — README / Project Status / Evidence Ledger claim-state-Git tree drift audit.

Question
--------
On a frozen commit, can every "done / measured / planned" claim in README.md,
docs/project_status.md and docs/evidence_ledger.md be explained by evidence
(code / tests / documented local runtime reports) in the current Git tree or on
this machine? Are the commit references in those documents real Git objects that
are consistent with the tree? Does an automated tool actually catch "docs say
done but the repository does not contain it"?

Pre-registered design (per docs/stage_experiments/details/S00/E00-06_*.md)
-------------------------------------------------------------------------
* Audit scope  : the three documents listed above (as tracked by git).
* Claim parse  : markdown tables; semantic columns mapped by header keywords
                 (# / claim text / label / evidence path / location path).
* Fact labels  : PLANNED / SOURCE / TEST / RUNTIME / MODEL / SERVICE / PORTABLE
                 mapped from document wording (planned, source-only,
                 test-verified, runtime-verified, Verified, Implemented, …).
* Evidence     : every evidence/location cell is tokenised (backticked code
                 spans, markdown relative links, slash separated tokens, globs)
                 and resolved against (a) the workspace and (b) the Git tree.
                 Markdown-link tokens resolve relative to the document, plain
                 code-span tokens resolve relative to the repository root (the
                 convention used by the audited docs). Classification:
                 tracked / local_ignored / workspace_only / missing /
                 outside_repo / non_path.
* Commit audit : every hex reference (7-40 chars) in the three docs is checked
                 against the real commit history; doc header "基线 Commit" is
                 compared with HEAD and with the last commit that touched the doc.
* Tool self-test: fixture claims (isolated runtime claim, fabricated commit,
                 planned claim with no evidence) must be detected correctly.

Raw output (docs/stage_experiments/S00/E00-06/raw/)
----------------------------------------------------
environment.json, git_facts.json, claim_records.jsonl, evidence_resolution.jsonl,
commit_audit.json, numeric_claims.json, drift_diff.json, self_test.json,
verdict.json, run_records/*.log, command.txt, file_manifest.json and the
aggregated run record e00_06_run_<id>.json.

Usage
-----
    python3 scripts/audit/run_e00_06_doc_claim_audit.py \
        --output-dir docs/stage_experiments/S00/E00-06/raw
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache",
                 ".pytest_cache", ".ruff_cache", "build", "third_party"}

EXPERIMENT_ID = "E00-06"
STAGE = "S00"

# The three audited documents (repo-root relative).
AUDIT_DOCS = [
    "README.md",
    "docs/project_status.md",
    "docs/evidence_ledger.md",
]

# ── tiny helpers ─────────────────────────────────────────────────────────────


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=str(_REPO_ROOT)
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except OSError:
        return ""


def _git_ok(*args: str) -> bool:
    try:
        proc = subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=str(_REPO_ROOT)
        )
        return proc.returncode == 0
    except OSError:
        return False


# ── git facts ────────────────────────────────────────────────────────────────


def collect_git_facts() -> Dict[str, Any]:
    head_full = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    tracked = _git("ls-files", "-z")
    tracked_files = [f for f in tracked.split("\0") if f]
    return {
        "head_full": head_full,
        "head_short": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": dirty,
        "head_subject": _git("log", "-1", "--format=%s"),
        "tracked_file_count": len(tracked_files),
        "tracked_files": tracked_files,
        "commits": _git("rev-list", "--all").split(),
        "exclude_dirs": sorted(_EXCLUDE_DIRS),
    }


def last_commit_for(path: str) -> str:
    return _git("log", "-1", "--format=%H", "--", path)


def check_ignore(path: str) -> Tuple[bool, str]:
    proc = subprocess.run(
        ["git", "check-ignore", "-v", "--", path],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    if proc.returncode != 0:
        return False, ""
    parts = proc.stdout.strip().split("\t", 1)
    rule = parts[0] if parts else proc.stdout.strip()
    return True, rule


def is_commit_object(full_or_short: str, commits: List[str]) -> bool:
    if full_or_short in commits:
        return True
    return any(c.startswith(full_or_short) for c in commits)


def commit_is_ancestor_of_head(candidate: str) -> bool:
    return _git_ok("merge-base", "--is-ancestor", candidate, "HEAD")


def collect_environment_block() -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "experiment_id": EXPERIMENT_ID,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_commit_short": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "started_at_utc": _now_utc(),
    }


# ── markdown table / claim parsing ───────────────────────────────────────────

_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")
_CODE_SPAN_RE = re.compile(r"`([^`]+)`")
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
# token like "4/6", "2026/5" -> ratios/dates, not file paths
_RATIO_RE = re.compile(r"^\d+(?:[.,]\d+)?/\d+$")
# extract a section-base directory from headings like "3.1 `hqsb/`（Python 包）"
_SECTION_BASE_RE = re.compile(r"`([^`]+/)`")


def _split_row(line: str) -> List[str]:
    line = line.strip()
    if not line.startswith("|"):
        return []
    line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def parse_markdown_tables(text: str) -> List[Dict[str, Any]]:
    """Return tables with {section, headers, rows:[dict]}."""
    lines = text.splitlines()
    tables: List[Dict[str, Any]] = []
    i = 0
    n = len(lines)
    current_section = ""
    while i < n:
        m = _HEADING_RE.match(lines[i].strip())
        if m:
            current_section = m.group(1).strip()
        if lines[i].strip().startswith("|"):
            block: List[List[str]] = []
            while i < n and lines[i].strip().startswith("|"):
                row = _split_row(lines[i])
                if row and not all(
                    re.fullmatch(r":?-{2,}:?", c or "-") for c in row
                ):
                    block.append(row)
                i += 1
            if len(block) >= 2:
                headers = block[0]
                rows = [dict(zip(headers, r)) for r in block[1:]]
                tables.append({
                    "section": current_section,
                    "headers": headers,
                    "rows": rows,
                })
            continue
        i += 1
    return tables


def _normalise_header(h: str) -> str:
    return h.lower().replace(" ", "").replace("#", "")


def classify_column(headers: List[str]) -> Dict[str, str]:
    """Map semantic roles to header indexes."""
    roles: Dict[str, str] = {}
    claim_cols: List[str] = []
    for h in headers:
        hh = _normalise_header(h)
        if h.strip() == "#" or hh == "编号" or hh == "id":
            roles.setdefault("id", h)
        elif "分级" in hh or hh == "状态" or hh == "status":
            roles.setdefault("label", h)
        elif "证据路径" in hh or "证据" in hh or "evidence" in hh:
            roles.setdefault("evidence", h)
        elif hh in ("位置", "路径", "location", "文件") or "位置" in hh:
            roles.setdefault("location", h)
        elif "声明" in hh or "能力" in hh or "内容" in hh or "名称" in hh \
                or "判定项" in hh or "阶段" in hh or "模块" in hh:
            claim_cols.append(h)
    if claim_cols:
        roles["claim"] = " / ".join(claim_cols)
    return roles


def label_from_text(raw: str) -> Dict[str, Any]:
    """Map a document's status/grading wording to an E00-06 fact label."""
    t = (raw or "").strip()
    tl = t.lower()
    if not t:
        return {"label_raw": "", "label": "", "requires_evidence": False}
    if "planned" in tl or "规划" in t or "空目录" in t or "待网络" in t:
        return {"label_raw": t, "label": "PLANNED", "requires_evidence": False}
    if "historical" in tl or "legacy" in tl or "历史" in t \
            or "未开始" in t or "纯规划" in t:
        return {"label_raw": t, "label": "HISTORICAL", "requires_evidence": False}
    if "test" in tl and "runtime" in tl:
        return {"label_raw": t, "label": "TEST+RUNTIME", "requires_evidence": True}
    if "test" in tl:
        return {"label_raw": t, "label": "TEST", "requires_evidence": True}
    if "verified" in tl or "runtime" in tl:
        return {"label_raw": t, "label": "RUNTIME", "requires_evidence": True}
    if "source" in tl:
        return {"label_raw": t, "label": "SOURCE", "requires_evidence": True}
    if "implemented" in tl:
        return {"label_raw": t, "label": "SOURCE", "requires_evidence": True}
    if "experimental" in tl:
        return {"label_raw": t, "label": "SOURCE", "requires_evidence": True}
    if "本地工具" in t or "已 gitignore" in t:
        return {"label_raw": t, "label": "LOCAL_TOOL", "requires_evidence": False}
    return {"label_raw": t, "label": "UNKNOWN", "requires_evidence": False}


def _clean_token(t: str) -> str:
    t = t.strip().strip("`").strip("'\"")
    if "#" in t and not t.startswith("#"):
        t = t.split("#", 1)[0]
    t = t.rstrip("/")
    return t


def section_base_dir(section: str) -> str:
    """Return a repo-root-relative prefix for inventory tables whose path
    column is written relative to a section directory (e.g. '3.1 `hqsb/`')."""
    m = _SECTION_BASE_RE.search(section)
    if not m:
        return ""
    base = m.group(1).rstrip("/")
    if base in ("", "."):
        return ""
    return base


def expand_braces(token: str) -> List[str]:
    """Expand shell-style ``{a,b}`` alternation (e.g. {test_util,test_metrics})."""
    m = re.search(r"\{([^{}]+)\}", token)
    if not m:
        return [token]
    variants = [v.strip() for v in m.group(1).split(",") if v.strip()]
    head, tail = token[:m.start()], token[m.end():]
    return [
        f"{head}{v}{tail}" for v in variants
    ]


def extract_path_tokens(cell: str, prefix: str = "") -> List[Dict[str, str]]:
    """Return candidate tokens with their reference base.

    ``origin`` == "link"  -> markdown link, resolve relative to the document dir
    ``origin`` == "root"  -> code span / plain token, resolve repo-root relative
    (the convention used by the three audited documents).
    ``prefix``            -> section-relative base applied to root tokens
    (used for inventory tables whose path column is section-relative).
    """
    if not cell:
        return []
    tokens: List[Dict[str, str]] = []
    # Markdown relative links resolve relative to the containing document.
    for target in _LINK_RE.findall(cell):
        if target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        t = _clean_token(target.split("#", 1)[0])
        if t and not _RATIO_RE.match(t):
            tokens.append({"token": t, "origin": "link"})
    # Backticked code spans: file paths are repo-root relative.
    for span in _CODE_SPAN_RE.findall(cell):
        s = span.strip()
        # skip clearly non-path spans such as CLI snippets
        if any(ch in s for ch in (" ", "=", "|")) and "/" not in s:
            continue
        t = _clean_token(s)
        if not t or _RATIO_RE.match(t):
            continue
        tokens.append({"token": t, "origin": "root"})
    # Remove spans/links already used.
    rest = _CODE_SPAN_RE.sub(" ", cell)
    rest = _LINK_RE.sub(" ", rest)
    for chunk in re.split(r"[；;，,、\s]+", rest):
        chunk = re.split(r"[（(]", chunk)[0].strip().strip("`").strip()
        if not chunk or "/" not in chunk:
            continue
        if _RATIO_RE.match(chunk):
            continue
        tokens.append({"token": chunk, "origin": "root"})
    # drop empty / duplicates (keep order), apply section prefix + braces,
    # and reject noise tokens (parentheses, stray "/", non path-like characters).
    _NOISE_RE = re.compile(r"[()（）:：=\[\]{}]")
    out: List[Dict[str, str]] = []
    seen = set()
    for t in tokens:
        token = t["token"]
        if t["origin"] == "root" and prefix and not token.startswith("/"):
            if not token.startswith(prefix):
                token = f"{prefix}/{token}"
        for variant in expand_braces(token):
            if not variant or variant in ("/", "./") or _NOISE_RE.search(variant):
                continue
            key = (variant, t["origin"])
            if key in seen:
                continue
            seen.add(key)
            out.append({"token": variant, "origin": t["origin"]})
    return out


def normalise_path_candidate(token: str, origin: str, doc_rel: str) -> str:
    """Resolve a candidate token to a repo-root relative path when possible.

    Returns "" when the token escapes the repository or is absolute.
    """
    if token.startswith("/") or token.startswith("~"):
        return ""
    base = os.path.dirname(doc_rel) if origin == "link" else "."
    p = os.path.normpath(os.path.join(base, token))
    if p == ".." or p.startswith("../"):
        return ""
    return p.replace(os.sep, "/")


# ── claim extraction ─────────────────────────────────────────────────────────

def extract_claims(doc_path: Path, doc_rel: str) -> Dict[str, Any]:
    text = doc_path.read_text(encoding="utf-8")
    claims: List[Dict[str, Any]] = []
    for table in parse_markdown_tables(text):
        roles = classify_column(table["headers"])
        has_evidence_col = "evidence" in roles or "location" in roles
        has_label_col = "label" in roles
        if not has_label_col or not has_evidence_col:
            continue
        claim_cols = roles.get("claim", "").split(" / ") if roles.get("claim") else []
        prefix = section_base_dir(table["section"])
        for row in table["rows"]:
            raw_label = row.get(roles.get("label", ""), "")
            lbl = label_from_text(raw_label)
            evidence_cell = row.get(roles.get("evidence", ""), "")
            location_cell = row.get(roles.get("location", ""), "")
            parts = []
            for c in claim_cols:
                v = row.get(c, "")
                if v:
                    parts.append(f"{c}:{v}")
            if not parts:
                parts = [
                    f"{k}:{v}" for k, v in row.items()
                    if k not in {roles.get("label"), roles.get("evidence"),
                                 roles.get("location"), roles.get("id")}
                ]
            cid = row.get(roles.get("id", ""), "") if roles.get("id") else ""
            token_items = extract_path_tokens(
                f"{evidence_cell} ; {location_cell}", prefix=prefix
            )
            claims.append({
                "source_doc": doc_rel,
                "section": table["section"],
                "section_base": prefix,
                "id": cid,
                "claim_text": " | ".join(parts),
                "label_raw": raw_label,
                "fact_label": lbl["label"],
                "requires_evidence": lbl["requires_evidence"],
                "evidence_cell": evidence_cell,
                "location_cell": location_cell,
                "token_items": token_items,
            })
    return {"text": text, "claims": claims}


# ── evidence resolution ──────────────────────────────────────────────────────

def build_workspace_index() -> Tuple[Set[str], Dict[str, str]]:
    """Index workspace files (repo relative) and a basename -> path map."""
    files: Set[str] = set()
    base: Dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(_REPO_ROOT):
        dirnames[:] = [
            d for d in dirnames
            if d not in _EXCLUDE_DIRS and not d.startswith(".")
        ]
        for fn in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fn), _REPO_ROOT)
            rp = rel.replace(os.sep, "/")
            files.add(rp)
            base.setdefault(fn, rp)
    return files, base


def resolve_one_token(
    item: Dict[str, str],
    doc_rel: str,
    tracked_files: Set[str],
    ws_files: Set[str],
    basename_index: Dict[str, str],
) -> Dict[str, Any]:
    token = item["token"]
    origin = item["origin"]
    entry: Dict[str, Any] = {"token": token, "origin": origin}

    if token.startswith("/") or token.startswith("~"):
        expanded = os.path.expanduser(token)
        entry["kind"] = "outside_repo"
        entry["local_exists"] = os.path.exists(expanded)
        entry["note"] = "absolute/~ path; local-only"
        return entry

    if "/" not in token and "*" not in token and "?" not in token:
        # A plain top-level directory name (tests/, hqsb/, ops/ …)
        if os.path.isdir(_REPO_ROOT / token):
            tracked_hit = any(
                f.startswith(token + "/") for f in tracked_files
            )
            ignored, ignore_rule = check_ignore(token)
            if tracked_hit:
                entry["kind"] = "tracked"
            elif ignored:
                entry["kind"] = "local_ignored"
                entry["ignore_rule"] = ignore_rule
            else:
                entry["kind"] = "workspace_only"
            entry["tracked"] = bool(tracked_hit)
            entry["exists"] = True
            entry["is_dir"] = True
            return entry
        # A bare file name with a known extension (repo-root basename lookup)
        if token.endswith((".py", ".cu", ".h", ".hpp", ".c", ".md", ".json",
                           ".jsonl", ".csv", ".txt", ".yaml", ".yml", ".sh",
                           ".toml", ".llir", ".ptx", ".ttgir")):
            match = basename_index.get(token)
            if match:
                entry["kind"] = "basename_resolved"
                entry["resolved_to"] = match
                entry["tracked"] = match in tracked_files
                entry["exists"] = True
            else:
                entry["kind"] = "missing_bare"
                entry["note"] = "bare file name not found in workspace"
            return entry
        entry["kind"] = "non_path"
        entry["note"] = "not a path token (CLI/binary/output description)"
        return entry

    norm = normalise_path_candidate(token, origin, doc_rel)
    if not norm:
        entry["kind"] = "outside_repo_or_escape"
        return entry
    entry["normalised"] = norm

    if any(ch in token for ch in "*?"):
        matches_ws = sorted(fnmatch.filter(ws_files, norm))
        matches_git = sorted(fnmatch.filter(tracked_files, norm))
        entry["kind"] = "glob"
        entry["workspace_matches"] = matches_ws
        entry["git_matches"] = matches_git
        entry["exists"] = bool(matches_ws)
        entry["tracked"] = bool(matches_git)
        return entry

    # Documentation shorthand like "stages/S00–S15.md": no single file exists,
    # but the referenced directory with S00..S15 stage docs may be tracked.
    if re.search(r"[Ss]\d+[–-][Ss]\d+", os.path.basename(norm)):
        d = os.path.dirname(norm)
        tracked_hit = any(f.startswith(d + "/") for f in tracked_files)
        entry["kind"] = "range_to_dir"
        entry["resolved_dir"] = d
        entry["tracked"] = bool(tracked_hit)
        entry["exists"] = bool(tracked_hit) or os.path.isdir(_REPO_ROOT / d)
        entry["note"] = "S00–S15 range shorthand pointing to a tracked dir"
        return entry

    exists = os.path.exists(_REPO_ROOT / norm)
    tracked_hit = norm in tracked_files or any(
        f.startswith(norm + "/") for f in tracked_files
    )
    ignored, ignore_rule = check_ignore(norm)
    if tracked_hit and exists:
        entry["kind"] = "tracked"
    elif exists:
        if ignored:
            entry["kind"] = "local_ignored"
            entry["ignore_rule"] = ignore_rule
        else:
            entry["kind"] = "workspace_only"
    else:
        entry["kind"] = "missing"
    entry["tracked"] = bool(tracked_hit)
    entry["exists"] = bool(exists)
    entry["is_dir"] = os.path.isdir(_REPO_ROOT / norm)
    return entry


# ── numeric claim scan ───────────────────────────────────────────────────────

_NUMERIC_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:tok/s|GB/s|MiB|GiB|ms|s\b|tokens?|checks|passed|%)"
    r"|\+?\d+(?:\.\d+)?\s*[x×]"
)


def numeric_present_in(text: str, token: str) -> bool:
    """True if ``token`` (e.g. "19.10 GB/s") is covered by ``text`` either as an
    exact substring or by its numeric value (decimal-equivalent anywhere)."""
    if token in text:
        return True
    m = re.search(r"(\d+(?:\.\d+)?)", token)
    if not m:
        return False
    target = float(m.group(1))
    for raw in re.finditer(r"\d+(?:\.\d+)?", text):
        try:
            if abs(float(raw.group()) - target) < 1e-6:
                return True
        except ValueError:
            continue
    return False


def extract_numeric_lines(doc_path: Path, doc_rel: str) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    lines = doc_path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines, 1):
        if re.search(r"[0-9a-fA-F]{40}", line):
            continue
        m = _NUMERIC_RE.findall(line)
        if not m:
            continue
        stripped = line.strip()
        if re.fullmatch(r"[|\-: ]+", stripped):
            continue
        hits.append({
            "source_doc": doc_rel,
            "line_no": idx,
            "context": stripped[:200],
            "numbers": [x.strip() for x in m],
        })
    return hits


# ── commit audit ─────────────────────────────────────────────────────────────

_HEX_REF_RE = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{7,40})(?![0-9a-fA-F])")


def scan_commit_references(doc_path: Path, doc_rel: str,
                           commits: List[str], head_full: str) -> Dict[str, Any]:
    text = doc_path.read_text(encoding="utf-8")
    seen: Dict[str, Dict[str, Any]] = {}
    for m in _HEX_REF_RE.finditer(text):
        cand = m.group(1)
        if len(cand) > 40:
            continue
        ctx_start = max(0, m.start() - 60)
        ctx = text[ctx_start:m.end() + 30].replace("\n", " ")
        rec = seen.get(cand)
        if rec is None:
            rec = {
                "reference": cand,
                "is_commit": is_commit_object(cand, commits),
                "len": len(cand),
                "contexts": [],
            }
            if rec["is_commit"]:
                rec["equals_head"] = cand == head_full
                rec["ancestor_of_head"] = commit_is_ancestor_of_head(cand)
            seen[cand] = rec
        rec["contexts"].append(f"{doc_rel}: …{ctx}…")
    refs = sorted(seen.values(), key=lambda r: r["reference"])
    return {
        "all_refs": refs,
        "commit_refs": [r for r in refs if r["is_commit"]],
        "non_commit_refs": [r for r in refs if not r["is_commit"]],
    }


# ── self test ────────────────────────────────────────────────────────────────

def run_self_test() -> Dict[str, Any]:
    """Verify the audit actually detects bad input (guard effectiveness)."""
    results: Dict[str, Any] = {}
    tmpdir = _REPO_ROOT / ".e00_06_selftest"
    tmpdir.mkdir(exist_ok=True)

    # fixture A: isolated runtime claim with a missing evidence path
    fixture_a = tmpdir / "fixture_a.md"
    fixture_a.write_text(
        "# Fixture\n\n"
        "| # | 声明 | 分级 | 证据路径 |\n"
        "|---|---|---|---|\n"
        "| FA1 | 某能力已完成 | runtime-verified | `does/not/exist/at_all.py` |\n",
        encoding="utf-8",
    )
    claim_a = extract_claims(fixture_a, "fixture_a.md")["claims"][0]
    tok = extract_path_tokens(claim_a["evidence_cell"] + " ; " +
                              claim_a["location_cell"])
    ws, base = build_workspace_index()
    res = [resolve_one_token(t, "fixture_a.md", set(), ws, base) for t in tok]
    ok_local = any(r.get("exists") for r in res)
    results["A_isolated_runtime_detected"] = (
        claim_a["fact_label"] == "RUNTIME" and not ok_local and bool(res)
    )

    # fixture B: fabricated commit reference
    fake = "0" * 40
    fixture_b = tmpdir / "fixture_b.md"
    fixture_b.write_text(f"基线 Commit: `{fake}`\n", encoding="utf-8")
    scan = scan_commit_references(fixture_b, "fixture_b.md", [], fake)
    results["B_bad_commit_detected"] = any(
        r["reference"] == fake and not r["is_commit"] for r in scan["all_refs"]
    )

    # fixture C: planned claim with no evidence must not be flagged
    fixture_c = tmpdir / "fixture_c.md"
    fixture_c.write_text(
        "# Fixture\n\n"
        "| # | 声明 | 分级 | 证据路径 |\n"
        "|---|---|---|---|\n"
        "| FC1 | 未来规划 | planned |  |\n",
        encoding="utf-8",
    )
    claim_c = extract_claims(fixture_c, "fixture_c.md")["claims"][0]
    results["C_planned_not_isolated"] = (
        claim_c["fact_label"] == "PLANNED"
        and claim_c["requires_evidence"] is False
    )

    shutil.rmtree(tmpdir, ignore_errors=True)
    results["all_pass"] = all(
        results.get(k) for k in (
            "A_isolated_runtime_detected", "B_bad_commit_detected",
            "C_planned_not_isolated",
        )
    )
    return results


# ── main audit ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="E00-06 doc-claim drift audit.")
    parser.add_argument(
        "--output-dir", default="docs/stage_experiments/S00/E00-06/raw",
        help="Directory for raw artifacts.",
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or f"run_{int(time.time() * 1000)}"
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_records").mkdir(exist_ok=True)
    started_at = _now_utc()

    commands = [
        "python3 scripts/audit/run_e00_06_doc_claim_audit.py "
        f"--output-dir {args.output_dir}"
    ]

    env_block = collect_environment_block()
    git_facts = collect_git_facts()
    head = git_facts["head_full"]
    tracked_set = set(git_facts["tracked_files"])
    commits_all = git_facts["commits"]

    (out_dir / "environment.json").write_text(
        json.dumps(env_block, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    git_facts_public = {k: v for k, v in git_facts.items() if k != "tracked_files"}
    git_facts_public["tracked_sample"] = git_facts["tracked_files"][:10]
    (out_dir / "git_facts.json").write_text(
        json.dumps(git_facts_public, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ws_files, basename_index = build_workspace_index()

    all_claims: List[Dict[str, Any]] = []
    evidence_lines: List[Dict[str, Any]] = []
    doc_meta: Dict[str, Any] = {}
    numeric_lines: List[Dict[str, Any]] = []
    doc_commit_scans: Dict[str, Dict[str, Any]] = {}

    for doc_rel in AUDIT_DOCS:
        doc_path = _REPO_ROOT / doc_rel
        if not doc_path.is_file():
            doc_meta[doc_rel] = {"exists": False, "note": "document missing"}
            continue
        tracked = doc_rel in tracked_set
        last_commit = last_commit_for(doc_rel)
        parsed = extract_claims(doc_path, doc_rel)
        doc_meta[doc_rel] = {
            "exists": True,
            "tracked_in_git": tracked,
            "last_commit_touching": last_commit,
            "last_commit_is_head": last_commit == head,
            "sha256": _file_sha256(doc_path),
            "size_bytes": doc_path.stat().st_size,
            "claim_count": len(parsed["claims"]),
        }

        for claim in parsed["claims"]:
            tokens = claim["token_items"]
            resolved = [
                resolve_one_token(t, doc_rel, tracked_set, ws_files,
                                  basename_index)
                for t in tokens
            ]
            ok_local = any(
                r.get("exists") or (
                    r.get("kind") == "outside_repo" and r.get("local_exists")
                )
                for r in resolved
            )
            ok_git = any(r.get("tracked") for r in resolved)
            claim.update({
                "evidence_tokens": [t["token"] for t in tokens],
                "evidence_resolutions": resolved,
                "evidence_ok_local": bool(ok_local),
                "evidence_ok_git": bool(ok_git),
                "isolated": (
                    claim["requires_evidence"]
                    and bool(resolved)
                    and not ok_local
                ),
                "no_evidence_token": (
                    claim["requires_evidence"] and not tokens
                ),
                "clone_invisible": claim["requires_evidence"] and not ok_git,
            })
            all_claims.append(claim)
            evidence_lines.append({
                "source_doc": claim["source_doc"],
                "section": claim["section"],
                "id": claim["id"],
                "claim_text": claim["claim_text"][:120],
                "fact_label": claim["fact_label"],
                "requires_evidence": claim["requires_evidence"],
                "isolated": claim["isolated"],
                "no_evidence_token": claim["no_evidence_token"],
                "clone_invisible": claim["clone_invisible"],
                "resolutions": resolved,
            })
        numeric_lines.extend(extract_numeric_lines(doc_path, doc_rel))
        doc_commit_scans[doc_rel] = scan_commit_references(
            doc_path, doc_rel, commits_all, head
        )

    commit_audit: Dict[str, Any] = {"documents": {}}
    for doc_rel, scan in doc_commit_scans.items():
        commit_audit["documents"][doc_rel] = {
            "commit_ref_count": len(scan["commit_refs"]),
            "commit_refs": scan["commit_refs"],
            "non_commit_hex_count": len(scan["non_commit_refs"]),
            "non_commit_hex_samples": scan["non_commit_refs"][:20],
        }
    commit_audit["head"] = head
    commit_audit["head_short"] = git_facts["head_short"]

    ledger_claim_texts = [
        c["claim_text"] for c in all_claims
        if c["source_doc"] == "docs/evidence_ledger.md"
    ]

    # Numbers that already appear in a *tracked* markdown report (excluding the
    # three audited docs) are considered evidence-backed (doc-level coverage).
    tracked_md_texts: List[str] = []
    for tracked_f in sorted(tracked_set):
        if not tracked_f.endswith(".md") or tracked_f in AUDIT_DOCS:
            continue
        try:
            tracked_md_texts.append(
                (_REPO_ROOT / tracked_f).read_text(encoding="utf-8")
            )
        except OSError:
            continue
    tracked_md_blob = "\n".join(tracked_md_texts)

    isolated_claims = [
        c for c in all_claims
        if c["requires_evidence"] and (c["isolated"] or c["no_evidence_token"])
    ]

    number_gaps: List[Dict[str, Any]] = []
    for line in numeric_lines:
        covered_in_ledger = any(
            num in " ".join(ledger_claim_texts) for num in line["numbers"]
        )
        covered_in_tracked_md = any(
            numeric_present_in(tracked_md_blob, num) for num in line["numbers"]
        )
        # Does the same line cite an evidence path that exists locally?
        toks = extract_path_tokens(line["context"])
        resolved_toks = [
            resolve_one_token(t, line["source_doc"], tracked_set, ws_files,
                              basename_index)
            for t in toks
        ]
        has_local_evidence = any(
            r.get("exists") or (
                r.get("kind") == "outside_repo" and r.get("local_exists")
            )
            for r in resolved_toks
        )
        line["covered_by_ledger"] = bool(covered_in_ledger)
        line["covered_by_tracked_md"] = bool(covered_in_tracked_md)
        line["has_local_evidence"] = bool(has_local_evidence)
        if not (covered_in_ledger or covered_in_tracked_md or has_local_evidence):
            number_gaps.append(line)

    drift: List[Dict[str, Any]] = []
    for c in isolated_claims:
        drift.append({
            "type": "isolated_claim",
            "source_doc": c["source_doc"],
            "section": c["section"],
            "id": c["id"],
            "claim_text": c["claim_text"][:150],
            "fact_label": c["fact_label"],
            "label_raw": c["label_raw"],
            "evidence_tokens": c["evidence_tokens"],
            "suggested_diff": (
                f"claim {c['id'] or c['claim_text'][:30]} is {c['fact_label']} "
                "but every evidence path is missing locally; demote to "
                "historical-unreproduced/planned or point to a real artifact"
            ),
        })
    for g in number_gaps:
        drift.append({
            "type": "number_without_evidence",
            "source_doc": g["source_doc"],
            "line_no": g["line_no"],
            "context": g["context"],
            "numbers": g["numbers"],
            "suggested_diff": (
                f"numeric claim on {g['source_doc']}:{g['line_no']} "
                "lacks ledger coverage and local evidence; demote or add raw"
            ),
        })

    bad_commits: List[Dict[str, Any]] = []
    for doc_rel, scan in doc_commit_scans.items():
        for r in scan["commit_refs"]:
            if not r.get("ancestor_of_head") and not r.get("equals_head"):
                bad_commits.append({"doc": doc_rel, **r})
    noncommit_refs_total = sum(
        len(scan["non_commit_refs"]) for scan in doc_commit_scans.values()
    )

    baseline_notes = []
    for doc_rel in AUDIT_DOCS:
        meta = doc_meta.get(doc_rel)
        if not meta or not meta["exists"]:
            continue
        baseline_notes.append({
            "doc": doc_rel,
            "last_commit_touching": meta["last_commit_touching"],
            "last_commit_is_head": meta["last_commit_is_head"],
            "tracked_in_git": meta["tracked_in_git"],
        })

    c1_details = [d for d in drift if d["type"] == "isolated_claim"]
    c2_details = [d for d in drift if d["type"] == "number_without_evidence"]
    verdict = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "head": head,
        "criterion_1_no_isolated_runtime_claim": {
            "pass": len(c1_details) == 0,
            "isolated_count": len(c1_details),
            "details": c1_details,
        },
        "criterion_2_no_evidence_number_demotion": {
            "pass": len(c2_details) == 0,
            "gap_count": len(c2_details),
            "details": c2_details,
        },
        "criterion_3_commit_consistent": {
            "pass": len(bad_commits) == 0,
            "bad_commit_count": len(bad_commits),
            "bad_commits": bad_commits,
            "non_commit_hex_references": noncommit_refs_total,
            "notes": ("non-commit hex refs are sha digests / run ids / "
                      "config hashes, audited separately"),
        },
        "documents": doc_meta,
        "baseline_notes": baseline_notes,
        "self_test": None,
        "overall": "PENDING",
    }

    self_test = run_self_test()

    (out_dir / "claim_records.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in all_claims) + "\n",
        encoding="utf-8",
    )
    (out_dir / "evidence_resolution.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in evidence_lines)
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "commit_audit.json").write_text(
        json.dumps(commit_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "numeric_claims.json").write_text(
        json.dumps({
            "scanned_docs": AUDIT_DOCS,
            "hits": numeric_lines,
            "gaps": number_gaps,
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "drift_diff.json").write_text(
        json.dumps(drift, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "self_test.json").write_text(
        json.dumps(self_test, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "command.txt").write_text(
        "\n".join(commands) + "\n", encoding="utf-8"
    )

    c1_pass = verdict["criterion_1_no_isolated_runtime_claim"]["pass"]
    c2_pass = verdict["criterion_2_no_evidence_number_demotion"]["pass"]
    c3_pass = verdict["criterion_3_commit_consistent"]["pass"]
    self_pass = bool(self_test["all_pass"])
    verdict["self_test"] = self_test
    verdict["criteria_summary"] = {
        "no_isolated_runtime_claim": c1_pass,
        "no_evidence_number_demotion": c2_pass,
        "commit_consistent": c3_pass,
        "tool_self_test": self_pass,
    }
    verdict["overall"] = (
        "PASS" if (c1_pass and c2_pass and c3_pass and self_pass) else "FAIL"
    )
    (out_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ended_at = _now_utc()
    record = {
        "stage": STAGE,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "environment": env_block,
        "git_facts": git_facts_public,
        "documents": doc_meta,
        "claim_count": len(all_claims),
        "isolated_claims": c1_details,
        "number_gap_count": len(c2_details),
        "commit_bad_count": len(bad_commits),
        "self_test": self_test,
        "decision": verdict["overall"],
        "criteria_summary": verdict["criteria_summary"],
    }
    json_path = out_dir / f"e00_06_run_{run_id}.json"
    json_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    file_manifest = {}
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            file_manifest[str(p.relative_to(out_dir))] = _file_sha256(p)
    (out_dir / "file_manifest.json").write_text(
        json.dumps(file_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[{EXPERIMENT_ID}] run_id={run_id}")
    print(f"[{EXPERIMENT_ID}] HEAD={git_facts['head_short']} "
          f"dirty={git_facts['dirty']}")
    print(f"[{EXPERIMENT_ID}] docs parsed: " + ", ".join(
        f"{d}->{doc_meta[d]['claim_count']}claims"
        for d in AUDIT_DOCS if doc_meta.get(d, {}).get("exists")
    ))
    print(f"[{EXPERIMENT_ID}] isolated_runtime={len(c1_details)} "
          f"number_gaps={len(c2_details)} bad_commits={len(bad_commits)} "
          f"noncommit_hex={noncommit_refs_total}")
    print(f"[{EXPERIMENT_ID}] self_test pass={self_pass}")
    print(f"[{EXPERIMENT_ID}] decision={verdict['overall']}")
    return 0 if verdict["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
