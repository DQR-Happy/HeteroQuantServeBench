#!/usr/bin/env python3
"""E02-03 per-step ITL figures — offline scientific plotting (Mac-only exception).

This script reads only the *already saved* E02-03 raw evidence and renders
static, print-ready figures. It intentionally imports no project module that
would initialise torch/CUDA or load a model, and it never writes to ``raw/``.
See ``AGENTS.md`` §3 for the local offline-plotting exception.

Metric boundary (frozen by E02-01 / E02-03):

* x-axis  : decode step index, 1-based. Step 1 is the first decode iteration
  *after* prefill, i.e. it produces output token #2. The first token comes from
  prefill and is not part of the ITL series.
* y-axis  : decode-step latency (ITL, ms). Under the current protocol each value
  is the module-level step latency measured as
  ``[cuda.synchronize] -> perf_counter -> forward+argmax -> [cuda.synchronize]``.
  It is **not** a pure kernel time and **not** a client-side streaming gap.
* series  : 6 workloads x 3 independent process runs x 3 requests = 54 series,
  5274 decode steps. Per workload the 9 series are aligned by step index; the
  main curve is the pointwise median and the band is the pointwise IQR
  (P25-P75). The IQR is an observed spread, not a 95% CI: the 9 requests are
  nested in 3 processes and are not 9 independent experiments.

Outputs (default ``docs/stage_experiments/S02/E02-03/figures/itl/``):

* ``itl_overview_6workloads.{pdf,svg,png}`` — 2x3 unified-axis overview.
* ``itl_detail_<workload>.{pdf,svg,png}``    — one figure per workload.
* ``itl_per_step_stats.csv``                  — pointwise statistics table.
* ``plot_manifest.json``                      — inputs/config/deps/command.
* ``README.md``                               — captions + reproduction.

Usage::

    .venv/bin/python scripts/analysis/plot_e02_03_itl.py
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless, no display / no Jupyter

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# ── Frozen design constants ────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]

WORKLOAD_ORDER = [
    "tiny",
    "short",
    "balanced",
    "long_prefill",
    "decode_heavy",
    "long_balanced",
]

# Authoritative (ISL, OSL) from configs/benchmarks/jetson_qwen3_fp16.yaml.
EXPECTED_CFG = {
    "tiny": (32, 16),
    "short": (128, 32),
    "balanced": (512, 128),
    "long_prefill": (2048, 32),
    "decode_heavy": (128, 256),
    "long_balanced": (2048, 128),
}

DISPLAY_NAME = {
    "tiny": "tiny",
    "short": "short",
    "balanced": "balanced",
    "long_prefill": "long-prefill",
    "decode_heavy": "decode-heavy",
    "long_balanced": "long-balanced",
}

COLOR_RUN = ["#0072B2", "#D55E00", "#009E73"]  # Okabe-Ito, colour-blind safe
LSTYLE_RUN = ["-", (0, (4.0, 1.8)), (0, (1.2, 1.2))]  # grayscale-safe channel
REP_ALPHA = [0.90, 0.55, 0.30]

COLOR_MEDIAN = "#1A1A1A"
COLOR_IQR = "#0072B2"
COLOR_RAW_OVERVIEW = "#8C8C8C"

# Unified linear y-range: covers every raw value (global min 95.55 ms,
# global max 152.89 ms) so the long-input first-step spike is not clipped.
YLIM = (90.0, 160.0)

FIG_W_OVERVIEW = 7.16  # double-column paper width (inches)
FIG_H_OVERVIEW = 3.80
FIG_W_DETAIL = 3.45    # single-column width (inches)
FIG_H_DETAIL = 2.70

PNG_DPI = 300

_FIG_DIR_NAME = "figures/itl"


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # Arial first: a clean, professional TrueType face that subsets
            # without the macOS Helvetica.ttc AAT-table warnings.
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,   # embed TrueType (subset) in PDF
            "ps.fonttype": 42,
            "svg.fonttype": "none",  # keep SVG text editable
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#4D4D4D",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.5,
            "grid.alpha": 1.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.color": "#4D4D4D",
            "ytick.color": "#4D4D4D",
            "axes.labelcolor": "#1A1A1A",
            "text.color": "#1A1A1A",
            "figure.dpi": PNG_DPI,
            "savefig.dpi": PNG_DPI,
        }
    )


# ── Data loading / verification ────────────────────────────────────────────

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runs(raw_dir: Path) -> List[dict]:
    runs: List[dict] = []
    for i in range(3):
        path = raw_dir / f"run_{i}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing raw run file: {path}")
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    return runs


def extract_matrix(runs: Sequence[dict], name: str) -> Tuple[np.ndarray, List[dict]]:
    """Return a ``(9, G-1)`` ITL matrix and its ``(run, repetition)`` metadata."""
    series: List[np.ndarray] = []
    meta: List[Dict[str, int]] = []
    for run in runs:
        wl = next((w for w in run["workloads"] if w["name"] == name), None)
        if wl is None:
            raise KeyError(f"workload {name!r} absent from run {run.get('run_index')}")
        for sample in wl["samples"]:
            series.append(np.asarray(sample["raw_itl_ms"], dtype=float))
            meta.append(
                {
                    "run": int(run["run_index"]),
                    "rep": int(sample["repetition"]),
                }
            )
    return np.vstack(series), meta


def verify_coverage(runs: Sequence[dict]) -> Dict[str, object]:
    """Audit identities, step counts, positivity and anomalies; never "fix" data."""
    problems: List[str] = []
    warnings: List[str] = []
    per_workload: Dict[str, dict] = {}
    total_series = 0
    total_steps = 0

    for name in WORKLOAD_ORDER:
        exp_isl, exp_osl = EXPECTED_CFG[name]
        seen_cfg = set()
        anomalies: List[dict] = []
        pairs = set()
        length_ok = True
        finite_ok = True
        positive_ok = True
        n_series = 0

        for run in runs:
            wl = next((w for w in run["workloads"] if w["name"] == name), None)
            if wl is None:
                problems.append(f"{name}: absent in run {run.get('run_index')}")
                continue
            seen_cfg.add((int(wl["input_tokens"]), int(wl["output_tokens"])))
            anomalies.extend(wl.get("anomalies", []))
            for sample in wl["samples"]:
                n_series += 1
                pairs.add((int(run["run_index"]), int(sample["repetition"])))
                arr = np.asarray(sample["raw_itl_ms"], dtype=float)
                if arr.size != exp_osl - 1:
                    length_ok = False
                    problems.append(
                        f"{name}: run {run['run_index']} rep {sample['repetition']} "
                        f"has {arr.size} ITL steps, expected {exp_osl - 1}"
                    )
                if not np.all(np.isfinite(arr)):
                    finite_ok = False
                if arr.size and np.any(arr <= 0):
                    positive_ok = False

        if seen_cfg and seen_cfg != {(exp_isl, exp_osl)}:
            problems.append(
                f"{name}: config drift across runs: {sorted(seen_cfg)}, "
                f"expected {[(exp_isl, exp_osl)]}"
            )
        if n_series != 9:
            problems.append(f"{name}: {n_series} series, expected 9")
        if len(pairs) != n_series:
            problems.append(f"{name}: duplicate (run, repetition) identities")
        if anomalies:
            # Kept, never silently dropped; surfaced as an explicit warning.
            warnings.append(f"{name}: {len(anomalies)} recorded anomaly entry(ies) kept")

        matrix, _ = extract_matrix(runs, name)
        total_series += matrix.shape[0]
        total_steps += matrix.shape[0] * matrix.shape[1]

        per_workload[name] = {
            "isl": exp_isl,
            "osl": exp_osl,
            "decode_steps": exp_osl - 1,
            "series": matrix.shape[0],
            "config_consistent": seen_cfg == {(exp_isl, exp_osl)},
            "length_ok": length_ok,
            "finite_ok": finite_ok,
            "positive_ok": positive_ok,
            "anomalies": len(anomalies),
            "global_min_ms": float(matrix.min()),
            "global_max_ms": float(matrix.max()),
        }

    if not (YLIM[0] <= min(v["global_min_ms"] for v in per_workload.values())):
        problems.append("unified y-range does not cover the global minimum")
    if not (YLIM[1] >= max(v["global_max_ms"] for v in per_workload.values())):
        problems.append("unified y-range does not cover the global maximum")
    if total_series != 54:
        problems.append(f"total series {total_series}, expected 54")
    if total_steps != 5274:
        problems.append(f"total steps {total_steps}, expected 5274")

    return {
        "per_workload": per_workload,
        "total_series": total_series,
        "total_steps": total_steps,
        "problems": problems,
        "warnings": warnings,
    }


def print_coverage(report: Dict[str, object]) -> None:
    print("E02-03 ITL coverage audit")
    print("=" * 88)
    print(
        f"{'workload':14s} {'ISL':>5s} {'OSL':>4s} {'steps':>5s} "
        f"{'series':>6s} {'min ms':>8s} {'max ms':>8s} {'cfg':>4s} "
        f"{'len':>4s} {'finite':>6s} {'>0':>4s} {'anom':>5s}"
    )
    for name in WORKLOAD_ORDER:
        v = report["per_workload"][name]
        print(
            f"{name:14s} {v['isl']:5d} {v['osl']:4d} {v['decode_steps']:5d} "
            f"{v['series']:6d} {v['global_min_ms']:8.2f} {v['global_max_ms']:8.2f} "
            f"{'ok' if v['config_consistent'] else 'BAD':>4s} "
            f"{'ok' if v['length_ok'] else 'BAD':>4s} "
            f"{'ok' if v['finite_ok'] else 'BAD':>6s} "
            f"{'ok' if v['positive_ok'] else 'BAD':>4s} "
            f"{v['anomalies']:5d}"
        )
    print("-" * 88)
    print(f"total series = {report['total_series']}   total steps = {report['total_steps']}")
    for w in report["warnings"]:
        print(f"WARN : {w}")
    for p in report["problems"]:
        print(f"ERROR: {p}")
    print("=" * 88)


# ── Statistics ─────────────────────────────────────────────────────────────

def pointwise_stats(matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """Pointwise (per-column) statistics across the 9 requests."""
    return {
        "median": np.median(matrix, axis=0),
        "p25": np.percentile(matrix, 25, axis=0),
        "p75": np.percentile(matrix, 75, axis=0),
        "min": np.min(matrix, axis=0),
        "max": np.max(matrix, axis=0),
        "n": np.full(matrix.shape[1], matrix.shape[0]),
    }


def steps_axis(matrix: np.ndarray) -> np.ndarray:
    return np.arange(1, matrix.shape[1] + 1)


def _style_axis(ax) -> None:
    ax.grid(True, which="major", axis="both")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#4D4D4D")
        spine.set_linewidth(0.6)
    ax.tick_params(which="minor", length=0)


def _markers_for(n_steps: int):
    return "o" if n_steps <= 32 else None


# ── Figure A: unified 2x3 overview ─────────────────────────────────────────

def plot_overview(
    matrices: Dict[str, np.ndarray],
    out_base: Path,
) -> List[Path]:
    _configure_style()
    fig, axes = plt.subplots(
        2, 3, figsize=(FIG_W_OVERVIEW, FIG_H_OVERVIEW), constrained_layout=True
    )

    for ax, name in zip(axes.ravel(), WORKLOAD_ORDER):
        matrix = matrices[name]
        stats = pointwise_stats(matrix)
        x = steps_axis(matrix)
        isl, osl = EXPECTED_CFG[name]

        # Individual requests, deliberately faded.
        marker = _markers_for(matrix.shape[1])
        for row in matrix:
            ax.plot(
                x, row, color=COLOR_RAW_OVERVIEW, lw=0.5, alpha=0.28,
                marker=marker, ms=1.6, mew=0.0, zorder=1,
            )

        # IQR band + median on top.
        ax.fill_between(
            x, stats["p25"], stats["p75"], color=COLOR_IQR, alpha=0.25,
            linewidth=0, zorder=2,
        )
        ax.plot(
            x, stats["median"], color=COLOR_MEDIAN, lw=1.6, zorder=3,
            marker=marker, ms=2.1, mew=0.0,
        )

        ax.set_xlim(1, matrix.shape[1])
        ax.set_ylim(*YLIM)
        ax.set_title(f"{DISPLAY_NAME[name]}  (ISL {isl} / OSL {osl})", pad=3.0)
        ax.text(
            0.97, 0.035, "9 requests · 3 processes",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.3, color="#5A5A5A",
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        _style_axis(ax)

    fig.supxlabel(
        "Decode step index  (1 = first step after prefill; produces output token 2)",
        fontsize=8,
    )
    fig.supylabel("Decode-step latency (ITL, ms)", fontsize=8)

    handles = [
        Line2D([], [], color=COLOR_MEDIAN, lw=1.6, label="pointwise median (n=9)"),
        Patch(facecolor=COLOR_IQR, alpha=0.25, label="IQR (P25\u2013P75)"),
        Line2D([], [], color=COLOR_RAW_OVERVIEW, lw=0.6, alpha=0.6,
               label="individual requests (9)"),
    ]
    fig.legend(
        handles=handles, loc="outside upper center", ncol=3, frameon=False,
    )

    return _save_figure(fig, out_base)


# ── Figure B: single-workload detail ───────────────────────────────────────

def plot_detail(
    name: str,
    matrix: np.ndarray,
    meta: Sequence[dict],
    out_base: Path,
) -> List[Path]:
    _configure_style()
    fig, ax = plt.subplots(figsize=(FIG_W_DETAIL, FIG_H_DETAIL), constrained_layout=True)

    stats = pointwise_stats(matrix)
    x = steps_axis(matrix)
    isl, osl = EXPECTED_CFG[name]
    marker = _markers_for(matrix.shape[1])

    for row, info in zip(matrix, meta):
        ax.plot(
            x, row,
            color=COLOR_RUN[info["run"]],
            ls=LSTYLE_RUN[info["run"]],
            lw=0.8,
            alpha=REP_ALPHA[info["rep"]],
            marker=marker, ms=1.7, mew=0.0,
            zorder=2,
        )

    ax.fill_between(
        x, stats["p25"], stats["p75"], color=COLOR_IQR, alpha=0.12,
        linewidth=0, zorder=1,
    )
    ax.plot(
        x, stats["median"], color=COLOR_MEDIAN, lw=2.2, zorder=4,
        marker=marker, ms=2.4, mew=0.0,
    )

    ax.set_xlim(1, matrix.shape[1])
    ax.set_ylim(*YLIM)
    ax.set_title(f"{DISPLAY_NAME[name]}  (ISL {isl} / OSL {osl})", pad=4.0)
    ax.set_xlabel("Decode step index")
    ax.set_ylabel("Decode-step latency (ITL, ms)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    _style_axis(ax)

    handles = [
        Line2D([], [], color=COLOR_RUN[r], ls=LSTYLE_RUN[r], lw=1.1, label=f"run {r}")
        for r in range(3)
    ]
    handles.append(Line2D([], [], color=COLOR_MEDIAN, lw=2.2, label="median (n=9)"))
    ax.legend(
        handles=handles, loc="upper right", frameon=True, framealpha=0.92,
        edgecolor="#D0D0D0", borderpad=0.35, handlelength=1.6, labelspacing=0.3,
    )

    return _save_figure(fig, out_base)


# ── Saving / tables / manifest / README ────────────────────────────────────

def _save_figure(fig, out_base: Path) -> List[Path]:
    written: List[Path] = []
    for ext in ("pdf", "svg", "png"):
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, format=ext, bbox_inches="tight", pad_inches=0.02)
        written.append(path)
    plt.close(fig)  # one figure at a time to bound memory
    return written


def write_stats_csv(matrices: Dict[str, np.ndarray], path: Path) -> None:
    lines = [
        "workload,step,context_length,n,median_ms,p25_ms,p75_ms,min_ms,max_ms"
    ]
    for name in WORKLOAD_ORDER:
        matrix = matrices[name]
        stats = pointwise_stats(matrix)
        isl, _osl = EXPECTED_CFG[name]
        for i in range(matrix.shape[1]):
            step = i + 1
            lines.append(
                f"{name},{step},{isl + step},{int(stats['n'][i])},"
                f"{stats['median'][i]:.4f},{stats['p25'][i]:.4f},"
                f"{stats['p75'][i]:.4f},{stats['min'][i]:.4f},{stats['max'][i]:.4f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    path: Path,
    raw_dir: Path,
    run_paths: Sequence[Path],
    report: Dict[str, object],
    figures: Sequence[Path],
) -> None:
    manifest = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generator": "scripts/analysis/plot_e02_03_itl.py",
        "command": " ".join([".venv/bin/python", *sys.argv]),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
        },
        "inputs": {
            p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size}
            for p in run_paths
        },
        "input_dir": _repo_rel(raw_dir),
        "metric": {
            "x": "decode step index (1-based, 1 = first step after prefill)",
            "y": "decode-step latency ITL (ms), synchronized module-level step time",
            "series_per_workload": 9,
            "processes": 3,
            "requests_per_process_per_workload": 3,
            "band": "pointwise IQR (P25-P75), observed spread not a 95% CI",
        },
        "unified_ylim_ms": list(YLIM),
        "workloads": {
            name: {
                "isl": EXPECTED_CFG[name][0],
                "osl": EXPECTED_CFG[name][1],
                "decode_steps": EXPECTED_CFG[name][1] - 1,
            }
            for name in WORKLOAD_ORDER
        },
        "coverage": report,
        "figures": [_repo_rel(p) for p in figures],
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _figure_files(out_dir: Path) -> List[Path]:
    files: List[Path] = []
    for name in ["itl_overview_6workloads"] + [
        f"itl_detail_{w}" for w in WORKLOAD_ORDER
    ]:
        for ext in ("pdf", "svg", "png"):
            files.append(out_dir / f"{name}.{ext}")
    return files


def write_readme(
    path: Path,
    out_dir: Path,
    raw_rel: str,
    report: Dict[str, object],
) -> None:
    _gmin = min(v["global_min_ms"] for v in report["per_workload"].values())
    _gmax = max(v["global_max_ms"] for v in report["per_workload"].values())
    text = f"""# E02-03 per-step ITL figures

Offline scientific figures for the six E02-03 workloads. Generated by
`scripts/analysis/plot_e02_03_itl.py` from the frozen raw evidence in
`{raw_rel}`. Raw JSON is read-only; the PASS/FAIL verdict is untouched.

## Metric boundary

- **x-axis** — decode step index, 1-based. Step 1 is the first decode iteration
  *after* prefill and produces output token #2; the first output token comes
  from prefill and is not part of the ITL series.
- **y-axis** — decode-step latency (ITL, ms) under the E02-01 clock: each value
  is bracketed by `torch.cuda.synchronize()` around a `perf_counter` window that
  contains the forward pass and the argmax. It is a module-level step latency,
  **not** a pure kernel time and **not** a client-side streaming inter-token gap.
- **context length** (CSV) — recorded after-forward as `ISL + decode_step`.
  This is the KV length at that step, **not** the new query length (which is 1).

## Statistics

Per workload the 9 request series (3 processes x 3 requests) are aligned by
step index. Main curve = pointwise median; band = pointwise IQR (P25-P75);
`min`/`max` are pointwise observed extremes. The IQR is an observed spread, not
a 95% confidence interval: the 9 requests are nested in 3 processes, so they are
not 9 independent experiments. No smoothing, fitting, clipping or outlier
removal is applied; early spikes are kept.

## Unified scale

All overview panels and detail figures share one linear y-range,
`{YLIM[0]:.0f}-{YLIM[1]:.0f} ms`, which covers every raw value (global observed
min {_gmin:.2f} ms, max {_gmax:.2f} ms) so the long-input first-step spike is
visible and panels stay comparable. Each panel keeps its own x-range
(15-255 steps); series of different lengths are never concatenated.

## Files

| file | content |
|---|---|
| `itl_overview_6workloads.pdf/svg/png` | 2x3 unified-axis overview |
| `itl_detail_<workload>.pdf/svg/png` | one detail figure per workload |
| `itl_per_step_stats.csv` | workload, step, context_length, n, median, P25, P75, min, max |
| `plot_manifest.json` | input SHA256, config, dependency versions, command |

Vector formats: PDF (embedded TrueType subset) and SVG (text kept editable);
raster: PNG at {PNG_DPI} dpi. Figures are produced at final print size
(overview {FIG_W_OVERVIEW} in wide, detail {FIG_W_DETAIL} in wide) so labels stay
readable at 100%.

## Captions

**Overview (EN).** Per-step decode latency (ITL) for the six E02-03 workloads at
batch size 1 (Qwen3-1.7B, FP16, eager attention, Jetson Orin). Solid black,
pointwise median of 9 requests (3 independent processes x 3 requests); shaded
band, pointwise IQR (P25-P75); faint grey lines, individual requests. Decode step
1 is the first step after prefill. All panels share the y-range
{YLIM[0]:.0f}-{YLIM[1]:.0f} ms; x-ranges differ per workload.

**总览（中文）。** 六个 E02-03 workload 的逐 step decode 时延（ITL），batch=1
（Qwen3-1.7B，FP16，eager attention，Jetson Orin）。黑实线为 9 条请求（3 个独立
进程 × 3 请求）的逐 step 中位数；阴影带为逐 step IQR（P25–P75）；浅灰细线为各
请求原始曲线。decode step 1 是 prefill 之后的第一个 decode 步。六面板共用
{YLIM[0]:.0f}–{YLIM[1]:.0f} ms 纵轴，横轴各自覆盖本 workload 的完整步数。

**Detail (EN).** A single workload with all 9 raw request curves plus the
pointwise median. Colour and line style identify the process run
(blue solid = run 0, vermillion dashed = run 1, green dash-dot = run 2);
opacity distinguishes the three repetitions within a run. Same y-range as the
overview. Points are drawn only for short series (<=32 steps).

**详情图（中文）。** 单个 workload 的全部 9 条原始请求曲线与逐 step 中位数。
颜色与线型标识进程 run（蓝实线=run 0，朱红虚线=run 1，绿点划线=run 2），
透明度区分同 run 的三个 repetition。纵轴与总览一致。仅短序列（≤32 步）显示采样点。

## Reproduce

```bash
cd {_REPO_ROOT}
python3 -m venv .venv                     # project-local, gitignored
.venv/bin/python -m pip install matplotlib=={matplotlib.__version__} numpy=={np.__version__}
.venv/bin/python scripts/analysis/plot_e02_03_itl.py
```

Optional: ``--raw-dir`` / ``--out-dir`` override the input and output paths.
"""
    path.write_text(text, encoding="utf-8")


# ── Entry point ────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E02-03 per-step ITL figures")
    parser.add_argument(
        "--raw-dir",
        default=str(_REPO_ROOT / "docs/stage_experiments/S02/E02-03/raw"),
        help="directory holding run_0.json / run_1.json / run_2.json",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_REPO_ROOT / "docs/stage_experiments/S02/E02-03" / _FIG_DIR_NAME),
        help="figure output directory",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(raw_dir)
    report = verify_coverage(runs)
    print_coverage(report)
    if report["problems"]:
        print("Aborting: raw-data coverage problems found (no data was modified).")
        return 2

    matrices = {name: extract_matrix(runs, name)[0] for name in WORKLOAD_ORDER}
    metas = {name: extract_matrix(runs, name)[1] for name in WORKLOAD_ORDER}

    figures: List[Path] = []
    figures += plot_overview(matrices, out_dir / "itl_overview_6workloads")
    for name in WORKLOAD_ORDER:
        figures += plot_detail(
            name, matrices[name], metas[name], out_dir / f"itl_detail_{name}"
        )

    write_stats_csv(matrices, out_dir / "itl_per_step_stats.csv")
    write_manifest(
        out_dir / "plot_manifest.json",
        raw_dir,
        [raw_dir / f"run_{i}.json" for i in range(3)],
        report,
        _figure_files(out_dir),
    )
    write_readme(
        out_dir / "README.md", out_dir, _repo_rel(raw_dir), report
    )

    print(f"wrote {len(figures)} figure files + stats/manifest/README to {_repo_rel(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
