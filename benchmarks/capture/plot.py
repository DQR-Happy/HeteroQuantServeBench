"""Offline plots of saved benchmark results only; no HQSB runtime imports."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.aggregate.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    labels = ["hqsb", "nanolog", "hindsight", "otel"]
    names = ["HQSB", "NanoLog", "Hindsight C", "OTel SDK"]
    colors = ["#285F9D", "#D78436"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    modes = ["hybrid_spin", "cooperative"]
    metrics = [("cohort_completion_rate", "Valid goodput (events/s)", 1),
               ("scheduled_delivery_ratio", "Delivered / scheduled (%)", 100),
               ("sampled_peak_PSS_KiB", "Sampled process PSS (MiB)", 1/1024)]
    for ax, (metric, ylabel, scale) in zip(axes, metrics):
        for k, mode in enumerate(modes):
            values, lower, upper = [], [], []
            for name in labels:
                hit = [r for r in rows if r["campaign"] == "pacer-ab-v6" and r["case_id"] == "pacer-10k"
                       and r["adapter"] == name and r["pacer"] == mode and r["measurement_pass"] == "throughput"]
                stats = hit[0][metric] if hit else {}
                value = stats.get("median")
                values.append(value * scale if value is not None else float("nan"))
                lower.append((value - stats.get("min", value)) * scale if value is not None else 0)
                upper.append((stats.get("max", value) - value) * scale if value is not None else 0)
            ax.bar([i + (k - 0.5) * 0.36 for i in range(4)], values, width=0.34,
                   yerr=[lower, upper], capsize=2,
                   label="Busy/hybrid pacer" if k == 0 else "Cooperative pacer", color=colors[k])
        ax.set_xticks(range(4), names, rotation=15)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[1].set_ylim(0, 105)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               ncol=2, frameon=False, fontsize=10)
    fig.suptitle("10,000 planned events/s: arrival-generator interaction matters", fontsize=13)
    fig.text(0.5, 0.015, "Medians and min-max of 3 process runs; stock buffers differ. PSS includes source/runtime, not collector-only memory.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.86))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.output / ("pacer-interaction." + suffix), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
