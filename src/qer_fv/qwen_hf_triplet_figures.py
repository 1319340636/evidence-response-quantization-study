"""Deterministic scientific figures for the Qwen HF triplet report."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


FIGURE_PROTOCOL = "qwen-hf-triplet-figures-v1-20260802"
PALETTE = ("#0072B2", "#E69F00", "#009E73")


def figure_specifications(result: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the exact plotted values before any rendering occurs."""
    contrast_order = result["contrast_order"]
    intervals = [result["contrasts"][name]["page_bootstrap"] for name in contrast_order]
    fp16 = result["hard_labels"]["awq_vs_fp16"]
    gptq = result["hard_labels"]["awq_vs_gptq"]
    return {
        "figure_protocol": FIGURE_PROTOCOL,
        "palette": list(PALETTE),
        "method_contrasts": {
            "labels": ["AWQ − FP16", "AWQ − GPTQ"],
            "estimands": list(contrast_order),
            "estimates": [float(row["estimate"]) for row in intervals],
            "lower": [float(row["lower"]) for row in intervals],
            "upper": [float(row["upper"]) for row in intervals],
            "zero_reference_line": True,
            "interval_label": "95% page-bootstrap CI",
        },
        "hard_label_metrics": {
            "routes": ["FP16", "GPTQ INT4", "AWQ INT4"],
            "accuracy": [
                float(fp16["accuracy"]["reference"]),
                float(gptq["accuracy"]["reference"]),
                float(fp16["accuracy"]["awq"]),
            ],
            "balanced_accuracy": [
                float(fp16["balanced_accuracy"]["reference"]),
                float(gptq["balanced_accuracy"]["reference"]),
                float(fp16["balanced_accuracy"]["awq"]),
            ],
            "mcc": [
                float(fp16["mcc"]["reference"]),
                float(gptq["mcc"]["reference"]),
                float(fp16["mcc"]["awq"]),
            ],
            "agreement": {
                "AWQ vs FP16": float(fp16["agreement"]),
                "AWQ vs GPTQ": float(gptq["agreement"]),
            },
        },
    }


def render_qwen_hf_triplet_figures(
    result: Mapping[str, Any], output_directory: str | Path
) -> dict[str, Path]:
    """Render the two minimal paper figures and their interpretation catalog."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    output = Path(output_directory)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    specs = figure_specifications(result)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 240,
        }
    )

    contrast = specs["method_contrasts"]
    estimates = contrast["estimates"]
    lower = contrast["lower"]
    upper = contrast["upper"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    positions = [1, 0]
    errors = [
        [estimate - bound for estimate, bound in zip(estimates, lower, strict=True)],
        [bound - estimate for estimate, bound in zip(estimates, upper, strict=True)],
    ]
    ax.errorbar(
        estimates,
        positions,
        xerr=errors,
        fmt="o",
        color=PALETTE[0],
        ecolor=PALETTE[0],
        capsize=5,
        linewidth=2,
        markersize=7,
        label="Estimate and 95% page-bootstrap CI",
    )
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.2)
    ax.set_yticks(positions, contrast["labels"])
    ax.set_xlabel("Change in evidence-interaction score (AWQ minus reference)")
    ax.set_title("Within-HF method-associated interaction contrasts")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.legend(loc="upper right", frameon=False)
    _save_pair(fig, figures / "figure-01-method-contrasts")
    plt.close(fig)

    hard = specs["hard_label_metrics"]
    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    centers = list(range(3))
    width = 0.23
    for offset, (label, values, color) in enumerate(
        (
            ("Accuracy", hard["accuracy"], PALETTE[0]),
            ("Balanced accuracy", hard["balanced_accuracy"], PALETTE[1]),
            ("MCC", hard["mcc"], PALETTE[2]),
        )
    ):
        positions = [center + (offset - 1) * width for center in centers]
        bars = ax.bar(positions, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(centers, hard["routes"])
    ax.set_ylim(0.0, 0.86)
    ax.set_ylabel("Score")
    ax.set_title("Hard-label performance by Qwen3.5-9B HF route")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.legend(loc="upper right", frameon=False, ncols=3)
    agreement = hard["agreement"]
    ax.text(
        0.5,
        0.025,
        "Paired prediction agreement: "
        f"AWQ vs FP16 = {agreement['AWQ vs FP16']:.3f}; "
        f"AWQ vs GPTQ = {agreement['AWQ vs GPTQ']:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        transform=ax.transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    _save_pair(fig, figures / "figure-02-hard-label-metrics")
    plt.close(fig)

    catalog = output / "figure-catalog.md"
    catalog.write_text(_catalog_text(specs), encoding="utf-8", newline="\n")
    names = (
        "figures/figure-01-method-contrasts.png",
        "figures/figure-01-method-contrasts.pdf",
        "figures/figure-02-hard-label-metrics.png",
        "figures/figure-02-hard-label-metrics.pdf",
        "figure-catalog.md",
    )
    return {name: output / name for name in names}


def _save_pair(fig: Any, stem: Path) -> None:
    fig.savefig(
        stem.with_suffix(".png"),
        bbox_inches="tight",
        metadata={"Software": FIGURE_PROTOCOL},
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Title": stem.name,
            "Author": "Experiment B",
            "Creator": FIGURE_PROTOCOL,
            "Producer": FIGURE_PROTOCOL,
            "CreationDate": datetime(2026, 8, 2, tzinfo=timezone.utc),
            "ModDate": datetime(2026, 8, 2, tzinfo=timezone.utc),
        },
    )


def _catalog_text(specs: Mapping[str, Any]) -> str:
    contrast = specs["method_contrasts"]
    hard = specs["hard_label_metrics"]
    return (
        "# Figure catalog\n\n"
        "## Figure 1 — Within-HF method-associated interaction contrasts\n\n"
        "**Purpose.** Show the magnitude and uncertainty of the two frozen "
        "paired contrasts without treating statistical significance as effect "
        "size. The dashed vertical line marks a null contrast of zero.\n\n"
        "**Caption.** Qwen3.5-9B AWQ INT4 minus FP16 and AWQ INT4 minus GPTQ "
        f"INT4 evidence-interaction effects. Points are page-balanced estimates "
        f"({contrast['estimates'][0]:.6f} and {contrast['estimates'][1]:.6f}); "
        f"bars are {contrast['interval_label']} "
        f"([{contrast['lower'][0]:.6f}, {contrast['upper'][0]:.6f}] and "
        f"[{contrast['lower'][1]:.6f}, {contrast['upper'][1]:.6f}]).\n\n"
        "**Interpretation.** Negative values indicate that the AWQ route changed "
        "the interaction score downward relative to each reference route. This "
        "is a method-associated, single-model result and not a pure causal "
        "effect of the AWQ algorithm.\n\n"
        "## Figure 2 — Hard-label performance by route\n\n"
        "**Purpose.** Place interaction changes beside conventional hard-label "
        "accuracy, balanced accuracy, and multiclass MCC, while retaining the "
        "paired prediction-agreement diagnostic.\n\n"
        "**Caption.** Scores for FP16, GPTQ INT4, and AWQ INT4 are respectively "
        f"accuracy {hard['accuracy']}, balanced accuracy "
        f"{hard['balanced_accuracy']}, and MCC {hard['mcc']}. Paired prediction "
        f"agreement is {hard['agreement']['AWQ vs FP16']:.6f} for AWQ versus "
        f"FP16 and {hard['agreement']['AWQ vs GPTQ']:.6f} for AWQ versus GPTQ.\n\n"
        "**Interpretation.** The hard-label metrics move in the same broad "
        "direction as the interaction contrasts in this frozen Qwen experiment. "
        "The figure does not establish generality across model families, and it "
        "does not remove residual differences in packing, loaders, or kernels.\n"
    )
