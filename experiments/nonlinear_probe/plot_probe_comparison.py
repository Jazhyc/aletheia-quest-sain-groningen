"""
Phase 2 figure: MLP vs logistic probes, cross-scenario layer curves plus
per-organism holdout accuracy.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#40403a",
    "xtick.color": "#6f6e64",
    "ytick.color": "#6f6e64",
    "text.color": "#26251f",
})

SERIES_COLORS = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100",
                 "gray": "#9b9a8f"}
REPO = Path(__file__).resolve().parents[2]


def plot_cross_panel(axes: plt.Axes, cross_frame: pd.DataFrame) -> None:
    """
    Cross-scenario balanced accuracy by layer (concat pooling), one color
    per direction, solid/filled for the MLP and dashed/open for logistic.

    :param axes: Target axes.
    :param cross_frame: cross_results.csv rows.
    """
    qwen = cross_frame[(cross_frame.base_model == "qwen") & (cross_frame.pooling == "concat")]
    directions = [
        ("varied", SERIES_COLORS["blue"], "trained on varied,\ntested on instructed", -0.012),
        ("instructed", SERIES_COLORS["yellow"], "trained on instructed,\ntested on varied", 0.012),
    ]
    for train_scenario, color, label, label_offset in directions:
        direction = qwen[qwen.train_scenario == train_scenario]
        for probe, linestyle, fill in [("mlp-512", "-", color), ("logistic", (0, (4, 3)), "white")]:
            frame = direction[direction.probe == probe].sort_values("layer")
            axes.plot(frame.layer, frame.balanced_accuracy, color=color, linewidth=2,
                      linestyle=linestyle, marker="o", markersize=4.5,
                      markerfacecolor=fill, markeredgecolor=color, zorder=3)
        last_value = direction[direction.layer == direction.layer.max()].balanced_accuracy.mean()
        axes.annotate(label, xy=(48.6, last_value + label_offset), color=color,
                      fontsize=9, fontweight="bold", va="center", linespacing=1.25)
    axes.set_xlim(35, 57)
    axes.set_ylim(0.70, 1.005)
    axes.set_xticks([36, 40, 44, 46, 48])
    axes.set_xlabel("decoder layer the probe reads from")
    axes.set_ylabel("balanced accuracy")
    axes.yaxis.grid(True, color="#e5e4dc", linewidth=0.8)
    axes.set_axisbelow(True)
    axes.set_title("Cross-scenario (concat pooling)", fontsize=10, loc="left")
    axes.annotate("solid, filled = MLP-512\ndashed, open = logistic",
                  xy=(0.03, 0.05), xycoords="axes fraction", color="#6f6e64",
                  fontsize=8.5, linespacing=1.4)


def plot_holdout_panel(axes: plt.Axes, holdout_frame: pd.DataFrame) -> None:
    """
    Dumbbell chart of balanced accuracy per held-out Qwen organism at layer
    46 concat: gray open circle = logistic, aqua filled circle = MLP.

    :param axes: Target axes.
    :param holdout_frame: holdout_results.csv rows.
    """
    qwen = holdout_frame[holdout_frame.base_model == "qwen"]
    organisms = qwen.holdout_organism.fillna("None (base)")
    pivot = qwen.assign(holdout_organism=organisms).pivot_table(
        index="holdout_organism", columns="probe", values="balanced_accuracy")
    pivot = pivot.sort_values("mlp-512")
    positions = range(len(pivot))
    for position, (organism, row) in zip(positions, pivot.iterrows()):
        axes.plot([row["logistic"], row["mlp-512"]], [position, position],
                  color="#c3c2b7", linewidth=1.4, zorder=2)
        axes.plot(row["logistic"], position, marker="o", markersize=6.5,
                  markerfacecolor="white", markeredgecolor=SERIES_COLORS["gray"],
                  markeredgewidth=1.6, linestyle="none", zorder=3)
        axes.plot(row["mlp-512"], position, marker="o", markersize=6.5,
                  color=SERIES_COLORS["aqua"], linestyle="none", zorder=4)
    axes.set_yticks(list(positions))
    axes.set_yticklabels(pivot.index)
    axes.set_xlim(0.82, 1.0)
    axes.set_xlabel("balanced accuracy on the held-out organism")
    axes.xaxis.grid(True, color="#e5e4dc", linewidth=0.8)
    axes.set_axisbelow(True)
    axes.set_title("Unseen organism, layer 46 concat", fontsize=10, loc="left")
    top_row = pivot.iloc[-1]
    top_position = len(pivot) - 1
    probe_pair = sorted([("logistic", SERIES_COLORS["gray"], top_row["logistic"]),
                         ("MLP-512", SERIES_COLORS["aqua"], top_row["mlp-512"])],
                        key=lambda item: item[2])
    for (name, color, value), alignment, nudge in zip(probe_pair, ("right", "left"),
                                                      (-0.006, 0.006)):
        axes.annotate(name, xy=(value + nudge, top_position), color=color, fontsize=9,
                      fontweight="bold", ha=alignment, va="center")


def main() -> None:
    """
    Render the two-panel probe comparison and save PNG + SVG next to the
    Phase 2 result CSVs; copy them into figures/ when refreshing the README.
    """
    result_dir = REPO / "results/whitebox/nonlinear_sweep"
    cross_frame = pd.read_csv(result_dir / "cross_results.csv")
    holdout_frame = pd.read_csv(result_dir / "holdout_results.csv")

    figure, (cross_axes, holdout_axes) = plt.subplots(
        1, 2, figsize=(10.6, 4.6), dpi=160, gridspec_kw={"width_ratios": [1.1, 1]})
    plot_cross_panel(cross_axes, cross_frame)
    plot_holdout_panel(holdout_axes, holdout_frame)

    figure.suptitle("MLP vs logistic probes on Qwen3.5-27B activations",
                    fontsize=12, x=0.065, y=1.06, ha="left")
    figure.text(0.065, 0.965,
                "Left: trained on all datasets of one scenario, tested on all datasets of"
                " the other. Right: trained on all datasets of both scenarios\nexcept the"
                " held-out organism's, tested on that organism — every accuracy averages"
                " over the test datasets. Both probes read the\nsame cached activations"
                " (concat of mean- and last-token pooling over the response).",
                fontsize=8.5, color="#6f6e64", va="top", linespacing=1.35)

    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(result_dir / "probe_comparison.png", bbox_inches="tight")
    figure.savefig(result_dir / "probe_comparison.svg", bbox_inches="tight")
    print("wrote", result_dir / "probe_comparison.png")


if __name__ == "__main__":
    main()
