"""
Layer curve figures for the probe sweeps: balanced accuracy vs layer.

Renders two variants: the Phase 1 figure (logistic probes only,
`layer_curve.*`) and the same figure with the Phase 2 MLP results overlaid
as dashed lines at layers 36-48 (`layer_curve_mlp.*`).
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

SERIES_COLORS = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100"}
REPO = Path(__file__).resolve().parents[2]


def render_layer_curve(linear_frame: pd.DataFrame, mlp_frame: pd.DataFrame | None,
                       out_dir: Path, out_stem: str) -> None:
    """
    Render one layer-curve figure and save it as PNG + SVG.

    :param linear_frame: Phase 1 sweep_results rows (logistic probes).
    :param mlp_frame: Phase 2 cross_results rows filtered to the MLP probe,
        overlaid as dashed lines on the Qwen series; None renders the plain
        Phase 1 figure.
    :param out_dir: Directory to write the image files to.
    :param out_stem: File name without extension.
    """
    pooled = linear_frame[linear_frame.pooling == "mean"]
    qwen_vi = pooled[(pooled.base_model == "qwen") & (pooled.train_scenario == "varied")]
    qwen_iv = pooled[(pooled.base_model == "qwen") & (pooled.train_scenario == "instructed")]
    gemma_cv = pooled[pooled.base_model == "gemma"]

    figure, axes = plt.subplots(figsize=(8.6, 4.8), dpi=160)

    axes.axhline(0.5, color="#9b9a8f", linewidth=1.2, linestyle=(0, (4, 4)), zorder=1)
    axes.annotate("chance", xy=(0.5, 0.503), color="#6f6e64", fontsize=9)

    axes.axvspan(36, 48, color="#26251f", alpha=0.05, zorder=0)
    axes.annotate("operating range\nL36–48", xy=(42, 0.545), color="#6f6e64",
                  fontsize=9, ha="center")

    series = [
        (qwen_vi.sort_values("layer"), SERIES_COLORS["blue"],
         "Qwen: trained on varied,\ntested on instructed", -0.036),
        (qwen_iv.sort_values("layer"), SERIES_COLORS["yellow"],
         "Qwen: trained on instructed,\ntested on varied", -0.004),
        (gemma_cv.sort_values("layer"), SERIES_COLORS["aqua"],
         "gemma: trained and tested\non instructed (5-fold CV)", 0.024),
    ]
    for frame, color, label, label_offset in series:
        axes.plot(frame.layer, frame.balanced_accuracy, color=color, linewidth=2,
                  marker="o", markersize=3.2, zorder=3)
        last_row = frame.iloc[-1]
        axes.annotate(label, xy=(last_row.layer + 1.2,
                                 last_row.balanced_accuracy + label_offset),
                      color=color, fontsize=9, fontweight="bold", va="center",
                      linespacing=1.25)

    if mlp_frame is not None:
        mlp_vi = mlp_frame[mlp_frame.train_scenario == "varied"]
        mlp_iv = mlp_frame[mlp_frame.train_scenario == "instructed"]
        for frame, color in [(mlp_vi.sort_values("layer"), SERIES_COLORS["blue"]),
                             (mlp_iv.sort_values("layer"), SERIES_COLORS["yellow"])]:
            axes.plot(frame.layer, frame.balanced_accuracy, color=color, linewidth=1.6,
                      linestyle=(0, (4, 3)), marker="o", markersize=3.2,
                      markerfacecolor="white", markeredgecolor=color, zorder=2)
        axes.annotate("solid = logistic probe, dashed = MLP-512 (gemma: logistic only)",
                      xy=(1, 0.462), color="#6f6e64", fontsize=8.5)

    axes.set_xlim(-1, 92)
    axes.set_ylim(0.45, 1.02)
    axes.set_xticks(range(0, 64, 8))
    axes.set_xlabel("decoder layer the probe reads from  (network depth: 0 = first, 63 = last)")
    axes.set_ylabel("balanced accuracy")
    axes.yaxis.grid(True, color="#e5e4dc", linewidth=0.8)
    axes.set_axisbelow(True)
    title = "Cross-scenario linear probe accuracy by layer" if mlp_frame is None \
        else "Cross-scenario probe accuracy by layer: logistic vs MLP"
    probe_word = "logistic regression" if mlp_frame is None else "probe"
    axes.set_title(title, fontsize=12, loc="left", y=1.16)
    axes.text(0, 1.025,
              f"Each point: one {probe_word} trained on mean-pooled response"
              " activations from all datasets of one scenario,\nscored on all"
              " datasets of the other ('trained on X, tested on Y'), so every"
              " point averages over the model organisms of the\ntest scenario.",
              transform=axes.transAxes, fontsize=8.5, color="#6f6e64", va="bottom",
              linespacing=1.35)

    figure.tight_layout()
    figure.savefig(out_dir / f"{out_stem}.png", bbox_inches="tight")
    figure.savefig(out_dir / f"{out_stem}.svg", bbox_inches="tight")
    plt.close(figure)
    print("wrote", out_dir / f"{out_stem}.png")


def main() -> None:
    """
    Render both layer-curve variants next to their sweep CSVs; copy them
    into figures/ when refreshing the README embeds.
    """
    linear_frame = pd.read_csv(REPO / "results/whitebox/linear_sweep/sweep_results.csv")
    mlp_frame = pd.read_csv(REPO / "results/whitebox/nonlinear_sweep/cross_results.csv")
    mlp_frame = mlp_frame[(mlp_frame.pooling == "mean") & (mlp_frame.probe == "mlp-512")]

    render_layer_curve(linear_frame, None,
                       REPO / "results/whitebox/linear_sweep", "layer_curve")
    render_layer_curve(linear_frame, mlp_frame,
                       REPO / "results/whitebox/nonlinear_sweep", "layer_curve_mlp")


if __name__ == "__main__":
    main()
