"""
lib/extraction/plot_sweep.py — the layer sweep, drawn.

One figure, one question: at which layer does the mind direction separate
held-out templates, and does the placebo direction do the same thing there?

Colors are the reference categorical palette's first three slots, used
unmodified — that trio is documented as validating all-pairs in both light and
dark modes (CVD dE 9.2 light / 9.4 dark). Do not re-step them by hand.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a capture box
import matplotlib.pyplot as plt

# Categorical slots 1-3. Identity, assigned in fixed order, never cycled.
MIND = "#2a78d6"      # slot 1, blue
PLACEBO = "#eb6834"   # slot 2, orange
CEILING = "#1baf7a"   # slot 3, aqua

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"


def plot_layer_sweep(
    core_sweep: list[dict],
    placebo_on_mind: dict[int, float],
    band: float,
    best_layer: int,
    out_path: str | Path,
    title: str = "Layer sweep — held-out accuracy",
    subtitle: str = "",
) -> Path:
    """Draw accuracy against layer for mind, ceiling, and placebo-on-mind; mark the argmax.

    THE PLACEBO SERIES IS placebo-direction-scored-on-MIND-items, not the
    placebo's accuracy on its own data. The latter is expected to be high and
    carries no information about the mind result; this one must sit at chance.

    Series carry distinct dash patterns and the most important one is drawn
    FIRST, so that where curves coincide exactly — common when a sweep
    saturates — the ones on top remain visible instead of being painted over.
    The chance band is a reference line in muted ink, not a fourth series.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def series(sweep, key):
        pts = [(s["layer"], s.get(key)) for s in sweep if "error" not in s]
        return ([p[0] for p in pts if p[1] is not None],
                [p[1] for p in pts if p[1] is not None])

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    # chance band: reference, deliberately recessive
    ax.axhline(band, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    # axes-fraction x so the label stays put when xlim changes below
    ax.text(0.01, band, f"chance band {band:.3f}", color=INK_MUTED, fontsize=8,
            va="bottom", ha="left", zorder=1, transform=ax.get_yaxis_transform())

    placebo_x = sorted(placebo_on_mind)
    placebo_series = (placebo_x, [placebo_on_mind[k] for k in placebo_x])
    end_labels: list[tuple[float, float, str]] = []

    # Drawn in order: the headline series first, controls over the top of it, so
    # exact overlap still reads. Dash patterns keep them separable in print and
    # for a reader who cannot rely on hue.
    for (x, y), color, label, dashes, z in (
        (series(core_sweep, "heldout_accuracy"), MIND, "Mind direction", None, 3),
        (series(core_sweep, "ceiling_accuracy"), CEILING, "Ceiling (trained LR)", (1, 2.2), 4),
        (placebo_series, PLACEBO, "Placebo direction on mind items", (5, 2.5), 5),
    ):
        if not x:
            continue
        line, = ax.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=4.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=z)
        if dashes:
            line.set_dashes(dashes)
        end_labels.append((x[-1], y[-1], f"{label}  {y[-1]:.3f}"))

    # the argmax — the preregistered selection, called out directly
    best = next((s for s in core_sweep if s.get("layer") == best_layer), None)
    if best and "error" not in best:
        acc = best["heldout_accuracy"]
        ax.plot([best_layer], [acc], marker="o", markersize=9, color=MIND,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        ax.annotate(f"argmax: L{best_layer}  {acc:.3f}",
                    xy=(best_layer, acc), xytext=(6, 10), textcoords="offset points",
                    color=INK, fontsize=9, fontweight="bold", zorder=5)

    # room on the right for the direct labels
    if layers_present := [s["layer"] for s in core_sweep if "error" not in s]:
        ax.set_xlim(min(layers_present) - 0.4, max(layers_present) + 5.2)

    # Direct labels, required relief: validate_palette flags the aqua step at
    # 2.74:1 on this surface (below 3:1), and a contrast WARN obligates visible
    # labels or a table view. Text wears ink, never the series color.
    # Coincident series would print their labels on top of each other, so nudge
    # them apart vertically — a saturated sweep makes exact ties the norm.
    min_gap = 0.045 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    placed: list[float] = []
    for x_end, y_end, text in sorted(end_labels, key=lambda t: -t[1]):
        y_label = y_end
        for prior in placed:
            if abs(y_label - prior) < min_gap:
                y_label = prior - min_gap
        placed.append(y_label)
        # A leader line only when the label had to move, so the reader can still
        # tell which curve it belongs to.
        arrow = None
        if y_label != y_end:
            arrow = dict(arrowstyle="-", color=GRID, linewidth=0.8, shrinkA=0, shrinkB=2)
        ax.annotate(text, xy=(x_end, y_end), xytext=(x_end + 0.35, y_label),
                    textcoords="data", color=INK, fontsize=8.5, va="center",
                    ha="left", zorder=6, annotation_clip=False, arrowprops=arrow)

    ax.set_xlabel("Layer (resid_post)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("Held-out accuracy", color=INK_MUTED, fontsize=9)
    ax.set_ylim(0.0, 1.08)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    layers = [s["layer"] for s in core_sweep if "error" not in s]
    if layers:
        ax.set_xticks(layers)

    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left", pad=18)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_MUTED,
                fontsize=8.5, va="bottom", ha="left")

    legend = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK)  # text wears ink, the swatch carries identity

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_layer_diagnostics(
    layers: list[int],
    mind_acc: list[float],
    placebo_on_mind: list[float],
    cohens_d: list[float],
    cos_mind_placebo: list[float],
    band: float,
    best_layer: int,
    out_path: str | Path,
    title: str = "Layer diagnostics",
    subtitle: str = "",
) -> Path:
    """Three measures against layer, as stacked panels sharing one x axis.

    Accuracy, effect size and cosine live on different scales, so they get
    separate panels rather than a second y axis — a dual-axis chart invites the
    reader to compare two curves whose vertical positions mean different
    things, which is exactly the wrong inference here.

    The bottom panel is the one to read: cos(mind, placebo) says how much the
    two rulers overlap. Near zero is what the design assumes; a rising curve
    means the "non-mental" contrast is drifting onto the same axis with depth.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), dpi=200, sharex=True,
                             gridspec_kw={"hspace": 0.28})
    fig.patch.set_facecolor(SURFACE)

    panels = [
        (axes[0], "Held-out accuracy", [(mind_acc, MIND, "Mind direction", None),
                                        (placebo_on_mind, PLACEBO, "Placebo on mind items", (5, 2.5))], band),
        (axes[1], "Cohen's d", [(cohens_d, MIND, "Mind direction", None)], None),
        (axes[2], "cos(mind, placebo)", [(cos_mind_placebo, CEILING, "Direction overlap", None)], 0.0),
    ]
    for ax, ylabel, series, ref in panels:
        ax.set_facecolor(SURFACE)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        if ref is not None:
            ax.axhline(ref, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
            label = f"chance band {ref:.3f}" if ref else "zero — no shared direction"
            ax.text(0.01, ref, label, color=INK_MUTED, fontsize=8, va="bottom",
                    ha="left", zorder=1, transform=ax.get_yaxis_transform())
        for values, color, label, dashes in series:
            line, = ax.plot(layers, values, color=color, linewidth=2.0, marker="o",
                            markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.2,
                            label=label, zorder=3)
            if dashes:
                line.set_dashes(dashes)
        ax.axvline(best_layer, color=INK_MUTED, linewidth=0.8, alpha=0.45, zorder=1)
        ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
        ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
        if len(series) > 1:
            leg = ax.legend(loc="lower right", frameon=False, fontsize=8.5)
            for t in leg.get_texts():
                t.set_color(INK)

    axes[0].set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left", pad=18)
    if subtitle:
        axes[0].text(0, 1.04, subtitle, transform=axes[0].transAxes, color=INK_MUTED,
                     fontsize=8.5, va="bottom", ha="left")
    axes[0].annotate(f"selected L{best_layer}", xy=(best_layer, max(mind_acc)),
                     xytext=(6, 6), textcoords="offset points", color=INK,
                     fontsize=8.5, fontweight="bold")
    axes[2].set_xlabel("Layer (resid_post)", color=INK_MUTED, fontsize=9)
    axes[2].set_xticks(layers[::2])

    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path
