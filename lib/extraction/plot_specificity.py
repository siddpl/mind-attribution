"""
lib/extraction/plot_specificity.py — figures for the specificity battery.

Four figures, one file each, no titles baked in (captions belong in the paper),
legends outside the plot area, consistent identity colors throughout:

    mind      blue      the direction under test
    placebo   orange    the subject-matched non-mental contrast
    negation  aqua      the named suspect for what they share
    BASELINE  ink       cos(placebo_A, placebo_B) — ambient alignment

The baseline is deliberately NOT a categorical hue. It is not a fourth series
to compare against the others; it is the zero-line every other cosine has to
be read against, so it wears neutral ink and a heavy dash.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MIND = "#2a78d6"
PLACEBO = "#eb6834"
NEGATION = "#1baf7a"
BASELINE = "#52514e"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"

DPI = 150


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)


def _legend_outside(ax, ncol: int = 1):
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
                    fontsize=8.5, ncol=ncol)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def _mark_layer(ax, layer: int) -> None:
    ax.axvline(layer, color=INK_MUTED, linewidth=0.9, linestyle=(0, (1, 2)),
               alpha=0.7, zorder=1)


def fig_accuracy_by_layer(layers, mind, placebo, negation, band, selected, out_path) -> Path:
    """Accuracy of three frozen directions on the SAME mind held-out items."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    # the gap is the result; shade it so it reads before any individual line
    ax.fill_between(layers, placebo, mind, where=np.array(mind) >= np.array(placebo),
                    color=MIND, alpha=0.10, zorder=1, label="mind − placebo gap")
    ax.axhline(band, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.text(0.01, band, f"chance band {band:.3f}", color=INK_MUTED, fontsize=8,
            va="bottom", ha="left", transform=ax.get_yaxis_transform(), zorder=2)
    _mark_layer(ax, selected)

    for y, c, lab, dash in ((mind, MIND, "Mind direction", None),
                            (placebo, PLACEBO, "Placebo direction", (5, 2.5)),
                            (negation, NEGATION, "Negation direction", (1, 2.2))):
        line, = ax.plot(layers, y, color=c, linewidth=2.0, marker="o", markersize=3.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.0, label=lab, zorder=3)
        if dash:
            line.set_dashes(dash)

    ax.set_xlabel("Layer (resid_post)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("Accuracy on mind held-out items", color=INK_MUTED, fontsize=9)
    ax.set_xticks(layers[::2])
    _legend_outside(ax)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def fig_margin_and_effect_size(layers, margin, d_mind, d_placebo, selected, out_path) -> Path:
    """Is mind-specificity LOCALIZED, or does everything rise together?

    Top panel is the mind-specific component: accuracy the mind direction has
    that the placebo does not. A step here means structure appears at a layer;
    a flat line while both accuracies climb means the climb is generic.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.4), dpi=DPI, sharex=True,
                             gridspec_kw={"hspace": 0.22})
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    _style(ax)
    ax.axhline(0.0, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.text(0.01, 0.0, "zero — no mind-specific advantage", color=INK_MUTED,
            fontsize=8, va="bottom", ha="left", transform=ax.get_yaxis_transform())
    _mark_layer(ax, selected)
    ax.fill_between(layers, 0, margin, color=MIND, alpha=0.12, zorder=1)
    ax.plot(layers, margin, color=MIND, linewidth=2.0, marker="o", markersize=3.5,
            markeredgecolor=SURFACE, markeredgewidth=1.0, label="margin", zorder=3)
    ax.set_ylabel("Margin (mind − placebo)", color=INK_MUTED, fontsize=9)
    _legend_outside(ax)

    ax = axes[1]
    _style(ax)
    _mark_layer(ax, selected)
    ax.plot(layers, d_mind, color=MIND, linewidth=2.0, marker="o", markersize=3.5,
            markeredgecolor=SURFACE, markeredgewidth=1.0, label="Mind direction", zorder=3)
    line, = ax.plot(layers, d_placebo, color=PLACEBO, linewidth=2.0, marker="o",
                    markersize=3.5, markeredgecolor=SURFACE, markeredgewidth=1.0,
                    label="Placebo direction", zorder=3)
    line.set_dashes((5, 2.5))
    ax.set_ylabel("Cohen's d on mind items", color=INK_MUTED, fontsize=9)
    ax.set_xlabel("Layer (resid_post)", color=INK_MUTED, fontsize=9)
    ax.set_xticks(layers[::2])
    _legend_outside(ax)

    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def fig_direction_overlap(layers, mind_placebo, mind_negation, placebo_negation,
                          baseline, selected, out_path) -> Path:
    """THE anisotropy argument. Read every cosine relative to the dashed baseline.

    cos(placebo_A, placebo_B) is two disjoint halves of the SAME non-mental
    stimulus set. They share no content with each other, so whatever they show
    is what the layer's geometry produces on its own.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.axhline(0.0, color=GRID, linewidth=1.0, zorder=1)
    _mark_layer(ax, selected)

    # baseline first and heaviest — it is the reference, not a competitor
    ax.plot(layers, baseline, color=BASELINE, linewidth=2.6, linestyle=(0, (6, 2.5)),
            marker="s", markersize=3.5, markeredgecolor=SURFACE, markeredgewidth=1.0,
            label="BASELINE: cos(placebo A, placebo B)\n(ambient alignment, no shared content)",
            zorder=4)
    for y, c, lab, dash in (
        (mind_placebo, MIND, "cos(mind, placebo)", None),
        (mind_negation, PLACEBO, "cos(mind, negation)", (5, 2.5)),
        (placebo_negation, NEGATION, "cos(placebo, negation)", (1, 2.2)),
    ):
        line, = ax.plot(layers, y, color=c, linewidth=2.0, marker="o", markersize=3.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.0, label=lab, zorder=3)
        if dash:
            line.set_dashes(dash)

    ax.set_xlabel("Layer (resid_post)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("Cosine similarity", color=INK_MUTED, fontsize=9)
    ax.set_xticks(layers[::2])
    _legend_outside(ax)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def fig_projection_distributions(proj_mind, proj_placebo, labels, thr_mind,
                                 thr_placebo, affirm_label, layer, out_path) -> Path:
    """What 0.923 and 0.714 actually look like. Accuracy hides distribution shape."""
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), dpi=DPI,
                             gridspec_kw={"hspace": 0.35})
    fig.patch.set_facecolor(SURFACE)

    for ax, proj, thr, name in ((axes[0], proj_mind, thr_mind, "Mind direction"),
                                (axes[1], proj_placebo, thr_placebo, "Placebo direction")):
        _style(ax)
        aff = proj[labels == affirm_label]
        den = proj[labels != affirm_label]
        bins = np.histogram_bin_edges(proj, bins=28)
        ax.hist(aff, bins=bins, color=MIND, alpha=0.62, label="affirm", zorder=3)
        ax.hist(den, bins=bins, color=PLACEBO, alpha=0.62, label="deny", zorder=3)
        ax.axvline(thr, color=INK, linewidth=1.6, linestyle=(0, (4, 2)), zorder=4,
                   label=f"frozen threshold {thr:+.2f}")
        ax.set_ylabel("items", color=INK_MUTED, fontsize=9)
        ax.set_xlabel(f"projection onto {name} (L{layer})", color=INK_MUTED, fontsize=9)
        _legend_outside(ax)

    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)
