#!/usr/bin/env python3
"""
scripts/run_e2_sweep.py — the H1 transfer test at EVERY layer.

run_e2.py answers the question at the one layer E1 selected. That is the
preregistered test and it stays the headline. But a verdict resting on a single
layer cannot distinguish "the self-content is not on this axis" from "L23 in
particular happens not to carry it", so this sweeps the same measurement down
the stack.

STILL NOTHING IS FITTED ON FIRST-PERSON DATA. At each layer the direction is
extracted from THIRD-PERSON training items and the threshold is fit on those
same third-person items — identical to what E1 does at its selected layer. The
first-person and mirror sets are only ever measured, never fitted to.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.extraction.diff_means import extract_direction, project
from lib.extraction.plot_specificity import (
    DPI, GRID, INK, INK_MUTED, MIND, NEGATION, PLACEBO, SURFACE, _legend_outside,
    _mark_layer, _style,
)
from lib.probes.linear_probe import (
    AFFIRM, DENY, ceiling_probe_accuracy, chance_band, direction_probe_accuracy,
    fit_threshold,
)
from run_e1 import git_commit, load_dataset, versioned

MIRROR = "#4a3aa7"  # categorical slot 7 — a fourth identity, not a re-used hue


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E2 transfer test across all layers")
    p.add_argument("--mind-hash", default="6bfee5fdf7cf")
    p.add_argument("--mind-stimuli", type=Path,
                   default=Path("data/contrast_pairs/contrast_pairs_generated.csv"))
    p.add_argument("--placebo-hash", default="1a4267a2af9c")
    p.add_argument("--placebo-stimuli", type=Path, default=Path("data/placebo/placebo.csv"))
    p.add_argument("--first-person-hash", default="7c4c7068563d")
    p.add_argument("--first-person-stimuli", type=Path,
                   default=Path("data/first_person/first_person.csv"))
    p.add_argument("--mirror-hash", default="97fbfddc8c42")
    p.add_argument("--mirror-stimuli", type=Path, default=Path("data/mirror/mirror.csv"))
    p.add_argument("--schema", type=Path, default=Path("data/schema.json"))
    p.add_argument("--selfref-schema", type=Path, default=Path("data/schema_selfref.json"))
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--position", default="claim_end")
    p.add_argument("--layer", type=int, default=23)
    p.add_argument("--alpha-sd", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=Path("results/e2_sweep.json"))
    p.add_argument("--figdir", type=Path, default=Path("results/figures"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    tp = SimpleNamespace(model=args.model, position=args.position, schema=args.schema)
    sr = SimpleNamespace(model=args.model, position=args.position, schema=args.selfref_schema)

    mind = load_dataset(tp, args.mind_hash, args.mind_stimuli, "mind")
    plac = load_dataset(tp, args.placebo_hash, args.placebo_stimuli, "placebo")
    fp = load_dataset(sr, args.first_person_hash, args.first_person_stimuli, "first_person")
    mir = load_dataset(tp, args.mirror_hash, args.mirror_stimuli, "mirror")

    layers = sorted(mind["acts_by_layer"])
    y_m, y_p = mind["labels"], plac["labels"]
    y_fp, y_mi = fp["labels"], mir["labels"]
    band_fp = chance_band(len(y_fp), args.alpha_sd)
    band_mi = chance_band(len(y_mi), args.alpha_sd)

    s = {k: [] for k in ("fp_dir", "fp_ceiling", "mirror_dir", "mirror_ceiling", "fp_placebo")}
    for L in layers:
        a_m, a_p = mind["acts_by_layer"][L], plac["acts_by_layer"][L]
        a_fp, a_mi = fp["acts_by_layer"][L], mir["acts_by_layer"][L]
        d_m = extract_direction(a_m[y_m == AFFIRM], a_m[y_m == DENY])
        d_p = extract_direction(a_p[y_p == AFFIRM], a_p[y_p == DENY])
        t_m = fit_threshold(a_m, y_m, d_m)
        t_p = fit_threshold(a_p, y_p, d_p)
        s["fp_dir"].append(direction_probe_accuracy(a_fp, y_fp, d_m, t_m))
        s["fp_placebo"].append(direction_probe_accuracy(a_fp, y_fp, d_p, t_p))
        s["mirror_dir"].append(direction_probe_accuracy(a_mi, y_mi, d_m, t_m))
        s["fp_ceiling"].append(ceiling_probe_accuracy(a_fp, y_fp))
        s["mirror_ceiling"].append(ceiling_probe_accuracy(a_mi, y_mi))

    figdir = args.figdir
    figdir.mkdir(parents=True, exist_ok=True)

    # ---- fig5: the transfer test at every layer ---------------------------
    f5 = versioned(figdir / "fig5_e2_transfer_by_layer.png")
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.axhline(band_fp, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.text(0.01, band_fp, f"chance band {band_fp:.3f}", color=INK_MUTED, fontsize=8,
            va="bottom", ha="left", transform=ax.get_yaxis_transform(), zorder=2)
    _mark_layer(ax, args.layer)
    for y, c, lab, dash in (
        (s["mirror_dir"], MIRROR, "Mirror (referent swap only)", None),
        (s["fp_ceiling"], NEGATION, "Ceiling on first-person", (1, 2.2)),
        (s["fp_dir"], MIND, "Mind direction on first-person", None),
        (s["fp_placebo"], PLACEBO, "Placebo on first-person", (5, 2.5)),
    ):
        line, = ax.plot(layers, y, color=c, linewidth=2.0, marker="o", markersize=3.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.0, label=lab, zorder=3)
        if dash:
            line.set_dashes(dash)
    ax.set_xlabel("Layer (resid_post)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("Accuracy (frozen direction)", color=INK_MUTED, fontsize=9)
    ax.set_xticks(layers[::2])
    _legend_outside(ax)
    fig.savefig(f5, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ---- fig6: the two self-reference sets, side by side, at the selected layer
    f6 = versioned(figdir / "fig6_e2_projection_split.png")
    L = args.layer
    a_m = mind["acts_by_layer"][L]
    d_m = extract_direction(a_m[y_m == AFFIRM], a_m[y_m == DENY])
    t_m = fit_threshold(a_m, y_m, d_m)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), dpi=DPI,
                             gridspec_kw={"hspace": 0.38})
    fig.patch.set_facecolor(SURFACE)
    for ax, ds, y, name in ((axes[0], mir, y_mi, "Mirror — referent swapped to \"I\""),
                            (axes[1], fp, y_fp, "First-person — experiential vs mundane")):
        _style(ax)
        proj = project(ds["acts_by_layer"][L], d_m)
        bins = np.histogram_bin_edges(proj, bins=24)
        ax.hist(proj[y == AFFIRM], bins=bins, color=MIND, alpha=0.62,
                label="experiential / affirm", zorder=3)
        ax.hist(proj[y == DENY], bins=bins, color=PLACEBO, alpha=0.62,
                label="mundane / deny", zorder=3)
        ax.axvline(t_m, color=INK, linewidth=1.6, linestyle=(0, (4, 2)), zorder=4,
                   label=f"frozen threshold {t_m:+.1f}")
        ax.set_ylabel("items", color=INK_MUTED, fontsize=9)
        ax.set_xlabel(f"{name} — projection onto the L{L} mind direction",
                      color=INK_MUTED, fontsize=9)
        _legend_outside(ax)
    fig.savefig(f6, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    i = layers.index(args.layer)
    report = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layers": layers, "series": s,
        "band_first_person": band_fp, "band_mirror": band_mi,
        "at_selected_layer": {k: v[i] for k, v in s.items()},
        "n_layers_fp_above_band": int(sum(1 for v in s["fp_dir"] if v > band_fp)),
        "n_layers_mirror_above_band": int(sum(1 for v in s["mirror_dir"] if v > band_mi)),
        "figures": [str(f5), str(f6)],
    }
    out = versioned(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))

    print("=" * 72)
    print(f"  E2 TRANSFER SWEEP — {args.model} @ {args.position}")
    print(f"  first-person band {band_fp:.3f} | mirror band {band_mi:.3f}")
    print(f"  first-person above band in {report['n_layers_fp_above_band']}/{len(layers)} layers"
          f"  (max {max(s['fp_dir']):.3f} at L{layers[int(np.argmax(s['fp_dir']))]})")
    print(f"  mirror       above band in {report['n_layers_mirror_above_band']}/{len(layers)} layers"
          f"  (max {max(s['mirror_dir']):.3f} at L{layers[int(np.argmax(s['mirror_dir']))]})")
    print(f"  ceiling on first-person: min {min(s['fp_ceiling']):.3f} max {max(s['fp_ceiling']):.3f}")
    print(f"  at L{args.layer}: fp {s['fp_dir'][i]:.3f} | mirror {s['mirror_dir'][i]:.3f} "
          f"| fp ceiling {s['fp_ceiling'][i]:.3f}")
    print(f"  {out}  |  {f5.name}  {f6.name}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
