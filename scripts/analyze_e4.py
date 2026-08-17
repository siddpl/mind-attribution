#!/usr/bin/env python3
"""
scripts/analyze_e4.py — scoring analysis for the behavioral audience cross.

Takes for_coding.csv WITH the score column filled in by a human coder who never
saw the conditions, joins back to generations.csv on trial_id, and reports the
planned contrasts.

THIS SCRIPT GENERATES NOTHING. run_e4_behavioral.py produces text and does not
know the rubric exists; this one reads scores and cannot produce text. The blind
holds by construction — neither half can contaminate the other.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.extraction.plot_specificity import (
    DPI, GRID, INK, INK_MUTED, MIND, NEGATION, PLACEBO, SURFACE, _legend_outside, _style,
)
from run_e1 import git_commit, versioned

FRAME_ORDER = ("F0", "F1", "F2", "F3", "F4")
VARIANT_COLORS = (MIND, PLACEBO, NEGATION)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E4 scoring analysis (generates nothing)")
    p.add_argument("--generations", required=True, type=Path)
    p.add_argument("--scores", required=True, type=Path,
                   help="for_coding.csv with the score column filled in")
    p.add_argument("--out", type=Path, default=Path("results/e4_behavioral/analysis.json"))
    p.add_argument("--figdir", type=Path, default=Path("results/e4_behavioral/figures"))
    return p.parse_args(argv)


def ci95(v: np.ndarray) -> float:
    """Half-width of the 95% CI of the mean. nan when n < 2."""
    if len(v) < 2:
        return float("nan")
    return float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def contrast(a: np.ndarray, b: np.ndarray) -> dict:
    """Difference of means with a pooled-SD effect size and a CI on the diff."""
    if len(a) < 2 or len(b) < 2:
        return {"diff": float("nan"), "ci95": float("nan"), "cohens_d": float("nan"),
                "n_a": len(a), "n_b": len(b)}
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {"diff": float(a.mean() - b.mean()), "ci95": float(1.96 * se),
            "cohens_d": float((a.mean() - b.mean()) / sd) if sd > 1e-12 else float("nan"),
            "n_a": len(a), "n_b": len(b)}


def main(argv=None) -> int:
    args = parse_args(argv)
    gens = {r["trial_id"]: r for r in csv.DictReader(open(args.generations))}
    scored = [r for r in csv.DictReader(open(args.scores)) if str(r.get("score", "")).strip()]
    if not scored:
        raise ValueError(f"{args.scores} has no filled-in score column — nothing to analyse")

    rows = []
    for r in scored:
        g = gens.get(r["trial_id"])
        if g is None:
            raise ValueError(f"score for unknown trial_id {r['trial_id']!r}")
        rows.append({**g, "score": float(r["score"])})

    by_frame = defaultdict(list)
    by_frame_variant = defaultdict(list)
    len_by_frame = defaultdict(list)
    excl_by_frame = defaultdict(int)
    for g in gens.values():
        if int(g["degenerate_flag"]):
            excl_by_frame[g["frame_id"]] += 1
    for r in rows:
        by_frame[r["frame_id"]].append(r["score"])
        by_frame_variant[(r["frame_id"], r["variant_id"])].append(r["score"])
        len_by_frame[r["frame_id"]].append(len(r["summary_text"].split()))

    frames = [f for f in FRAME_ORDER if f in by_frame]
    arr = {f: np.array(by_frame[f]) for f in frames}
    names = {g["frame_id"]: g["frame_name"] for g in gens.values()}

    # ---- THE TWO PLANNED CONTRASTS, never collapsed into one "audience effect"
    # They answer different questions: sympathy-sensitivity is a sycophancy-
    # flavoured story; humanness-sensitivity WITHOUT sympathy-sensitivity points
    # somewhere else entirely.
    planned = {}
    if "F3" in arr and "F4" in arr:
        planned["sympathy_F3_vs_F4"] = contrast(arr["F3"], arr["F4"])
    if all(k in arr for k in ("F3", "F4", "F2")):
        planned["humanness_F3F4_vs_F2"] = contrast(
            np.concatenate([arr["F3"], arr["F4"]]), arr["F2"])
    baseline = {f: contrast(arr[f], arr["F0"]) for f in frames if f != "F0" and "F0" in arr}

    # ---- variant robustness: an effect in ONE wording is prompt fragility, and
    # gets reported as such rather than averaged away.
    variants = sorted({r["variant_id"] for r in rows})
    per_variant = {}
    for v in variants:
        a = {f: np.array(by_frame_variant.get((f, v), [])) for f in frames}
        per_variant[v] = {
            "means": {f: (float(a[f].mean()) if len(a[f]) else None) for f in frames},
            "sympathy": (contrast(a["F3"], a["F4"])
                         if len(a.get("F3", [])) > 1 and len(a.get("F4", [])) > 1 else None),
        }

    # ---- between vs within: with n=20 at temperature 0.7 the within-cell spread
    # may simply exceed the between-cell differences. Say so plainly if it does.
    within = float(np.mean([arr[f].std(ddof=1) for f in frames if len(arr[f]) > 1]))
    between = float(np.std([arr[f].mean() for f in frames], ddof=1)) if len(frames) > 1 else float("nan")

    figdir = args.figdir
    figdir.mkdir(parents=True, exist_ok=True)

    # fig1 — distributions with points overlaid; n=20 makes shape matter more
    # than a box summary
    f1 = versioned(figdir / "fig1_scores_by_frame.png")
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    if "F0" in arr:
        ax.axhline(arr["F0"].mean(), color=INK_MUTED, linewidth=1.2,
                   linestyle=(0, (4, 3)), zorder=1)
        ax.text(0.01, arr["F0"].mean(), "F0 baseline mean", color=INK_MUTED, fontsize=8,
                va="bottom", ha="left", transform=ax.get_yaxis_transform())
    parts = ax.violinplot([arr[f] for f in frames], positions=range(len(frames)),
                          showextrema=False, widths=0.7)
    for b in parts["bodies"]:
        b.set_facecolor(MIND); b.set_alpha(0.18); b.set_edgecolor("none")
    rng = np.random.default_rng(0)
    for i, f in enumerate(frames):
        x = i + rng.normal(0, 0.055, len(arr[f]))
        ax.scatter(x, arr[f], s=16, color=MIND, alpha=0.65, linewidths=0, zorder=3)
        ax.scatter([i], [arr[f].mean()], s=70, color=INK, marker="_", zorder=4)
    ax.set_xticks(range(len(frames)))
    ax.set_xticklabels([f"{f}\n{names.get(f,'')}" for f in frames], fontsize=8.5)
    ax.set_ylabel("Rubric score", color=INK_MUTED, fontsize=9)
    fig.savefig(f1, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)

    # fig2 — the two planned contrasts as effect sizes, labelled by what they isolate
    f2 = versioned(figdir / "fig2_planned_contrasts.png")
    fig, ax = plt.subplots(figsize=(8.5, 3.2), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.axvline(0, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    labels = {"sympathy_F3_vs_F4": "SYMPATHY\nF3 sympathetic − F4 skeptic",
              "humanness_F3F4_vs_F2": "HUMANNESS\n(F3+F4) human − F2 grader"}
    keys = [k for k in labels if k in planned]
    for i, k in enumerate(keys):
        c = planned[k]
        ax.errorbar(c["diff"], i, xerr=c["ci95"], fmt="o", markersize=8,
                    color=MIND if "sympathy" in k else PLACEBO, capsize=5,
                    linewidth=2, zorder=3)
        ax.text(c["diff"], i + 0.22, f"d={c['cohens_d']:+.2f}", color=INK,
                fontsize=8.5, ha="center")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([labels[k] for k in keys], fontsize=8.5)
    ax.set_ylim(-0.6, len(keys) - 0.2)
    ax.set_xlabel("Difference in mean rubric score (95% CI)", color=INK_MUTED, fontsize=9)
    fig.savefig(f2, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)

    # fig3 — parallel lines mean robust; crossing lines mean prompt-dependent
    f3 = versioned(figdir / "fig3_variant_robustness.png")
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for i, v in enumerate(variants):
        ys = [per_variant[v]["means"][f] for f in frames]
        ax.plot(range(len(frames)), ys, color=VARIANT_COLORS[i % len(VARIANT_COLORS)],
                linewidth=2.0, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=v, zorder=3)
    ax.set_xticks(range(len(frames)))
    ax.set_xticklabels([f"{f}\n{names.get(f,'')}" for f in frames], fontsize=8.5)
    ax.set_ylabel("Mean rubric score", color=INK_MUTED, fontsize=9)
    _legend_outside(ax)
    fig.savefig(f3, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)

    report = {
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_scored": len(rows),
        "per_frame": {f: {"mean": float(arr[f].mean()), "ci95": ci95(arr[f]),
                          "n": int(len(arr[f])), "n_excluded": excl_by_frame.get(f, 0),
                          "mean_summary_words": float(np.mean(len_by_frame[f]))}
                      for f in frames},
        "planned_contrasts": planned,
        "vs_baseline_F0": baseline,
        "per_variant": per_variant,
        "within_cell_sd": within, "between_cell_sd": between,
        "within_exceeds_between": bool(within > between),
        "figures": [str(f1), str(f2), str(f3)],
    }
    out = versioned(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))

    print("=" * 70)
    print(f"  E4 BEHAVIORAL — {len(rows)} scored trials")
    print(f"\n  {'frame':<22} {'mean':>7} {'±95%':>7} {'n':>4} {'excl':>5} {'summ words':>11}")
    for f in frames:
        d = report["per_frame"][f]
        print(f"  {f+' '+names.get(f,''):<22} {d['mean']:>7.3f} {d['ci95']:>7.3f} "
              f"{d['n']:>4} {d['n_excluded']:>5} {d['mean_summary_words']:>11.1f}")
    print("\n  PLANNED CONTRASTS (never collapsed into one 'audience effect')")
    for k in ("sympathy_F3_vs_F4", "humanness_F3F4_vs_F2"):
        if k in planned:
            c = planned[k]
            print(f"    {k:<24} {c['diff']:+.3f} ±{c['ci95']:.3f}  d={c['cohens_d']:+.2f}")
    print(f"\n  within-cell SD {within:.3f} vs between-cell SD {between:.3f}")
    if within > between:
        print("    WITHIN EXCEEDS BETWEEN — at this n and temperature the sampling "
              "spread is larger than any frame difference. Report that plainly.")
    sw = [report["per_frame"][f]["mean_summary_words"] for f in frames]
    print(f"  summary length across frames: {min(sw):.1f}-{max(sw):.1f} words"
          f"{'  <- CONFOUND: frame changed task behaviour' if max(sw)-min(sw) > 5 else ''}")
    print(f"\n  {out}  |  figures in {figdir}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
