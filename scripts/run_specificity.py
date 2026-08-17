#!/usr/bin/env python3
"""
scripts/run_specificity.py — the specificity battery for E1.

ONE QUESTION: how much of the mind direction's separation is about minds, and
how much is shared with any contrast direction read at the same layer?

E1 passed Kill-1 (held-out 0.923, band 0.577) — but the placebo separates the
same items at 0.714, cos(mind, placebo) = +0.425, and both rise monotonically
with depth. Two readings are currently indistinguishable:

  (a) SHARED CONTENT — deep layers represent "asserted property" generically
      and mind is one instance. A real limit on the specificity claim.
  (b) AMBIENT ANISOTROPY — deep representations are anisotropic, so ANY two
      difference-of-means directions drift toward alignment regardless of
      content. An artifact of where we read, not what we found.

Part 1 builds the baseline that tells them apart. Nothing here re-caches; it
runs entirely on activations already on disk.

Glue only: every direction comes from extract_direction, every accuracy from
direction_probe_accuracy, on the same code path as E1.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.extraction.diff_means import cosine, extract_direction, project
from lib.extraction.plot_specificity import (
    fig_accuracy_by_layer,
    fig_direction_overlap,
    fig_margin_and_effect_size,
    fig_projection_distributions,
)
from lib.probes.linear_probe import (
    AFFIRM,
    DENY,
    chance_band,
    direction_probe_accuracy,
    fit_threshold,
    summarize_projections,
)
from run_e1 import git_commit, load_dataset, versioned

# The two disjoint halves of the placebo claim set. Same entities, templates
# and denial devices on both sides; only the asserted physical property
# differs — and the halves share no property with each other.
PLACEBO_HALF_A = ("durability", "speed", "weight")
PLACEBO_HALF_B = ("age", "visibility", "noise")  # noise carries prop "loudness"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E1 specificity battery")
    p.add_argument("--mind-hash", default="6bfee5fdf7cf")
    p.add_argument("--mind-stimuli", type=Path,
                   default=Path("data/contrast_pairs/contrast_pairs_generated.csv"))
    p.add_argument("--mind-heldout-hash", default="1d82440d3c1a")
    p.add_argument("--mind-heldout-stimuli", type=Path,
                   default=Path("data/contrast_pairs/contrast_pairs_heldout.csv"))
    p.add_argument("--placebo-hash", default="1a4267a2af9c")
    p.add_argument("--placebo-stimuli", type=Path, default=Path("data/placebo/placebo.csv"))
    p.add_argument("--placebo-heldout-hash", default="3eac7c59e34f")
    p.add_argument("--placebo-heldout-stimuli", type=Path,
                   default=Path("data/placebo/placebo_heldout.csv"))
    p.add_argument("--negation-hash", default="d49ce786bab2")
    p.add_argument("--negation-stimuli", type=Path,
                   default=Path("data/placebo/negation_control.csv"))
    p.add_argument("--schema", type=Path, default=Path("data/schema.json"))
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--position", default="claim_end")
    p.add_argument("--layer", type=int, default=None,
                   help="selected layer; defaults to E1's choice via --e1-report")
    p.add_argument("--e1-report", type=Path, default=Path("results/e1_primary.json"))
    p.add_argument("--alpha-sd", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=Path("results/specificity_report.json"))
    p.add_argument("--figdir", type=Path, default=Path("results/figures"))
    return p.parse_args(argv)


def _extract(acts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The ONLY way a direction is built in this script — including both placebo
    halves. A baseline built by a different code path is not a baseline."""
    return extract_direction(acts[labels == AFFIRM], acts[labels == DENY])


def main(argv=None) -> int:
    args = parse_args(argv)
    ctx = SimpleNamespace(model=args.model, position=args.position, schema=args.schema)

    def need(ds_hash, stimuli, name):
        # No re-caching: a missing set is a hard stop naming the hash, never a
        # silently skipped comparison.
        cache = Path("results/activation_cache") / args.model.replace("/", "_") / ds_hash
        if not (cache / "manifest.json").exists():
            raise FileNotFoundError(
                f"{name}: no cache for hash {ds_hash} at {cache}. This script does "
                f"not capture — run scripts/run_cache.py for {stimuli} first."
            )
        return load_dataset(ctx, ds_hash, Path(stimuli), name)

    mind = need(args.mind_hash, args.mind_stimuli, "mind")
    mind_ho = need(args.mind_heldout_hash, args.mind_heldout_stimuli, "mind_heldout")
    plac = need(args.placebo_hash, args.placebo_stimuli, "placebo")
    plac_ho = need(args.placebo_heldout_hash, args.placebo_heldout_stimuli, "placebo_heldout")
    neg = need(args.negation_hash, args.negation_stimuli, "negation")

    layer = args.layer
    if layer is None:
        if not args.e1_report.exists():
            raise FileNotFoundError(f"--layer not given and no E1 report at {args.e1_report}")
        layer = int(json.loads(args.e1_report.read_text())["core"]["best_layer"])

    layers = sorted(mind["acts_by_layer"])
    y_mind, y_mind_ho = mind["labels"], mind_ho["labels"]
    y_plac, y_plac_ho = plac["labels"], plac_ho["labels"]
    y_neg = neg["labels"]

    # placebo halves, by claim_id, from the TRAIN rows only
    claim_ids = np.array([r.get("claim_id") for r in plac["rows"]])
    mask_a = np.isin(claim_ids, PLACEBO_HALF_A)
    mask_b = np.isin(claim_ids, PLACEBO_HALF_B)
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        raise ValueError(
            f"placebo halves are empty (A={mask_a.sum()}, B={mask_b.sum()}); "
            f"claim_ids present: {sorted(set(claim_ids.tolist()))}"
        )

    band = chance_band(int(np.isin(y_mind_ho, [AFFIRM, DENY]).sum()), args.alpha_sd)
    series: dict[str, list] = {k: [] for k in (
        "mind_acc", "placebo_acc", "negation_acc", "negation_acc_on_placebo",
        "margin", "d_mind", "d_placebo",
        "cos_mind_placebo", "cos_mind_negation", "cos_placebo_negation", "cos_baseline",
    )}

    for L in layers:
        a_mind, a_mind_ho = mind["acts_by_layer"][L], mind_ho["acts_by_layer"][L]
        a_plac, a_plac_ho = plac["acts_by_layer"][L], plac_ho["acts_by_layer"][L]
        a_neg = neg["acts_by_layer"][L]

        d_mind = _extract(a_mind, y_mind)
        d_plac = _extract(a_plac, y_plac)
        d_neg = _extract(a_neg, y_neg)
        d_a = _extract(a_plac[mask_a], y_plac[mask_a])
        d_b = _extract(a_plac[mask_b], y_plac[mask_b])

        t_mind = fit_threshold(a_mind, y_mind, d_mind)
        t_plac = fit_threshold(a_plac, y_plac, d_plac)
        t_neg = fit_threshold(a_neg, y_neg, d_neg)

        series["mind_acc"].append(direction_probe_accuracy(a_mind_ho, y_mind_ho, d_mind, t_mind))
        series["placebo_acc"].append(direction_probe_accuracy(a_mind_ho, y_mind_ho, d_plac, t_plac))
        series["negation_acc"].append(direction_probe_accuracy(a_mind_ho, y_mind_ho, d_neg, t_neg))
        series["negation_acc_on_placebo"].append(
            direction_probe_accuracy(a_plac_ho, y_plac_ho, d_neg, t_neg))
        series["margin"].append(series["mind_acc"][-1] - series["placebo_acc"][-1])
        series["d_mind"].append(
            summarize_projections(project(a_mind_ho, d_mind), y_mind_ho)["cohens_d"])
        series["d_placebo"].append(
            summarize_projections(project(a_mind_ho, d_plac), y_mind_ho)["cohens_d"])
        series["cos_mind_placebo"].append(cosine(d_mind, d_plac))
        series["cos_mind_negation"].append(cosine(d_mind, d_neg))
        series["cos_placebo_negation"].append(cosine(d_plac, d_neg))
        series["cos_baseline"].append(cosine(d_a, d_b))

    i = layers.index(layer)
    at = {k: v[i] for k, v in series.items()}
    baseline, observed = at["cos_baseline"], at["cos_mind_placebo"]
    excess = observed - baseline

    # ---- figures ----------------------------------------------------------
    figdir = args.figdir
    figdir.mkdir(parents=True, exist_ok=True)

    def fp(name):
        return versioned(figdir / name)

    f1 = fig_accuracy_by_layer(layers, series["mind_acc"], series["placebo_acc"],
                               series["negation_acc"], band, layer,
                               fp("fig1_accuracy_by_layer.png"))
    f2 = fig_margin_and_effect_size(layers, series["margin"], series["d_mind"],
                                    series["d_placebo"], layer,
                                    fp("fig2_margin_and_effect_size.png"))
    f3 = fig_direction_overlap(layers, series["cos_mind_placebo"],
                               series["cos_mind_negation"], series["cos_placebo_negation"],
                               series["cos_baseline"], layer,
                               fp("fig3_direction_overlap.png"))
    a_mind_ho = mind_ho["acts_by_layer"][layer]
    d_mind = _extract(mind["acts_by_layer"][layer], y_mind)
    d_plac = _extract(plac["acts_by_layer"][layer], y_plac)
    f4 = fig_projection_distributions(
        project(a_mind_ho, d_mind), project(a_mind_ho, d_plac), y_mind_ho,
        fit_threshold(mind["acts_by_layer"][layer], y_mind, d_mind),
        fit_threshold(plac["acts_by_layer"][layer], y_plac, d_plac),
        AFFIRM, layer, fp("fig4_projection_distributions.png"))

    # ---- verdict ----------------------------------------------------------
    if excess <= 0.05:
        verdict = "AMBIENT ANISOTROPY (b)"
        meaning = ("the mind/placebo overlap is no larger than two unrelated contrasts "
                   "show at this layer, so it is geometry, not shared content. The "
                   "specificity claim survives.")
    elif excess >= 0.15:
        verdict = "SHARED CONTENT (a)"
        meaning = (f"the overlap exceeds ambient alignment by {excess:+.3f}. That excess "
                   f"is real shared content, and the mind-specific effect is the margin, "
                   f"not the raw accuracy.")
    else:
        verdict = "MIXED — mostly ambient, some excess"
        meaning = (f"the overlap sits {excess:+.3f} above ambient. Most of the alignment "
                   f"is geometry, but not all of it; report both numbers.")

    report = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_layer": layer,
        "chance_band": band,
        "placebo_half_a": list(PLACEBO_HALF_A),
        "placebo_half_b": list(PLACEBO_HALF_B),
        "n_half_a_pairs": int(mask_a.sum() // 2),
        "n_half_b_pairs": int(mask_b.sum() // 2),
        "layers": layers,
        "series": series,
        "at_selected_layer": at,
        "anisotropy_baseline": baseline,
        "observed_cos_mind_placebo": observed,
        "excess_over_baseline": excess,
        "verdict": verdict,
        "verdict_meaning": meaning,
        "figures": [str(f1), str(f2), str(f3), str(f4)],
    }
    out_path = versioned(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=float))

    # ---- 15-line human summary -------------------------------------------
    print("=" * 74)
    print(f"  SPECIFICITY BATTERY — {args.model} @ {args.position}, layer L{layer}")
    print(f"  placebo half A {PLACEBO_HALF_A} ({int(mask_a.sum()//2)} pairs)")
    print(f"  placebo half B {PLACEBO_HALF_B} ({int(mask_b.sum()//2)} pairs)")
    print("-" * 74)
    print(f"  cos(mind, placebo)          {observed:+.3f}   <- what E1 reported")
    print(f"  BASELINE cos(plcA, plcB)    {baseline:+.3f}   <- ambient, no shared content")
    print(f"  excess over baseline        {excess:+.3f}")
    print("-" * 74)
    print(f"  cos(mind, negation)         {at['cos_mind_negation']:+.3f}")
    print(f"  cos(placebo, negation)      {at['cos_placebo_negation']:+.3f}")
    print(f"  negation acc on mind items  {at['negation_acc']:.3f}   (band {band:.3f})")
    print(f"  negation acc on placebo     {at['negation_acc_on_placebo']:.3f}")
    print(f"  margin (mind - placebo)     {at['margin']:+.3f}"
          f"   d_mind {at['d_mind']:.2f} / d_placebo {at['d_placebo']:.2f}")
    print("-" * 74)
    print(f"  VERDICT: {verdict}")
    print(f"    {meaning}")
    print(f"  report {out_path}   figures {figdir}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
