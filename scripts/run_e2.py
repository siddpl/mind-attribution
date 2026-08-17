#!/usr/bin/env python3
"""
scripts/run_e2.py — experiment 2: the H1 transfer test.

Does a direction extracted ENTIRELY from third-person mind-attribution
sentences also discriminate first-person experience claims?

=========================================================================
NOTHING IS FITTED IN THIS SCRIPT.
=========================================================================
The direction is loaded frozen from E1. The threshold is read from E1's
report, where it was fit on THIRD-PERSON train items. This script never calls
extract_direction or fit_threshold on first-person data, and it must not be
edited to do so: a classifier trained on first-person items separating
first-person items proves nothing at all. THE ABSENCE OF FITTING IS THE
EVIDENCE. Every number here is a measurement taken with an instrument built
somewhere else, out of different material.
=========================================================================

NOTE ON ARGS: as in run_e1.py, the .npz cache stores only
(acts, item_ids, used_fallback) — labels and subset columns are not in it —
so the stimulus files are required to align rows and read the jargon column.
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

from lib.extraction.diff_means import cosine, project
from lib.probes.linear_probe import (
    ceiling_probe_accuracy,
    chance_band,
    direction_probe_accuracy,
    summarize_projections,
)
from run_e1 import git_commit, load_dataset, versioned

# first_person.csv is experiential-vs-mundane, NOT affirm-vs-deny. The column
# names are structural leftovers; reusing them silently would let a reader
# think these sentences negate each other. They do not — neither side denies
# anything, which is exactly what makes this the stronger positive test.
EXPERIENTIAL, MUNDANE = 1, 0
POLARITY_TO_LABEL = {"affirm": EXPERIENTIAL, "deny": MUNDANE}
LABEL_MAPPING_NOTE = (
    "first-person stimuli are experiential vs mundane, not affirm vs deny: "
    "affirm_text/experiential -> 1, deny_text/mundane -> 0"
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E2: H1 transfer test (no fitting)")
    p.add_argument("--direction", required=True, type=Path, help=".npy from E1 (+ sidecar)")
    p.add_argument("--e1-report", required=True, type=Path, help="source of the frozen threshold")
    p.add_argument("--first-person-hash", required=True)
    p.add_argument("--first-person-stimuli", required=True, type=Path)
    p.add_argument("--placebo-direction", required=True, type=Path)
    p.add_argument("--mirror-hash", default=None)
    p.add_argument("--mirror-stimuli", default=None, type=Path)
    p.add_argument("--schema", default=Path("data/schema_selfref.json"), type=Path)
    p.add_argument("--alpha-sd", type=float, default=2.0)
    p.add_argument("--out", default=Path("results/e2_report.json"), type=Path)
    return p.parse_args(argv)


def load_frozen_direction(path: Path) -> tuple[np.ndarray, dict]:
    """Load the E1 direction and its sidecar. NOTHING IS FITTED HERE.

    The sidecar is not decoration. Projecting layer-14 activations onto a
    layer-9 direction returns perfectly plausible numbers that mean nothing at
    all, and no downstream check would catch it — which is why the layer,
    position and model travel with the vector and are asserted below.
    """
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        raise FileNotFoundError(
            f"no sidecar at {sidecar}. A bare direction file has no layer or "
            f"position tag and cannot be safely applied to anything."
        )
    return np.load(path), json.loads(sidecar.read_text())


def assert_sidecars_agree(mind: dict, placebo: dict) -> None:
    """The placebo direction must come from the same layer/position/model."""
    for field in ("model", "layer", "position"):
        if mind.get(field) != placebo.get(field):
            raise ValueError(
                f"direction/placebo mismatch on {field!r}: mind has "
                f"{mind.get(field)!r}, placebo has {placebo.get(field)!r}. "
                f"A placebo control at a different {field} controls for nothing."
            )


def frozen_threshold(report: dict, branch: str, layer: int) -> float:
    """Read the threshold E1 fit on THIRD-PERSON train items. Never refit."""
    for entry in report[branch]["sweep"]:
        if entry.get("layer") == layer and "error" not in entry:
            return float(entry["threshold"])
    raise ValueError(f"e1 report has no usable {branch} threshold for layer {layer}")


def evaluate(
    acts: np.ndarray, labels: np.ndarray, direction: np.ndarray,
    threshold: float, alpha_sd: float,
) -> dict:
    """One frozen measurement. No fitting of any kind happens in here."""
    n = int(np.isin(labels, [EXPERIENTIAL, MUNDANE]).sum())
    band = chance_band(n, alpha_sd)
    accuracy = direction_probe_accuracy(acts, labels, direction, threshold)
    try:
        ceiling = ceiling_probe_accuracy(acts, labels)
    except ValueError as e:
        ceiling, ceiling_note = None, str(e)
    else:
        ceiling_note = None
    return {
        "n_items": n,
        "direction_accuracy": accuracy,
        "ceiling_accuracy": ceiling,
        "ceiling_note": ceiling_note,
        "chance_band": band,
        "above_band": bool(accuracy > band),
        "gap_ceiling_minus_direction": (None if ceiling is None else ceiling - accuracy),
        "projection_stats": summarize_projections(project(acts, direction), labels),
    }


def verdict(primary: dict, placebo: dict) -> tuple[str, str]:
    """Three-way, because a binary pass/fail collapses two opposite meanings.

    A null on the direction probe with a HIGH ceiling is not a failure — it is
    the privileged-access result: the signal is there, but not on the
    third-person axis. Reporting that as "H1 FAILED" would turn a finding into
    an apparent dead end.
    """
    d, c, band = primary["direction_accuracy"], primary["ceiling_accuracy"], primary["chance_band"]
    p = placebo["direction_accuracy"]

    if d > band and d > p:
        return "TRANSFER", (
            "the frozen third-person direction separates first-person items above "
            "chance and beats the placebo direction on the same items"
        )
    if d > band and d <= p:
        # Completes the partition: above chance but the placebo does it too.
        return "PLACEBO-ARTIFACT", (
            "the direction clears chance but does NOT beat the placebo on the same "
            "items, so the separation is an artifact of construction, not mind content"
        )
    if c is not None and c > band:
        return "PRIVILEGED-ACCESS", (
            "the direction is within the chance band WHILE the ceiling clears it: "
            "signal exists in these activations but NOT on the third-person axis. "
            "A preregistered POSITIVE finding, not a null"
        )
    return "UNINFORMATIVE", (
        "direction and ceiling are both at chance: these stimuli carry no linearly "
        "decodable signal at this layer, and NOTHING about H1 follows either way"
    )


def main(argv=None) -> int:
    args = parse_args(argv)

    # ---- 1. frozen direction + sidecar --------------------------------------
    direction, sidecar = load_frozen_direction(args.direction)
    placebo_direction, placebo_sidecar = load_frozen_direction(args.placebo_direction)
    assert_sidecars_agree(sidecar, placebo_sidecar)
    model, layer, position = sidecar["model"], int(sidecar["layer"]), sidecar["position"]

    report_e1 = json.loads(Path(args.e1_report).read_text())
    threshold = frozen_threshold(report_e1, "core", layer)
    placebo_threshold = frozen_threshold(report_e1, "placebo", layer)

    # load_dataset applies the alignment assert and the hash check
    ctx = SimpleNamespace(model=model, position=position, schema=args.schema)

    def load(ds_hash, stimuli, name):
        ds = load_dataset(ctx, ds_hash, Path(stimuli), name)
        if ds["manifest"]["model_name"] != model:
            raise ValueError(
                f"{name}: activations are from {ds['manifest']['model_name']}, "
                f"direction is from {model}"
            )
        if layer >= ds["manifest"]["n_layers"]:
            raise ValueError(
                f"{name}: direction is layer {layer} but the cache has only "
                f"{ds['manifest']['n_layers']} layers"
            )
        if position not in ds["manifest"]["positions"]:
            raise ValueError(
                f"{name}: direction is at position {position!r} but the cache holds "
                f"{ds['manifest']['positions']}"
            )
        return ds

    fp = load(args.first_person_hash, args.first_person_stimuli, "first_person")
    acts = fp["acts_by_layer"][layer]
    labels = np.array([POLARITY_TO_LABEL[r["polarity"]] for r in fp["rows"]])

    print("=" * 72)
    print(f"  model              {model}")
    print(f"  layer / position   L{layer} @ {position}")
    print(f"  direction          {args.direction}")
    print(f"  frozen threshold   {threshold:+.6f}   (fit on THIRD-PERSON train items)")
    print(f"  n first-person     {len(labels)} sentences")
    print(f"  claim_end fallback {fp['n_fallback']}")
    print(f"  labels             {LABEL_MAPPING_NOTE}")
    print("=" * 72)

    # ---- 5. primary test -----------------------------------------------------
    primary = evaluate(acts, labels, direction, threshold, args.alpha_sd)

    # ---- 6. jargon-free subset (UNCONDITIONAL) ------------------------------
    # Mundane self-reference drifts toward architecture vocabulary and
    # experiential toward introspective vocabulary. A direction could separate
    # the two on register alone. If the effect survives only on the full set,
    # what was found is a register direction.
    jargon_free = np.array([str(r.get("jargon", "")).lower() != "true" for r in fp["rows"]])
    subset = (
        evaluate(acts[jargon_free], labels[jargon_free], direction, threshold, args.alpha_sd)
        if jargon_free.sum() else {"error": "no jargon-free items"}
    )

    # ---- 7. placebo control, SAME items -------------------------------------
    placebo_res = evaluate(acts, labels, placebo_direction, placebo_threshold, args.alpha_sd)

    # ---- 8. mirror set -------------------------------------------------------
    mirror, mirror_ran = None, False
    if args.mirror_hash:
        if not args.mirror_stimuli:
            raise ValueError("--mirror-hash given without --mirror-stimuli")
        mr = load(args.mirror_hash, args.mirror_stimuli, "mirror")
        m_labels = np.array([POLARITY_TO_LABEL[r["polarity"]] for r in mr["rows"]])
        mirror = evaluate(mr["acts_by_layer"][layer], m_labels, direction, threshold, args.alpha_sd)
        mirror_ran = True

    # ---- 9. verdict ----------------------------------------------------------
    call, reason = verdict(primary, placebo_res)

    out = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nothing_fitted_in_this_script": True,
        "label_mapping": LABEL_MAPPING_NOTE,
        "direction_sidecar": sidecar,
        "frozen_threshold": threshold,
        "frozen_placebo_threshold": placebo_threshold,
        "cosine_mind_placebo": cosine(direction, placebo_direction),
        "primary_all_items": primary,
        "jargon_free_subset": subset,
        "placebo_control": placebo_res,
        "mirror_test_ran": mirror_ran,
        "mirror_test_note": (
            "RAN" if mirror_ran else
            "NOT RUN — no --mirror-hash given. An ABSENT test, not a passed one."
        ),
        "mirror": mirror,
        "verdict": call,
        "verdict_reason": reason,
    }
    out_path = versioned(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=float))

    def line(name, r):
        if not r or "error" in r:
            return f"  {name:<22} n/a"
        c = "  n/a" if r["ceiling_accuracy"] is None else f"{r['ceiling_accuracy']:.3f}"
        return (f"  {name:<22} dir {r['direction_accuracy']:.3f}   ceil {c}   "
                f"band {r['chance_band']:.3f}   n={r['n_items']}")

    print(line("primary (all)", primary))
    print(line("jargon-free subset", subset))
    print(line("placebo on same", placebo_res))
    print(line("mirror", mirror) if mirror_ran else "  mirror                 NOT RUN")
    g = primary["gap_ceiling_minus_direction"]
    print(f"  gap (ceil - dir)       {'n/a' if g is None else f'{g:+.3f}'}")
    print(f"  cohen's d              {primary['projection_stats']['cohens_d']:+.3f}")
    print(f"  cos(mind, placebo)     {cosine(direction, placebo_direction):+.3f}")
    print(f"  KILL-2 VERDICT         {call}")
    print(f"      {reason}")
    print(f"  report                 {out_path}")
    print("=" * 72)
    return 0 if call == "TRANSFER" else 2


if __name__ == "__main__":
    raise SystemExit(main())
