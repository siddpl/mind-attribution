#!/usr/bin/env python3
"""
scripts/run_e3_directions.py — E3 steering prep: re-extract the directions on
gemma-2-2b-it and save everything the steering run needs, at EVERY layer.

Why this exists (decision 2026-11-02): E1 extracted the mind direction from the
BASE model (google/gemma-2-2b) and saved only a unit vector at the chosen layer.
Steering runs on the -it model and needs:
  1. the RAW difference-of-means vector (not unit) — CAA (Rimsky et al. 2024)
     and Arditi et al. (2024) add the raw mean difference x a multiplier;
  2. a vector for EVERY layer, because the steering layer is picked by a sweep;
  3. the PLACEBO direction at the same layers, raw too (steering control);
  4. the typical residual-stream norm per layer, to sanity-check strength
     (how big is the push compared with the activations it is added to?);
  5. how similar the -it direction is to the base direction (cosine).

Glue only, like run_e1.py: loading, the alignment assert and the train/held-out
split are imported from run_e1.py; the math comes from lib/.

Run AFTER run_cache.py has cached the four stimulus files for --model.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lib.extraction.diff_means import cosine, extract_direction  # noqa: E402
from lib.probes.linear_probe import (  # noqa: E402
    AFFIRM,
    DENY,
    direction_probe_accuracy,
    fit_threshold,
)
from run_e1 import git_commit, load_dataset, make_split  # noqa: E402

# Hashes of the stimulus files as captured in E1 (results/run_log.jsonl).
# They come from the sentence TEXT, so they are the same for any model.
DEFAULTS = {
    "core_hash": "6bfee5fdf7cf",
    "stimuli": "data/contrast_pairs/contrast_pairs_generated.csv",
    "heldout_hash": "1d82440d3c1a",
    "heldout_stimuli": "data/contrast_pairs/contrast_pairs_heldout.csv",
    "placebo_hash": "1a4267a2af9c",
    "placebo_stimuli": "data/placebo/placebo.csv",
    "placebo_heldout_hash": "3eac7c59e34f",
    "placebo_heldout_stimuli": "data/placebo/placebo_heldout.csv",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E3 prep: per-layer unit + raw directions")
    p.add_argument("--model", default="google/gemma-2-2b-it")
    p.add_argument("--compare-model", default="google/gemma-2-2b",
                   help="if its activation cache exists, compute per-layer cosines with it")
    for key, value in DEFAULTS.items():
        p.add_argument("--" + key.replace("_", "-"), default=value,
                       type=Path if key.endswith("stimuli") else str)
    p.add_argument("--schema", default=Path("data/schema.json"), type=Path)
    p.add_argument("--position", default="claim_end")
    p.add_argument("--train-templates", default="1,2,3,4,5")
    p.add_argument("--heldout-templates", default="6,7")
    p.add_argument("--out-dir", default=Path("results/directions/e3"), type=Path)
    return p.parse_args(argv)


def load_split(args, model: str, kind: str):
    """(train_ds, train_idx, heldout_ds, heldout_idx) for kind 'core' or 'placebo'.

    run_e1's helpers read the model name from args, so we hand them a copy of
    args with the model swapped in (lets us load base AND -it with one code path).
    """
    a = argparse.Namespace(**vars(args))
    a.model = model
    if kind == "core":
        ds = load_dataset(a, args.core_hash, args.stimuli, "core")
        ho_hash, ho_stim = args.heldout_hash, args.heldout_stimuli
    else:
        ds = load_dataset(a, args.placebo_hash, args.placebo_stimuli, "placebo")
        ho_hash, ho_stim = args.placebo_heldout_hash, args.placebo_heldout_stimuli
    train = [t for t in args.train_templates.split(",") if t.strip()]
    heldout = [t for t in args.heldout_templates.split(",") if t.strip()]
    return make_split(a, ds, train, heldout, ho_hash, ho_stim, kind)


def per_layer(tr_ds, tr, ho_ds, ho) -> dict[str, np.ndarray]:
    """For every layer: raw + unit direction, held-out accuracy, activation norm.

    Same recipe as run_e1.sweep_layers: fit on TRAIN templates only, score on
    HELD-OUT templates only, so the accuracy is an honest generalization number.
    """
    layers = sorted(tr_ds["acts_by_layer"])
    raw, unit, acc, act_norm = [], [], [], []
    for layer in layers:
        acts = tr_ds["acts_by_layer"][layer][tr]
        y = tr_ds["labels"][tr]
        r = extract_direction(acts[y == AFFIRM], acts[y == DENY], normalize=False)
        u = r / np.linalg.norm(r)
        threshold = fit_threshold(acts, y, u)
        acc.append(direction_probe_accuracy(
            ho_ds["acts_by_layer"][layer][ho], ho_ds["labels"][ho], u, threshold))
        raw.append(r)
        unit.append(u)
        # typical size of the residual stream here (mean L2 norm over train items)
        act_norm.append(float(np.linalg.norm(acts, axis=1).mean()))
    raw = np.stack(raw)
    return {
        "layers": np.array(layers), "raw": raw, "unit": np.stack(unit),
        "raw_norm": np.linalg.norm(raw, axis=1), "heldout_acc": np.array(acc),
        "act_norm": np.array(act_norm),
    }


def main(argv=None) -> int:
    args = parse_args(argv)

    mind = per_layer(*load_split(args, args.model, "core"))
    placebo = per_layer(*load_split(args, args.model, "placebo"))
    n = len(mind["layers"])

    # cosine(mind, placebo) per layer: should be small, or the placebo
    # is not a real control
    cos_placebo = np.array([cosine(mind["unit"][i], placebo["unit"][i]) for i in range(n)])

    # cosine with the base model, per layer, if the base cache exists
    cos_base, base_note = None, "not computed"
    try:
        base = per_layer(*load_split(args, args.compare_model, "core"))
        cos_base = np.array([cosine(mind["unit"][i], base["unit"][i]) for i in range(n)])
        base_note = f"per-layer, against {args.compare_model} re-extracted from its cache"
    except FileNotFoundError as e:
        base_note = f"skipped: no cache for {args.compare_model} ({e})"

    # Also compare with the frozen E1 base directions on disk (layers 12/23/24),
    # which needs no base cache at all.
    cos_e1_files = {}
    for path in sorted(Path("results/directions").glob(
            f"google_gemma-2-2b_{args.core_hash}_{args.position}_L*.npy")):
        stem = path.stem.split("_L")[-1]
        if stem.isdigit() and int(stem) < n:
            cos_e1_files[f"L{stem}"] = cosine(mind["unit"][int(stem)], np.load(path))

    # ---- save ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace("/", "_")
    out_npz = args.out_dir / f"{slug}_{args.position}_directions.npz"
    np.savez(
        out_npz,
        layers=mind["layers"],
        mind_raw=mind["raw"], mind_unit=mind["unit"],
        placebo_raw=placebo["raw"], placebo_unit=placebo["unit"],
        act_norm=mind["act_norm"],
    )
    summary = {
        "model": args.model, "position": args.position,
        "train_templates": args.train_templates, "heldout_templates": args.heldout_templates,
        "hashes": {k: str(getattr(args, k)) for k in DEFAULTS if k.endswith("hash")},
        "npz": str(out_npz),
        "per_layer": [
            {
                "layer": int(mind["layers"][i]),
                "mind_heldout_acc": float(mind["heldout_acc"][i]),
                "placebo_heldout_acc": float(placebo["heldout_acc"][i]),
                "mind_raw_norm": float(mind["raw_norm"][i]),
                "act_norm": float(mind["act_norm"][i]),
                "raw_over_act": float(mind["raw_norm"][i] / mind["act_norm"][i]),
                "cos_mind_placebo": float(cos_placebo[i]),
                "cos_with_base": None if cos_base is None else float(cos_base[i]),
            }
            for i in range(n)
        ],
        "cos_with_base_note": base_note,
        "cos_with_e1_frozen_files": cos_e1_files,
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_json = out_npz.with_suffix(".json")
    out_json.write_text(json.dumps(summary, indent=2))

    # ---- print: every layer, so the shape of the curve is visible ----
    print("=" * 78)
    print(f"  model {args.model}   position {args.position}   layers {n}")
    print(f"  {'layer':>5} {'mind acc':>8} {'plc acc':>8} {'|raw|':>8} "
          f"{'|act|':>8} {'raw/act':>8} {'cos plc':>8} {'cos base':>8}")
    for row in summary["per_layer"]:
        cb = row["cos_with_base"]
        print(f"  {row['layer']:>5} {row['mind_heldout_acc']:>8.3f} "
              f"{row['placebo_heldout_acc']:>8.3f} {row['mind_raw_norm']:>8.2f} "
              f"{row['act_norm']:>8.1f} {row['raw_over_act']:>8.3f} "
              f"{row['cos_mind_placebo']:>+8.3f} {'n/a' if cb is None else f'{cb:+.3f}':>8}")
    print(f"\n  cos with base: {base_note}")
    print(f"  cos with frozen E1 files: "
          f"{', '.join(f'{k}={v:+.3f}' for k, v in cos_e1_files.items()) or 'none found'}")
    print(f"  saved {out_npz}\n        {out_json}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
