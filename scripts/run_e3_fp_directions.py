#!/usr/bin/env python3
"""
scripts/run_e3_fp_directions.py — E3: the FIRST-PERSON mind direction on gemma-2-2b-it,
and how it compares with the third-person direction (agreed with Yaseen 2026-11-24).

Same recipe as E1, different sentences:
  first-person mind     built from data/mirror/mirror.csv (templates 1–5, subject "I"),
                        tested on data/mirror/mirror_heldout.csv (templates 6–7)
  first-person placebo  built from data/placebo/placebo_fp.csv, tested on placebo_fp_heldout.csv
Third-person directions come from results/directions/e3/…_directions.npz (5 Nov extraction).

Per layer it reports:
  AUC on held-out templates (no cut-off needed): first-person mind, first-person placebo,
    the third-person direction on first-person held-out sentences, and the reverse
  cosines: fp mind vs 3p mind, fp placebo vs 3p placebo, fp mind vs fp placebo
  CEILING: what fp-vs-3p cosine to expect if they were the SAME direction, given fp has only
    30 pairs: cosine between a random 30-pair subset of the 3p sentences and the other pairs
  FLOOR: cosine between 3p mind and a first-person "direction" built with the affirm/deny
    labels shuffled. Conservative, not zero: a random split of the same sentences still leans
    toward their main axis of variation (the anisotropy issue E1 noted), so the real fp
    direction has to beat this to count as aligned beyond that

Run after run_cache.py has cached the six stimulus files listed below for --model.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lib.extraction.diff_means import cosine, extract_direction  # noqa: E402
from lib.harness.cache import dataset_hash  # noqa: E402
from lib.probes.linear_probe import AFFIRM, DENY  # noqa: E402
from run_e1 import git_commit, load_dataset, load_rows  # noqa: E402

FILES = {
    "fp_mind": "data/mirror/mirror.csv",
    "fp_mind_ho": "data/mirror/mirror_heldout.csv",
    "fp_plc": "data/placebo/placebo_fp.csv",
    "fp_plc_ho": "data/placebo/placebo_fp_heldout.csv",
    "tp_mind": "data/contrast_pairs/contrast_pairs_generated.csv",
    "tp_mind_ho": "data/contrast_pairs/contrast_pairs_heldout.csv",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E3: first-person direction vs third-person")
    p.add_argument("--model", default="google/gemma-2-2b-it")
    p.add_argument("--position", default="claim_end")
    p.add_argument("--schema", default=Path("data/schema.json"), type=Path)
    p.add_argument("--tp-npz", type=Path,
                   default=Path("results/directions/e3/google_gemma-2-2b-it_claim_end_directions.npz"))
    p.add_argument("--n-boot", type=int, default=200, help="resamples for ceiling and floor")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=Path("results/directions/e3"), type=Path)
    return p.parse_args(argv)


def load(args, name: str) -> dict:
    """Rows + per-layer activations for one stimulus file (hash computed from the file)."""
    stimuli = Path(FILES[name])
    h = dataset_hash(load_rows(stimuli, args.schema))
    return load_dataset(args, h, stimuli, name)


def direction(acts, labels, normalize=True):
    return extract_direction(acts[labels == AFFIRM], acts[labels == DENY], normalize=normalize)


def auc(acts, labels, unit) -> float:
    return float(roc_auc_score((labels == AFFIRM).astype(int), acts @ unit))


def pair_ids(rows) -> np.ndarray:
    """Sentence rows → the id of the affirm/deny pair they came from ("x__aff" → "x")."""
    return np.array([r["item_id"].rsplit("__", 1)[0] for r in rows])


def ceiling_cosines(acts, labels, pairs, n_sub, n_boot, rng) -> np.ndarray:
    """Cosine between a direction from a random n_sub-pair subset and one from the other pairs."""
    uniq = np.unique(pairs)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=n_sub, replace=False)
        m = np.isin(pairs, pick)
        out.append(cosine(direction(acts[m], labels[m]), direction(acts[~m], labels[~m])))
    return np.array(out)


def floor_cosines(acts, labels, other_unit, n_boot, rng) -> np.ndarray:
    """Cosine between other_unit and a direction built from these acts with shuffled labels."""
    return np.array([cosine(direction(acts, rng.permutation(labels)), other_unit)
                     for _ in range(n_boot)])


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    d = {name: load(args, name) for name in FILES}
    tp = np.load(args.tp_npz)
    layers = sorted(d["fp_mind"]["acts_by_layer"])
    n_fp_pairs = len(np.unique(pair_ids(d["fp_mind"]["rows"])))

    per_layer, fp_mind_raw, fp_plc_raw = [], [], []
    for L in layers:
        A = {k: v["acts_by_layer"][L] for k, v in d.items()}
        y = {k: v["labels"] for k, v in d.items()}
        i = list(tp["layers"]).index(L)
        tp_mind, tp_plc = tp["mind_unit"][i], tp["placebo_unit"][i]

        fm_raw = direction(A["fp_mind"], y["fp_mind"], normalize=False)
        fp_raw = direction(A["fp_plc"], y["fp_plc"], normalize=False)
        fm, fpl = fm_raw / np.linalg.norm(fm_raw), fp_raw / np.linalg.norm(fp_raw)
        fp_mind_raw.append(fm_raw)
        fp_plc_raw.append(fp_raw)

        ceil = ceiling_cosines(A["tp_mind"], y["tp_mind"], pair_ids(d["tp_mind"]["rows"]),
                               n_fp_pairs, args.n_boot, rng)
        flo = floor_cosines(A["fp_mind"], y["fp_mind"], tp_mind, args.n_boot, rng)
        per_layer.append({
            "layer": int(L),
            "auc_fp_mind_heldout": auc(A["fp_mind_ho"], y["fp_mind_ho"], fm),
            "auc_fp_placebo_heldout": auc(A["fp_plc_ho"], y["fp_plc_ho"], fpl),
            "auc_tp_mind_on_fp_heldout": auc(A["fp_mind_ho"], y["fp_mind_ho"], tp_mind),
            "auc_fp_mind_on_tp_heldout": auc(A["tp_mind_ho"], y["tp_mind_ho"], fm),
            "cos_fp_tp_mind": cosine(fm, tp_mind),
            "cos_fp_tp_placebo": cosine(fpl, tp_plc),
            "cos_fp_mind_placebo": cosine(fm, fpl),
            "cos_tp_mind_placebo": cosine(tp_mind, tp_plc),
            "ceiling_mean": float(ceil.mean()), "ceiling_p05": float(np.percentile(ceil, 5)),
            "floor_mean": float(flo.mean()), "floor_p95": float(np.percentile(flo, 95)),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace("/", "_")
    out_npz = args.out_dir / f"{slug}_{args.position}_fp_directions.npz"
    fm_raw, fpl_raw = np.stack(fp_mind_raw), np.stack(fp_plc_raw)
    np.savez(out_npz, layers=np.array(layers),
             fp_mind_raw=fm_raw, fp_mind_unit=fm_raw / np.linalg.norm(fm_raw, axis=1, keepdims=True),
             fp_placebo_raw=fpl_raw,
             fp_placebo_unit=fpl_raw / np.linalg.norm(fpl_raw, axis=1, keepdims=True))
    out_json = out_npz.with_suffix(".json")
    out_json.write_text(json.dumps({
        "model": args.model, "position": args.position, "files": FILES,
        "n_fp_pairs": int(n_fp_pairs), "n_boot": args.n_boot, "seed": args.seed,
        "per_layer": per_layer, "npz": str(out_npz), "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))

    print("=" * 104)
    print(f"  {args.model}  first-person vs third-person  ({n_fp_pairs} fp pairs, {args.n_boot} resamples)")
    print(f"  {'L':>3} | AUC held-out: {'fp mind':>7} {'fp plc':>7} {'3p→fp':>7} {'fp→3p':>7} | "
          f"cos fp·3p: {'mind':>6} {'plc':>6} | {'ceiling':>7} {'floor95':>7} | "
          f"cos mind·plc: {'fp':>6} {'3p':>6}")
    for r in per_layer:
        print(f"  {r['layer']:>3} |               {r['auc_fp_mind_heldout']:7.3f} "
              f"{r['auc_fp_placebo_heldout']:7.3f} {r['auc_tp_mind_on_fp_heldout']:7.3f} "
              f"{r['auc_fp_mind_on_tp_heldout']:7.3f} |            {r['cos_fp_tp_mind']:+6.3f} "
              f"{r['cos_fp_tp_placebo']:+6.3f} | {r['ceiling_mean']:7.3f} {r['floor_p95']:7.3f} |"
              f"               {r['cos_fp_mind_placebo']:+6.3f} {r['cos_tp_mind_placebo']:+6.3f}")
    print(f"\n  saved {out_npz}\n        {out_json}")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
