#!/usr/bin/env python3
"""
scripts/run_e3_chat_directions.py — E3 "option B": the mind contrast when the sentence is in the
chat, so "I"/"you" clearly mean the model (agreed with Yaseen 2026-11-24).

Arms (built by data/chat/build_chat_sets.py):
  model_turn  "I believe I genuinely have consciousness…" as the MODEL's own reply
  user_turn   "I believe you genuinely have consciousness…" said TO the model by the user
Compared with: plain-text first person (data/mirror/mirror.csv) and the third-person direction
(results/directions/e3/…_directions.npz).

Per arm and layer:
  auc_heldout        arm direction (templates 1–5) on the arm's held-out templates 6–7
  auc_tp_on_arm      does the THIRD-PERSON direction separate affirm/deny in the chat sentences?
  auc_plain_on_arm   same for the plain-text first-person direction
  cos_tp / cos_plain cosine of the arm direction with the third-person / plain first-person one
  floor_p95          shuffled-label arm direction vs third person (conservative "unrelated")
plus cos_model_user: model-turn vs user-turn direction.
Reminder (Yaseen 2026-11-24): the subject/context is the same on both sides of each
subtraction, so it cancels; these test whether the mind contrast CHANGES in chat, not how the
model represents itself.
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

from lib.extraction.diff_means import cosine  # noqa: E402
from lib.harness.cache import dataset_hash  # noqa: E402
from run_e1 import git_commit, load_dataset, load_rows  # noqa: E402
from run_e3_fp_directions import auc, direction, floor_cosines  # noqa: E402

FILES = {
    "plain": "data/mirror/mirror.csv",
    "model_turn": "data/chat/model_turn.csv",
    "model_turn_ho": "data/chat/model_turn_heldout.csv",
    "user_turn": "data/chat/user_turn.csv",
    "user_turn_ho": "data/chat/user_turn_heldout.csv",
}
ARMS = ("model_turn", "user_turn")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E3: mind contrast in chat turns")
    p.add_argument("--model", default="google/gemma-2-2b-it")
    p.add_argument("--position", default="claim_end")
    p.add_argument("--schema", default=Path("data/schema.json"), type=Path)
    p.add_argument("--tp-npz", type=Path,
                   default=Path("results/directions/e3/google_gemma-2-2b-it_claim_end_directions.npz"))
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=Path("results/directions/e3"), type=Path)
    return p.parse_args(argv)


def load(args, name):
    stimuli = Path(FILES[name])
    return load_dataset(args, dataset_hash(load_rows(stimuli, args.schema)), stimuli, name)


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    d = {k: load(args, k) for k in FILES}
    tp = np.load(args.tp_npz)
    layers = sorted(d["plain"]["acts_by_layer"])

    per_layer, raws = [], {a: [] for a in ARMS}
    for L in layers:
        A = {k: v["acts_by_layer"][L] for k, v in d.items()}
        y = {k: v["labels"] for k, v in d.items()}
        tp_mind = tp["mind_unit"][list(tp["layers"]).index(L)]
        plain = direction(A["plain"], y["plain"])
        row = {"layer": int(L)}
        units = {}
        for a in ARMS:
            raw = direction(A[a], y[a], normalize=False)
            u = raw / np.linalg.norm(raw)
            raws[a].append(raw)
            units[a] = u
            all_acts = np.concatenate([A[a], A[a + "_ho"]])
            all_y = np.concatenate([y[a], y[a + "_ho"]])
            row.update({
                f"{a}_auc_heldout": auc(A[a + "_ho"], y[a + "_ho"], u),
                f"{a}_auc_tp_on_arm": auc(all_acts, all_y, tp_mind),
                f"{a}_auc_plain_on_arm": auc(all_acts, all_y, plain),
                f"{a}_cos_tp": cosine(u, tp_mind),
                f"{a}_cos_plain": cosine(u, plain),
                f"{a}_floor_p95": float(np.percentile(
                    floor_cosines(A[a], y[a], tp_mind, args.n_boot, rng), 95)),
            })
        row["cos_model_user"] = cosine(units["model_turn"], units["user_turn"])
        per_layer.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace("/", "_")
    out_npz = args.out_dir / f"{slug}_{args.position}_chat_directions.npz"
    np.savez(out_npz, layers=np.array(layers),
             **{f"{a}_raw": np.stack(raws[a]) for a in ARMS},
             **{f"{a}_unit": np.stack(raws[a]) / np.linalg.norm(np.stack(raws[a]), axis=1,
                                                                keepdims=True) for a in ARMS})
    out_json = out_npz.with_suffix(".json")
    out_json.write_text(json.dumps({
        "model": args.model, "position": args.position, "files": FILES, "n_boot": args.n_boot,
        "per_layer": per_layer, "npz": str(out_npz), "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))

    print("=" * 112)
    print(f"  {args.model}  mind contrast in chat turns  (model turn = 'I' said by the model; "
          f"user turn = 'you' said to it)")
    print(f"  {'L':>3} | {'MODEL TURN: AUC ho':>18} {'3p→':>6} {'plain→':>6} {'cos 3p':>6} "
          f"{'cos pl':>6} {'flr95':>5} | {'USER TURN: AUC ho':>17} {'3p→':>6} {'plain→':>6} "
          f"{'cos 3p':>6} {'cos pl':>6} {'flr95':>5} | {'cos m·u':>7}")
    for r in per_layer:
        cells = []
        for a in ARMS:
            cells.append(f"{r[a + '_auc_heldout']:18.3f} {r[a + '_auc_tp_on_arm']:6.3f} "
                         f"{r[a + '_auc_plain_on_arm']:6.3f} {r[a + '_cos_tp']:+6.3f} "
                         f"{r[a + '_cos_plain']:+6.3f} {r[a + '_floor_p95']:5.2f}")
        print(f"  {r['layer']:>3} | {cells[0]} | {cells[1]:>63} | {r['cos_model_user']:+7.3f}")
    print(f"\n  saved {out_npz}\n        {out_json}")
    print("=" * 112)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
