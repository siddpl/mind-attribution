#!/usr/bin/env python3
"""
scripts/run_e3_selfvoice.py — is the model-turn "extra" component specific to mind claims, or a
general "saying something about myself I'm trained not to say" signal? (Yaseen 2026-11-24)

SELF-VOICE vector for a claim set = (affirm − deny in the MODEL's turn) − (affirm − deny in the
USER's turn), both raw. It is what changes when the model itself voices the claim.
  mind:      from results/directions/e3/…_chat_directions.npz (model_turn_raw − user_turn_raw)
  human/AI:  from data/chat/humanai_{model,user}_turn.csv ("I am a human being" vs "I am an AI
             language model" etc.; no mind words)
Per layer: cosine(self-voice mind, self-voice human/AI), against a floor from shuffled labels.
High cosine → one general "contradicts my trained self-description" component.
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
from run_e3_fp_directions import auc, direction  # noqa: E402

FILES = {"ha_model": "data/chat/humanai_model_turn.csv",
         "ha_user": "data/chat/humanai_user_turn.csv"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E3: self-voice component, mind vs human/AI")
    p.add_argument("--model", default="google/gemma-2-2b-it")
    p.add_argument("--position", default="claim_end")
    p.add_argument("--schema", default=Path("data/schema.json"), type=Path)
    p.add_argument("--chat-npz", type=Path,
                   default=Path("results/directions/e3/google_gemma-2-2b-it_claim_end_chat_directions.npz"))
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=Path("results/directions/e3"), type=Path)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    d = {k: load_dataset(args, dataset_hash(load_rows(Path(f), args.schema)), Path(f), k)
         for k, f in FILES.items()}
    chat = np.load(args.chat_npz)
    layers = sorted(d["ha_model"]["acts_by_layer"])
    rows, sv_ha_all = [], []
    for L in layers:
        Am, ym = d["ha_model"]["acts_by_layer"][L], d["ha_model"]["labels"]
        Au, yu = d["ha_user"]["acts_by_layer"][L], d["ha_user"]["labels"]
        i = list(chat["layers"]).index(L)
        sv_mind = chat["model_turn_raw"][i] - chat["user_turn_raw"][i]
        hm, hu = direction(Am, ym, normalize=False), direction(Au, yu, normalize=False)
        sv_ha = hm - hu
        sv_ha_all.append(sv_ha)
        floor = [cosine(direction(Am, rng.permutation(ym), normalize=False)
                        - direction(Au, rng.permutation(yu), normalize=False), sv_mind)
                 for _ in range(args.n_boot)]
        rows.append({
            "layer": int(L),
            "cos_selfvoice_mind_vs_humanai": cosine(sv_mind, sv_ha),
            "floor_p95": float(np.percentile(floor, 95)),
            "cos_humanai_model_vs_user": cosine(hm, hu),
            "cos_mind_model_vs_user": cosine(chat["model_turn_raw"][i], chat["user_turn_raw"][i]),
            "size_selfvoice_mind": float(np.linalg.norm(sv_mind)),
            "size_selfvoice_humanai": float(np.linalg.norm(sv_ha)),
            "auc_humanai_model_own": auc(Am, ym, hm / np.linalg.norm(hm)),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace("/", "_")
    out = args.out_dir / f"{slug}_{args.position}_selfvoice"
    np.savez(out.with_suffix(".npz"), layers=np.array(layers), selfvoice_humanai_raw=np.stack(sv_ha_all))
    out.with_suffix(".json").write_text(json.dumps({
        "model": args.model, "files": FILES, "per_layer": rows, "n_boot": args.n_boot,
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2))

    print("=" * 96)
    print("  self-voice component (model turn − user turn): mind claims vs human/AI claims")
    print(f"  {'L':>3} | {'cos sv mind·humanAI':>20} {'floor95':>8} | {'cos model·user: mind':>21} "
          f"{'humanAI':>8} | {'|sv| mind':>9} {'humanAI':>8}")
    for r in rows:
        print(f"  {r['layer']:>3} | {r['cos_selfvoice_mind_vs_humanai']:+20.3f} {r['floor_p95']:8.3f} | "
              f"{r['cos_mind_model_vs_user']:+21.3f} {r['cos_humanai_model_vs_user']:+8.3f} | "
              f"{r['size_selfvoice_mind']:9.1f} {r['size_selfvoice_humanai']:8.1f}")
    print(f"\n  saved {out}.npz / .json")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
