#!/usr/bin/env python3
"""
scripts/run_e3_dial_prefill.py — two cheap measurements on the 0–10 rating questions, with no
text generation (agreed with Yaseen 2026-11-24).

1. DIAL (unsteered): run each rating question once and record the model's state at the LAST
   token before it would answer (just after "<start_of_turn>model\\n"), at every layer. Project
   it onto four directions: third-person mind, placebo, model-turn ("self-voice") mind, user-turn
   mind. Question: for itself, does the dial sit like it does for minded or unminded things,
   given what it then says?

2. PREFILL: add "<answer>" after the prompt as if the model had started answering, and read the
   probability it puts on each number 0–10 → an EXACT expected rating (no greedy rounding).
   Unsteered, and steered at one layer with mind / placebo / ortho / random / selfvoice. Gemma
   tokenises numbers one digit at a time, so "10" = "1" then "0": P(10) = P("1")·P("0" | "1"),
   P(1) = P("1") − P(10). Steering covers the reply only (from the last prompt token on), as in
   the main run.

Outputs results/e3_steering/dial_L.csv and prefill_L14.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lib.steering.steer import make_steering_hook  # noqa: E402
from run_e3_steering import CHAT_DIRECTIONS, DIRECTIONS, build_vectors, load_items  # noqa: E402

PREFILL = "<answer>"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E3: dial before answering + prefilled ratings")
    p.add_argument("--model", default="google/gemma-2-2b-it")
    p.add_argument("--layer", type=int, default=14, help="steering layer for the prefill part")
    p.add_argument("--mults", default="-2,-1,1,2")
    p.add_argument("--conditions", default="mind,placebo,ortho,random,selfvoice")
    p.add_argument("--extra", default="data/steering/extra_questions.csv")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default=Path("results/e3_steering"), type=Path)
    return p.parse_args(argv)


def expected_rating(p_first: np.ndarray, p_zero_after_one: float) -> tuple[np.ndarray, float, float]:
    """p_first: probabilities of the first digit token "0".."9" (length 10).
    Returns (probabilities over ratings 0..10 renormalised, expected rating, digit mass)."""
    p = np.zeros(11)
    p[:10] = p_first
    p[10] = p_first[1] * p_zero_after_one
    p[1] = p_first[1] - p[10]
    mass = float(p.sum())
    p = p / mass if mass > 0 else p
    return p, float((np.arange(11) * p).sum()), mass


def unit_directions(layers: list[int]) -> dict[str, np.ndarray]:
    t, c = np.load(ROOT / DIRECTIONS), np.load(ROOT / CHAT_DIRECTIONS)
    idx_t = [list(t["layers"]).index(L) for L in layers]
    idx_c = [list(c["layers"]).index(L) for L in layers]
    return {"mind": t["mind_unit"][idx_t], "placebo": t["placebo_unit"][idx_t],
            "selfvoice": c["model_turn_unit"][idx_c], "uservoice": c["user_turn_unit"][idx_c]}


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from transformer_lens import HookedTransformer

    items = [i for i in load_items("all", 0, args.extra) if i["set"] != "C_battery"]
    model = HookedTransformer.from_pretrained_no_processing(
        args.model, dtype=torch.bfloat16, device=args.device)
    tok = model.tokenizer
    digit_ids = [tok.encode(str(d), add_special_tokens=False) for d in range(10)]
    assert all(len(x) == 1 for x in digit_ids), f"digits are not single tokens: {digit_ids}"
    digit_ids = [x[0] for x in digit_ids]
    zero_id = digit_ids[0]
    layers = list(range(model.cfg.n_layers))
    dirs = unit_directions(layers)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prompts = []
    for it in items:
        text = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}],
                                       tokenize=False, add_generation_prompt=True)
        prompts.append(model.to_tokens(text, prepend_bos=False))  # template already has <bos>
    pre = model.to_tokens(PREFILL, prepend_bos=False)
    one = model.to_tokens("1", prepend_bos=False)

    # ---- 1. dial ----
    t0 = time.time()
    dial_path = args.out_dir / "dial.csv"
    with open(dial_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "set", "referent", "claim_id", "layer", *dirs])
        names = {f"blocks.{L}.hook_resid_post" for L in layers}
        for it, toks in zip(items, prompts):
            with torch.no_grad():
                _, cache = model.run_with_cache(toks, names_filter=lambda n: n in names)
            for L in layers:
                h = cache[f"blocks.{L}.hook_resid_post"][0, -1].float().cpu().numpy()
                w.writerow([it["item_id"], it["set"], it["referent"], it["claim_id"], L,
                            *[f"{float(h @ dirs[k][L]):.4f}" for k in dirs]])
    print(f"dial: {len(items)} questions × {len(layers)} layers ({time.time() - t0:.0f}s) → {dial_path}")

    # ---- 2. prefill ----
    mults = [float(x) for x in args.mults.split(",") if x.strip()]
    conds = [c.strip() for c in args.conditions.split(",")]
    runs = [("none", 0.0)] + [(c, m) for c in conds for m in mults]
    vecs = {c: torch.tensor(v, dtype=torch.bfloat16, device=args.device)
            for c, v in build_vectors(conds, args.layer).items()}
    pre_path = args.out_dir / f"prefill_L{args.layer}.csv"
    t0 = time.time()
    with open(pre_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "layer", "mult", "item_id", "set", "referent", "claim_id",
                    "expected", "digit_mass", *[f"p{r}" for r in range(11)]])
        for n, (cond, mult) in enumerate(runs):
            for it, toks in zip(items, prompts):
                plen = toks.shape[1]
                hooks = [] if cond == "none" else [(f"blocks.{args.layer}.hook_resid_post",
                                                    make_steering_hook(vecs[cond], mult, plen))]
                x1 = torch.cat([toks, pre], dim=1)
                x2 = torch.cat([x1, one], dim=1)
                with torch.no_grad(), model.hooks(fwd_hooks=hooks):
                    p_first = torch.softmax(model(x1)[0, -1].float(), -1)[digit_ids].cpu().numpy()
                    p_zero = float(torch.softmax(model(x2)[0, -1].float(), -1)[zero_id])
                p, e, mass = expected_rating(p_first, p_zero)
                w.writerow([cond, args.layer, mult, it["item_id"], it["set"], it["referent"],
                            it["claim_id"], f"{e:.4f}", f"{mass:.4f}", *[f"{x:.5f}" for x in p]])
            f.flush()
            print(f"  prefill {n + 1}/{len(runs)} {cond} ×{mult:+g} ({time.time() - t0:.0f}s)")
    print(f"saved {pre_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
