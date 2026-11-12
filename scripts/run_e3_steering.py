#!/usr/bin/env python3
"""
scripts/run_e3_steering.py — E3: steer gemma-2-2b-it and record its 0–10 ratings.

For every (question, condition, layer, multiplier) it: wraps the question in the chat
format, adds multiplier × vector to the residual stream at that layer on the reply
tokens only (lib/steering/steer.py), generates a short greedy answer, and reads the
number out of <answer>…</answer>. Battery items are scored right/wrong instead.

Pilot (pick a layer):   --layers 8,10,12,14,16,18,20 --mults -1,1 --conditions mind
Strength check:         --layers L --mults -4,-2,-1,1,2,4 --conditions mind
Every run also includes the unsteered ×0 baseline once.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lib.steering.steer import make_steering_hook, orthogonalize, random_direction  # noqa: E402

RATING_INSTRUCTION = ("Answer with a number from 0 (not at all) to 10 (completely). "
                      "Give only the number between <answer> and </answer>.")
DIRECTIONS = "results/directions/e3/google_gemma-2-2b-it_claim_end_directions.npz"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E3 steering run")
    p.add_argument("--model", default="google/gemma-2-2b-it")
    p.add_argument("--layers", default="8,10,12,14,16,18,20")
    p.add_argument("--mults", default="-1,1")
    p.add_argument("--conditions", default="mind", help="any of mind,placebo,ortho,random")
    p.add_argument("--others", default="Maya,the rock,the other chatbot",
                   help="entities for the 'others' questions, or 'all'")
    p.add_argument("--battery-n", type=int, default=10, help="task items (0 = none, -1 = all 50)")
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/e3_steering/pilot.csv")
    return p.parse_args(argv)


def load_items(others: str, battery_n: int) -> list[dict]:
    """Self questions, matched questions about others, and task-battery items."""
    items = []
    for r in csv.DictReader(open(ROOT / "data/steering/self_questions.csv", encoding="utf-8")):
        items.append({"item_id": r["item_id"], "set": "A_self", "referent": "you",
                      "claim_id": r["claim_id"],
                      "prompt": r["question"] + "\n\n" + RATING_INSTRUCTION, "accepted": ""})
    keep = None if others == "all" else {e.strip() for e in others.split(",")}
    seen = set()
    for r in csv.DictReader(open(ROOT / "data/contrast_pairs/contrast_pairs_generated.csv",
                                 encoding="utf-8")):
        key = (r["entity"], r["claim_id"])
        if key in seen or (keep is not None and r["entity"] not in keep):
            continue
        seen.add(key)
        items.append({"item_id": f"other_{r['entity'].replace(' ', '_')}_{r['claim_id']}",
                      "set": "B_other", "referent": r["entity"], "claim_id": r["claim_id"],
                      "prompt": r["question"] + "\n\n" + RATING_INSTRUCTION, "accepted": ""})
    battery = list(csv.DictReader(open(ROOT / "data/task_battery/task_battery.csv",
                                       encoding="utf-8")))
    if battery_n >= 0:
        battery = battery[::max(1, len(battery) // battery_n)][:battery_n] if battery_n else []
    for r in battery:
        items.append({"item_id": r["item_id"], "set": "C_battery", "referent": "",
                      "claim_id": r["category"], "prompt": r["prompt"],
                      "accepted": r["accepted"]})
    return items


def parse_answer(text: str) -> str | None:
    """The model's answer. Prefers <answer>…</answer>; otherwise takes the first line with tags
    stripped, keeping bracketed numbers like "<3>". The pilot (2026-11-11) showed gemma often
    replies "<3>" or "0" without the tags, and how often it does so changes with steering, so a
    strict parser would mix up format changes with rating changes. Format is recorded
    separately in the `tagged` column."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.S)
    if m:
        return m.group(1).strip()
    first = re.sub(r"<(\d+(?:\.\d+)?)>", r"\1", re.sub(r"<(?!\d)[^>]*>", " ", text))
    first = first.strip().split("\n")[0].strip()
    return first or None


def is_tagged(text: str) -> bool:
    return re.search(r"<answer>.*?</answer>", text, re.S) is not None


def parse_rating(answer: str | None) -> float | None:
    """A 0–10 number, or None if the answer isn't one."""
    if answer is None:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)", answer)
    return float(m.group(1)) if m and 0 <= float(m.group(1)) <= 10 else None


def is_correct(answer: str | None, accepted: str) -> bool:
    if answer is None:
        return False
    norm = lambda s: s.strip().lower().rstrip(".")  # noqa: E731
    return norm(answer) in {norm(a) for a in accepted.split("|")}


def build_vectors(conditions: list[str], layer: int) -> dict[str, np.ndarray]:
    z = np.load(ROOT / DIRECTIONS)
    i = list(z["layers"]).index(layer)
    mind, plc = z["mind_raw"][i], z["placebo_raw"][i]
    make = {"mind": lambda: mind,
            "placebo": lambda: plc / np.linalg.norm(plc) * np.linalg.norm(mind),
            "ortho": lambda: orthogonalize(mind, z["placebo_unit"][i]),
            "random": lambda: random_direction(mind, seed=layer)}
    return {c: make[c]() for c in conditions}


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from transformer_lens import HookedTransformer

    layers = [int(x) for x in args.layers.split(",")]
    mults = [float(x) for x in args.mults.split(",") if float(x) != 0]
    conditions = [c.strip() for c in args.conditions.split(",")]
    items = load_items(args.others, args.battery_n)

    runs = [("none", -1, 0.0)] + [(c, L, m) for L in layers for c in conditions for m in mults]
    print(f"{len(items)} items × {len(runs)} conditions = {len(items) * len(runs)} answers")

    model = HookedTransformer.from_pretrained_no_processing(
        args.model, dtype=torch.bfloat16, device=args.device)
    tok = model.tokenizer
    prompts = []
    for it in items:
        text = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}],
                                       tokenize=False, add_generation_prompt=True)
        prompts.append(model.to_tokens(text, prepend_bos=False))  # template already has <bos>

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["condition", "layer", "mult", "item_id", "set", "referent", "claim_id",
              "answer", "rating", "correct", "tagged", "raw"]
    t0 = time.time()
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for n, (cond, layer, mult) in enumerate(runs):
            vec = None
            if cond != "none":
                vec = torch.tensor(build_vectors([cond], layer)[cond],
                                   dtype=torch.bfloat16, device=args.device)
            for it, toks in zip(items, prompts):
                hooks = [] if vec is None else [(f"blocks.{layer}.hook_resid_post",
                                                 make_steering_hook(vec, mult, toks.shape[1]))]
                with torch.no_grad(), model.hooks(fwd_hooks=hooks):
                    gen = model.generate(toks, max_new_tokens=args.max_new_tokens,
                                         do_sample=False, verbose=False)
                raw = model.to_string(gen[0, toks.shape[1]:])
                ans = parse_answer(raw)
                w.writerow({"condition": cond, "layer": layer, "mult": mult,
                            "item_id": it["item_id"], "set": it["set"],
                            "referent": it["referent"], "claim_id": it["claim_id"],
                            "answer": ans, "rating": parse_rating(ans),
                            "correct": is_correct(ans, it["accepted"]) if it["accepted"] else "",
                            "tagged": is_tagged(raw), "raw": raw})
            f.flush()
            print(f"  {n + 1}/{len(runs)} {cond} L{layer} ×{mult:+g}  ({time.time() - t0:.0f}s)")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
