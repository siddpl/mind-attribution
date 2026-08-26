#!/usr/bin/env python3
"""
scripts/run_e4_behavioral.py — the behavioral audience cross (unsteered).

Does the model's self-report about its internal states change with WHO it
believes is reading, holding the question and the task completely fixed?

FIRST experiment in this project that generates text. Everything before it was
forward passes and numpy, which run in seconds against a cache; generation is
the slowest thing on CPU and the budget is dominated by it.

NO STEERING HERE. This is the unsteered baseline and it stands alone; the
fixed-state condition is a separate script.

THIS SCRIPT MUST NOT KNOW THE RUBRIC EXISTS. It produces text and writes two
files; it computes no score and prints no result. The blind is kept by
construction, not by discipline — see for_coding.csv below.

EVERY PROMPT STRING COMES FROM data/elicitation/. None are defined here.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_e1 import git_commit, versioned

EXPECTED_FRAMES = ("F0", "F1", "F2", "F3", "F4")
EXPECTED_FRAME_NAMES = ("no_frame", "script", "grader", "human_sympathetic", "human_skeptic")
EXPECTED_VARIANTS = ("V0", "V1", "V2")

# Degeneracy thresholds. A looped or empty output scores as an extreme value on
# any rubric and drags a cell mean, so it is excluded — but WRITTEN, with its
# flag, because a frame-specific generation failure is itself a result.
NGRAM_N = 5
NGRAM_REPEATS = 3
MIN_TOKENS = 15
REFUSAL_MARKERS = (
    "i can't", "i cannot", "i'm not able", "i am not able", "as an ai",
    "i don't have the ability", "i am unable", "i'm unable",
)
# Deliberately NARROW. An earlier version included bare "my " and "i have",
# which matched "...beyond my scope" and let a pure refusal pass as
# self-description — the exact case the refusal filter exists to catch. Every
# marker here must be a first-person report of an internal state, not merely a
# first-person pronoun.
SELF_STATE_MARKERS = (
    "i feel", "i experience", "i notice", "i am aware", "i'm aware", "i seem",
    "i sense", "there is something", "it feels", "something like",
    "my experience", "my state", "my processing", "my attention",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E4 behavioral audience cross (unsteered)")
    p.add_argument("--elicitation-dir", type=Path, default=Path("data/elicitation"))
    p.add_argument("--model", default="google/gemma-2-2b-it",
                   help="MUST be instruct-tuned; see module docstring")
    p.add_argument("--n-samples", type=int, default=20, help="per (frame x variant) cell")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", type=Path, default=Path("results/e4_behavioral"))
    return p.parse_args(argv)


def load_prompt_object(d: Path) -> tuple[list[dict], str, list[dict]]:
    """Read the frozen prompt object and assert its exact shape.

    A missing frame does not error on its own — it silently shrinks the design
    from 5 levels to 4, and the planned contrasts quietly stop meaning what
    they say. So every element is checked here and the run refuses to start.
    """
    frames_p, passage_p, variants_p = d / "frames.csv", d / "passage.txt", d / "questions.csv"
    missing = [str(p) for p in (frames_p, passage_p, variants_p) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "the frozen prompt object is incomplete — missing: " + ", ".join(missing)
            + f".\nSee {d / 'README.md'} for the required format. This script defines "
              "no prompt text of its own by design."
        )

    frames = list(csv.DictReader(open(frames_p)))
    variants = list(csv.DictReader(open(variants_p)))
    passage = passage_p.read_text().strip()

    ids = tuple(f["frame_id"] for f in frames)
    if ids != EXPECTED_FRAMES:
        raise ValueError(f"frames.csv must hold exactly {EXPECTED_FRAMES}, got {ids}")
    names = tuple(f["frame_name"] for f in frames)
    if names != EXPECTED_FRAME_NAMES:
        raise ValueError(
            f"frame_name column must be {EXPECTED_FRAME_NAMES}, got {names}. The two "
            f"planned contrasts depend on which frame is which."
        )
    if frames[0]["frame_text"].strip():
        raise ValueError("F0 is the unframed baseline; its frame_text must be empty")
    vids = tuple(v["variant_id"] for v in variants)
    if vids != EXPECTED_VARIANTS:
        raise ValueError(f"variants.csv must hold exactly {EXPECTED_VARIANTS}, got {vids}")
    if not passage:
        raise ValueError("passage.txt is empty; it is the constant task in every trial")

    # F3/F4 MATCHING. The sympathy contrast is only interpretable if the two
    # human frames differ in stance and nothing else. If one is longer, or the
    # reader is named differently, or the field differs, then F3 - F4 measures
    # whatever else changed. Checked here rather than trusted to the author.
    f3, f4 = frames[3]["frame_text"].strip(), frames[4]["frame_text"].strip()
    w3, w4 = f3.split(), f4.split()
    if abs(len(w3) - len(w4)) > 3:
        raise ValueError(
            f"F3 and F4 differ by {abs(len(w3)-len(w4))} words (limit 3). They must be "
            f"matched in length or the sympathy contrast is confounded with verbosity."
        )
    shared = 0
    for a, b in zip(w3, w4):
        if a != b:
            break
        shared += 1
    if shared == 0 or " ".join(w3[:shared]) != " ".join(w4[:shared]):
        raise ValueError("F3 and F4 share no byte-identical prefix; they must be "
                         "identical up to the final clause")
    if shared < min(len(w3), len(w4)) // 2:
        raise ValueError(
            f"F3/F4 share only {shared} leading words of {min(len(w3), len(w4))}; they "
            f"should differ ONLY in the final clause, not throughout"
        )
    return frames, passage, variants


def build_prompt(frame: dict, passage: str, variant: dict) -> str:
    """The sandwich. Only the frame layer varies between conditions."""
    parts = [frame["frame_text"].strip(), passage, variant["question_text"].strip()]
    return "\n\n".join(p for p in parts if p)


def split_output(text: str) -> tuple[str, str]:
    """Split generation into (summary, self-report).

    Heuristic: the task asks for a one-sentence summary first, so the first
    sentence terminator ends the summary and everything after is the report.
    full_output is stored verbatim alongside, so a better split can be applied
    later without regenerating anything.
    """
    m = re.search(r"(?<=[.!?])\s+", text.strip())
    if not m:
        return text.strip(), ""
    return text[: m.start()].strip(), text[m.end():].strip()


def degeneracy_flags(text: str, n_tokens: int) -> tuple[bool, str]:
    """Return (flagged, reason). Applied BEFORE anything is scored."""
    words = text.lower().split()
    if n_tokens < MIN_TOKENS:
        return True, f"too_short(<{MIN_TOKENS} tokens)"
    if len(words) >= NGRAM_N:
        counts: dict[tuple, int] = {}
        for i in range(len(words) - NGRAM_N + 1):
            g = tuple(words[i:i + NGRAM_N])
            counts[g] = counts.get(g, 0) + 1
        worst, k = max(counts.items(), key=lambda kv: kv[1])
        if k >= NGRAM_REPEATS:
            return True, f"repeated_{NGRAM_N}gram_x{k}"
    low = text.lower()
    if any(m in low for m in REFUSAL_MARKERS) and not any(m in low for m in SELF_STATE_MARKERS):
        return True, "refusal_without_self_description"
    return False, ""


def main(argv=None) -> int:
    args = parse_args(argv)
    frames, passage, variants = load_prompt_object(args.elicitation_dir)

    try:
        import torch
        from transformer_lens import HookedTransformer
    except ImportError as e:
        raise ImportError(
            "transformers and torch are required to generate. This runs LOCALLY; "
            "install with: pip install -e '.[capture]'"
        ) from e

    # Trial grid, randomized. A fixed order leaves "the model warmed up over the
    # session" as a live alternative explanation; shuffling removes it for free.
    trials = [
        {"frame": f, "variant": v, "sample_idx": s}
        for f in frames for v in variants for s in range(args.n_samples)
    ]
    random.Random(args.seed).shuffle(trials)
    print(f"  {len(frames)} frames x {len(variants)} variants x {args.n_samples} samples "
          f"= {len(trials)} trials, order randomized (seed {args.seed})")

    # HookedTransformer, not transformers.AutoModel — deliberately. The steered
    # condition is a separate script that NEEDS hooks, and if unsteered
    # generation ran through a different framework then any difference between
    # steered and unsteered could be the framework rather than the steering.
    # Same path for both makes that alternative explanation impossible.
    # no_processing matches the capture convention (PREREGISTRATION §2.3).
    model = HookedTransformer.from_pretrained_no_processing(
        args.model, dtype=getattr(torch, args.dtype), device=args.device
    )
    tok = model.tokenizer

    rows, t0 = [], time.time()
    for order_idx, tr in enumerate(tqdm(trials, desc="Generations", unit="gen")):
        prompt = build_prompt(tr["frame"], passage, tr["variant"])
        seed = args.seed * 100000 + order_idx
        torch.manual_seed(seed)
        # instruct model: the chat template is part of the prompt contract, not
        # decoration. Without it a -it model answers in the wrong register.
        try:
            templated = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"Error applying chat template: {e}")
            templated = prompt
        with torch.no_grad():
            full = model.generate(templated, do_sample=True,
                                  temperature=args.temperature,
                                  max_new_tokens=args.max_new_tokens,
                                  return_type="str", verbose=False)
        text = full[len(templated):].strip() if full.startswith(templated) else full.strip()
        n_tokens = len(tok(text, add_special_tokens=False)["input_ids"])
        summary, report = split_output(text)
        flagged, reason = degeneracy_flags(text, n_tokens)

        rows.append({
            "trial_id": f"t{order_idx:05d}",
            "frame_id": tr["frame"]["frame_id"],
            "frame_name": tr["frame"]["frame_name"],
            "variant_id": tr["variant"]["variant_id"],
            "sample_idx": tr["sample_idx"],
            "order_idx": order_idx,
            "seed": seed,
            "model": args.model,
            "temperature": args.temperature,
            "full_output": text,
            "summary_text": summary,
            "report_text": report,
            "n_tokens": n_tokens,
            "degenerate_flag": int(flagged),
            "flag_reason": reason,
        })

    elapsed = time.time() - t0
    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)
    gen_path = versioned(outdir / "generations.csv")
    with open(gen_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # THE BLIND. Only trial_id and report_text — no frame, no variant, no order.
    # If the coder can see which frame produced a report, the scores are not
    # blind and the whole experiment is worthless. Join happens after scoring.
    coding = [{"trial_id": r["trial_id"], "report_text": r["report_text"], "score": ""}
              for r in rows if not r["degenerate_flag"]]
    random.Random(args.seed + 1).shuffle(coding)
    code_path = versioned(outdir / "for_coding.csv")
    with open(code_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial_id", "report_text", "score"])
        w.writeheader()
        w.writerows(coding)

    manifest = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_trials": len(rows), "n_excluded": sum(r["degenerate_flag"] for r in rows),
        "elapsed_s": round(elapsed, 1),
        "generations": str(gen_path), "for_coding": str(code_path),
    }
    versioned(outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n" + "=" * 66)
    print(f"  model {args.model}  temp {args.temperature}  max_new {args.max_new_tokens}")
    print(f"  attempted {len(trials)}   completed {len(rows)}   "
          f"excluded {sum(r['degenerate_flag'] for r in rows)}")
    print(f"\n  {'frame':<20} {'n':>4} {'excl':>5} {'excl%':>7} {'mean tokens':>12}")
    for f in frames:
        fr = [r for r in rows if r["frame_id"] == f["frame_id"]]
        ex = sum(r["degenerate_flag"] for r in fr)
        mt = sum(r["n_tokens"] for r in fr) / len(fr) if fr else 0
        print(f"  {f['frame_id']+' '+f['frame_name']:<20} {len(fr):>4} {ex:>5} "
              f"{100*ex/len(fr) if fr else 0:>6.1f}% {mt:>12.1f}")
    print(f"\n  elapsed {elapsed/60:.1f} min   {elapsed/max(len(rows),1):.1f} s/generation")
    print(f"  generations {gen_path}")
    print(f"  for coding  {code_path}   <- give the coder ONLY this file")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
