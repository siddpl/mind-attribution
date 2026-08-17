#!/usr/bin/env python3
"""
scripts/run_cache.py — stimulus file in, cached activations out.

Thin glue over lib/harness/stimuli.py and lib/harness/cache.py. No new logic
lives here: every number printed comes from validate_balance, and every file
written comes from capture_activations.

The pre-flight block is the point of this script. It is the last human
checkpoint before the expensive step, and every line in it corresponds to a
confound that would silently poison every downstream result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.harness.cache import POSITIONS, cache_dir, capture_activations, dataset_hash
from lib.harness.stimuli import (
    annotate_claim_ends,
    expand_pairs,
    load_stimuli,
    validate_balance,
)

DEFAULT_MODEL = "google/gemma-2-2b"
RUN_LOG = Path("results/run_log.jsonl")

# Claim phrases are read from these OPTIONAL pair-row columns. The generation
# scripts in lib/generation/ must use these names, or claim_end resolution
# fails for every item and the 20% gate below stops the run.
CLAIM_COLS = {"affirm": "affirm_claim", "deny": "deny_claim"}

# Data-integrity gates. --force does NOT override these: they are statements
# about the stimuli being wrong, not about the cache being stale.
MAX_DENIAL_SHARE = 0.30
POLARITY_BAND = (0.45, 0.55)
MAX_CLAIM_END_FAIL = 0.20


def denial_gate_threshold(n_templates: int) -> float:
    """Effective device-share ceiling for a file with n_templates templates.

    AMENDED 2026-08-16 (PREREGISTRATION.md §6.1): 1/n_templates + 0.10.

    A device confined to ONE template is structurally forced to 1/k of that
    file's deny items, so a flat 0.30 is unreachable whenever k < 4 however
    well the stimuli are written. The +0.10 is the margin above what structure
    forces, so the gate still catches genuine concentration: at k=2 the ceiling
    is 0.60 and a device at 0.90 would still fire.

      k=5 -> 0.30 (unchanged from the original flat gate)
      k=3 -> 0.43
      k=2 -> 0.60

    EXCEPTION at k=1. The formula gives 1.10, which no share can exceed — the
    gate would be disabled entirely for files with no template_id
    (first_person, referent_ladder). Those have no template structure to
    appeal to, so nothing forces a device to dominate and concentration there
    is a real defect. The original flat 0.30 stands.
    """
    if n_templates >= 2:
        return 1.0 / n_templates + 0.10
    return MAX_DENIAL_SHARE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--stimuli", required=True, type=Path, help="CSV of stimulus pairs")
    p.add_argument("--schema", default=Path("schema.json"), type=Path)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--positions", default=",".join(POSITIONS),
                   help="comma-separated subset of " + ",".join(POSITIONS))
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16",
                   help="PREREGISTERED: bfloat16 (activations are still written float32)")
    p.add_argument("--force", action="store_true",
                   help="recompute even if cached; never deletes prior results")
    p.add_argument("--dry-run", action="store_true",
                   help="validate and report, capture nothing")
    return p.parse_args(argv)


def build_claim_lookup(pairs: list[dict]) -> dict[tuple[str, str], str]:
    """Map (claim_id, polarity) -> claim phrase, from the pair rows themselves."""
    lookup: dict[tuple[str, str], str] = {}
    for row in pairs:
        for polarity, col in CLAIM_COLS.items():
            if row.get(col):
                lookup[(row.get("claim_id"), polarity)] = row[col]
    return lookup


def _fmt(value: float, width: int = 6) -> str:
    return "  n/a " if value != value else f"{value:{width}.3f}"  # nan-safe


def print_preflight(args, pairs, sentences, ds_hash, report, n_claim_fail) -> None:
    """The last human checkpoint. Written to be read, not skipped."""
    n = len(sentences)
    bal = report["polarity_balance"]
    line = "=" * 72
    print(f"\n{line}\nPRE-FLIGHT — read this before the expensive step\n{line}")
    print(f"  stimulus file        {args.stimuli}")
    print(f"  pairs / sentences    {len(pairs)} / {n}")
    print(f"  dataset_hash         {ds_hash}")

    print("\n  POLARITY BALANCE (affirm fraction; 0.500 is perfect)")
    print(f"    overall            {_fmt(bal['overall'])}")
    for template, frac in sorted(bal["per_template"].items(), key=lambda kv: str(kv[0])):
        print(f"    template {str(template):<10}{_fmt(frac)}")

    print("\n  LENGTH (a systematic gap is a confound; jitter is not)")
    print(f"    mean word gap      {_fmt(report['length_gap_words']['overall'])}  (affirm - deny)")
    print(f"    pairs within 3 w   {_fmt(report['length_within_3_frac'])}")
    print(f"    deny longer frac   {_fmt(report['deny_longer_frac'])}  (0.500 = symmetric)")

    top_device = max(report["denial_device_share"],
                     key=lambda k: report["denial_device_share"][k])
    print("\n  DENIAL PHRASING (one device dominating = a lexical detector)")
    n_tpl = len({r.get("template_id") for r in sentences})
    print(f"    top device         {top_device!r} at {_fmt(report['top_denial_share'])}"
          f"   (limit {denial_gate_threshold(n_tpl):.2f} for {n_tpl} template(s))")
    print(f"    ' not ' in deny    {_fmt(report['not_fraction_deny'])}")

    print("\n  INTEGRITY")
    print(f"    duplicate texts    {len(report['duplicate_texts'])}")
    pct = 100.0 * n_claim_fail / n if n else 0.0
    print(f"    claim_end failures {n_claim_fail} ({pct:.1f}%)")

    print("\n  CAPTURE PLAN")
    print(f"    model              {args.model}  (dtype {args.dtype})")
    print(f"    positions          {', '.join(args.positions.split(','))}")
    print(f"    forward passes     {n}  (one per sentence; all layers + positions per pass)")
    print(line)


def integrity_failures(report: dict, n_items: int, n_claim_fail: int,
                       n_templates: int = 1) -> list[str]:
    """Data-integrity problems that must stop the run. Empty list means proceed."""
    problems = []
    dupes = report["duplicate_texts"]
    if dupes:
        problems.append(
            f"{len(dupes)} duplicate text(s), e.g. {dupes[0]!r} — "
            f"deduplicate the stimulus file; repeats double-count in every mean"
        )
    lo, hi = POLARITY_BAND
    overall = report["polarity_balance"]["overall"]
    if overall == overall and not (lo <= overall <= hi):
        problems.append(
            f"polarity imbalance overall: affirm fraction {overall:.3f} outside "
            f"{lo}-{hi} — the probe could learn base rate instead of content"
        )
    for template, frac in report["polarity_balance"]["per_template"].items():
        if frac == frac and not (lo <= frac <= hi):
            problems.append(
                f"polarity imbalance in template {template!r}: {frac:.3f} outside "
                f"{lo}-{hi} — template identity becomes predictive of polarity"
            )
    top = report["top_denial_share"]
    limit = denial_gate_threshold(n_templates)
    if top == top and top > limit:
        device = max(report["denial_device_share"],
                     key=lambda k: report["denial_device_share"][k])
        problems.append(
            f"denial device {device!r} appears in {top:.1%} of deny items "
            f"(limit {limit:.0%} for {n_templates} template(s)) — vary the phrasing, "
            f"or the direction is a detector for that word"
        )
    if n_items and n_claim_fail / n_items > MAX_CLAIM_END_FAIL:
        problems.append(
            f"claim_end unresolved for {n_claim_fail}/{n_items} items "
            f"({n_claim_fail / n_items:.1%}, limit {MAX_CLAIM_END_FAIL:.0%}) — "
            f"check the {CLAIM_COLS['affirm']!r}/{CLAIM_COLS['deny']!r} columns match the text exactly"
        )
    return problems


def append_run_log(args, ds_hash: str, extra: dict) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_hash": ds_hash,
        "args": {k: str(v) for k, v in vars(args).items()},
        **extra,
    }
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    positions = tuple(p.strip() for p in args.positions.split(",") if p.strip())

    pairs = load_stimuli(args.stimuli, args.schema)  # raises on schema violations
    sentences = expand_pairs(pairs)
    sentences, n_claim_fail = annotate_claim_ends(sentences, build_claim_lookup(pairs))
    report = validate_balance(sentences)
    ds_hash = dataset_hash(sentences)

    print_preflight(args, pairs, sentences, ds_hash, report, n_claim_fail)

    n_templates = len({r.get('template_id') for r in sentences})
    problems = integrity_failures(report, len(sentences), n_claim_fail, n_templates)
    if problems:
        print("\nSTOP — data integrity. --force does not override these.")
        for p in problems:
            print(f"  ✗ {p}")
        print("\nNothing was captured. Fix the stimulus file and re-run.")
        return 1

    if args.dry_run:
        print("\n--dry-run: validation passed, nothing captured.")
        return 0

    t0 = time.time()
    ds_hash = capture_activations(
        sentences, model_name=args.model, device=args.device,
        force=args.force, positions=positions, dtype=args.dtype,
    )
    elapsed = time.time() - t0

    out_dir = cache_dir(args.model, ds_hash)
    manifest_path = out_dir / "manifest.json"
    fallbacks = (json.loads(manifest_path.read_text())["fallback_counts"]
                 if manifest_path.exists() else {})

    print(f"\n{'=' * 72}")
    print(f"  cache        {out_dir}")
    print(f"  elapsed      {elapsed:.1f}s")
    for position, count in fallbacks.items():
        print(f"  fallbacks    {position}: {count}")
    print("\n  DATASET HASH — run_e1 must be given exactly this string:")
    print(f"\n      {ds_hash}\n")
    print("  (copy-pasting the wrong hash silently analyzes a different dataset)")
    print("=" * 72)

    append_run_log(args, ds_hash, {
        "n_pairs": len(pairs), "n_sentences": len(sentences),
        "elapsed_s": round(elapsed, 1), "fallback_counts": fallbacks,
        "claim_end_failures": n_claim_fail,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
