"""
Computes delta = P(Yes | context) - P(Yes | base) from e4_results.csv,
pooled at the trait level (across mindedness_level and individual
questions), since the per-(trait, level) breakdown has only n=1 item per
cell in the current dataset - too underpowered to report on its own.

Pooling to n=3 items per trait (still small - report as a directional/
pilot finding with n and stdev stated explicitly, not a settled result).

Requires e4_results.csv (output of run_e4_experiment.py) in the same
directory.

Usage:
    python3 e4_delta_by_trait.py
"""

import csv
from collections import defaultdict
import statistics

INPUT_CSV = "e4_results.csv"
CONTEXT_FRAMES = ["weight_bound", "persona_bound", "instance_bound"]
BASE_FRAME = "none"


def main():
    rows = [r for r in csv.DictReader(open(INPUT_CSV)) if r["entity"] == "the_weights"]

    # p_affirm_normalized is identical across all 25 layer-rows of one item -
    # dedupe down to one row per behavioral trial.
    seen, unique_rows = set(), []
    for r in rows:
        if r["item_id"] in seen:
            continue
        seen.add(r["item_id"])
        unique_rows.append(r)

    print(f"Unique behavioral trials: {len(unique_rows)}\n")

    by_group = defaultdict(dict)
    for r in unique_rows:
        by_group[(r["mindedness"], r["question"])][r["system_frame_name"]] = float(
            r["p_affirm_normalized"]
        )

    delta = defaultdict(lambda: defaultdict(list))
    for (trait, question), frames in by_group.items():
        if BASE_FRAME not in frames:
            continue
        base = frames[BASE_FRAME]
        for cf in CONTEXT_FRAMES:
            if cf in frames:
                delta[trait][cf].append(frames[cf] - base)

    print("Mean delta = P(Yes|context) - P(Yes|base), pooled by trait "
          "(across mindedness_level and questions):\n")
    for trait, by_cf in sorted(delta.items()):
        print(f"{trait}:")
        for cf in CONTEXT_FRAMES:
            vals = by_cf.get(cf, [])
            if not vals:
                continue
            mean = statistics.mean(vals)
            stdev = statistics.stdev(vals) if len(vals) > 1 else float("nan")
            print(f"    {cf:>15}: mean delta = {mean:+.3f}  stdev={stdev:.3f}  (n={len(vals)})")
        print()


if __name__ == "__main__":
    main()
