"""
build_contrast_pairs.py — Stage 3 cross-product.

Writes two files:
  contrast_pairs_generated.csv  — entities x claims x templates 1-5 (extraction)
  contrast_pairs_heldout.csv    — entities x claims x templates 6-7 (test only)

The held-out file exists so you can eyeball it, but nothing in the extraction
pipeline should ever read it. Column layout is a superset of
contrast_pairs_seed_tagged.csv, so compare_full_vs_experience.py can point at
either file without changes.

Usage:
    python build_contrast_pairs.py [--outdir .]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from templates import (  # adjust to `from lib.generation.templates import ...`
    DEFAULT_CLAIMS,
    DEFAULT_ENTITIES,
    EXTRACTION_TEMPLATES,
    HELDOUT_TEMPLATES,
    build_rows,
    lint_rows,
    render_pair,
    summarize,
    write_csv,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    extraction = build_rows(template_ids=EXTRACTION_TEMPLATES)
    heldout = build_rows(template_ids=HELDOUT_TEMPLATES, allow_heldout=True)

    write_csv(extraction, str(outdir / "contrast_pairs_generated.csv"))
    write_csv(heldout, str(outdir / "contrast_pairs_heldout.csv"))

    for name, rows in (("extraction", extraction), ("held-out", heldout)):
        s = summarize(rows)
        print(
            f"{name:>10}: {s['pairs']} pairs ({s['sentences']} sentences) = "
            f"{s['entities']} entities x {s['claims']} claims x {s['templates']} templates"
        )
        print(
            f"            experience={s['experience_pairs']}  agency={s['agency_pairs']}"
        )

    print("\n--- lint (extraction set) ---")
    warnings = lint_rows(extraction)
    if not warnings:
        print("clean")
    else:
        by_template: dict[str, int] = {}
        for w in warnings:
            tid = w.split("__t")[1].split(":")[0]
            by_template[tid] = by_template.get(tid, 0) + 1
        print(f"{len(warnings)} warnings, by template: {dict(sorted(by_template.items()))}")
        for w in warnings[:3]:
            print(f"  {w}")
        if len(warnings) > 3:
            print(f"  ... and {len(warnings) - 3} more")

    print("\n--- worked example: the dog x consciousness ---")
    dog = next(e for e in DEFAULT_ENTITIES if e.id == "dog")
    caring = next(c for c in DEFAULT_CLAIMS if c.id == "consciousness")
    for tid in (*EXTRACTION_TEMPLATES, *HELDOUT_TEMPLATES):
        a, d = render_pair(dog, caring, tid)
        print(f"\nT{tid}")
        print(f"  affirm: {a}")
        print(f"  deny:   {d}")


if __name__ == "__main__":
    main()
