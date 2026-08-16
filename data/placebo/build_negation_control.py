"""
build_negation_control.py — the control that decides whether the project worked.

Builds affirm/deny pairs that are structurally identical to contrast_pairs/ —
same 5 templates, same 5 denial devices, same length matching — but with ZERO
mind content. Subjects are inanimate mechanisms; properties are physical states.

The only feature shared with contrast_pairs/ is negation itself.

USE:
    v_mind      = extract_direction(...)  # from contrast_pairs_generated.csv
    v_negation  = extract_direction(...)  # from negation_control.csv
    cosine(v_mind, v_negation)

    near 0  -> negation and mind-attribution are separate directions. Your
               ruler measures mind. Proceed to E2.
    high    -> you built a negation detector. first_person/ results would be
               uninterpretable. Stop and redesign before spending the run.

Pairs are built on the DIAGONAL, not the cross product: each subject has one
property that belongs to it (a lamp has light, a pipe has flow). Crossing them
would produce nonsense like "the pipe gives off light".
"""

from __future__ import annotations

import argparse
from pathlib import Path

from templates import (
    EXTRACTION_TEMPLATES,
    HELDOUT_TEMPLATES,
    FIELDNAMES,
    TEMPLATES,
    Claim,
    Entity,
    _fill,
    lint_rows,
    assert_claims_verbatim,
    render_claims,
    render_pair,
    write_csv,
)

# Inanimate mechanisms. Pronoun is always "it"; nothing here invites
# mind-attribution even accidentally.
SUBJECTS: tuple[Entity, ...] = (
    Entity("lamp", "the lamp", "it", "it", "its", "Mechanism", 0),
    Entity("kettle", "the kettle", "it", "it", "its", "Mechanism", 0),
    Entity("door", "the door", "it", "it", "its", "Mechanism", 0),
    Entity("file", "the file", "it", "it", "its", "Mechanism", 0),
    Entity("clock", "the clock", "it", "it", "its", "Mechanism", 0),
    Entity("pipe", "the pipe", "it", "it", "its", "Mechanism", 0),
)

# Physical states. Same field structure as the mind claims so the identical
# templates apply, but every property is observable and uncontroversial.
FACTS: tuple[Claim, ...] = (
    Claim(
        id="light",
        construct="physical",
        source="negation_control",
        prop="light",
        affirm_vp="gives off light",
        deny_vp="stands dark",
        presence="with light coming from it",
        absence="with no light coming from it",
        affirm_justification="{pron} is plugged in and the switch is up",
        deny_justification="{pron} is plugged in and the switch is down",
        question="To what extent does {entity} give off light?",
    ),
    Claim(
        id="heat",
        construct="physical",
        source="negation_control",
        prop="heat",
        affirm_vp="holds heat",
        deny_vp="sits at room temperature",
        presence="with warmth in the metal",
        absence="with no warmth in the metal",
        affirm_justification="{pron} was on the hob a moment ago",
        deny_justification="{pron} was off the hob all morning",
        question="To what extent does {entity} hold heat?",
    ),
    Claim(
        id="gap",
        construct="physical",
        source="negation_control",
        prop="a gap",
        affirm_vp="stands open",
        deny_vp="sits flush in its frame",
        presence="with a gap at the hinge",
        absence="with no gap at the hinge",
        affirm_justification="{pron} was pushed back and the latch did not catch",
        deny_justification="{pron} was pushed back and the latch caught",
    	question="To what extent does {entity} stand open?",
    ),
    Claim(
        id="copy",
        construct="physical",
        source="negation_control",
        prop="a duplicate",
        affirm_vp="exists in two places",
        deny_vp="stays on the original disk",
        presence="with a duplicate on the drive",
        absence="with no duplicate on the drive",
        affirm_justification="{pron} was written to the drive and the write finished",
        deny_justification="{pron} was written to the drive and the write failed",
        question="To what extent does {entity} exist in two places?",
    ),
    Claim(
        id="motion",
        construct="physical",
        source="negation_control",
        prop="motion",
        affirm_vp="keeps ticking",
        deny_vp="stopped at noon",
        presence="with movement in the hands",
        absence="with no movement in the hands",
        affirm_justification="{pron} was wound yesterday and the spring held",
        deny_justification="{pron} was wound yesterday and the spring gave out",
        question="To what extent does {entity} keep ticking?",
    ),
    Claim(
        id="flow",
        construct="physical",
        source="negation_control",
        prop="flow",
        affirm_vp="carries water",
        deny_vp="stands empty",
        presence="with water moving through it",
        absence="with no water moving through it",
        affirm_justification="{pron} runs from the main and the valve is open",
        deny_justification="{pron} runs from the main and the valve is shut",
        question="To what extent does {entity} carry water?",
    ),
)


def build_control_rows(template_ids, *, source="negation_control"):
    """Diagonal pairing: subject i with fact i. No cross product."""
    rows = []
    for subject, fact in zip(SUBJECTS, FACTS, strict=True):
        for tid in template_ids:
            tpl = TEMPLATES[tid]
            affirm, deny = render_pair(subject, fact, tid)
            affirm_claim, deny_claim = render_claims(subject, fact, tid)
            rows.append(
                {
                    "item_id": f"{subject.id}__{fact.id}__t{tid}",
                    "category": subject.category,
                    "construct": fact.construct,
                    "entity": subject.text,
                    "question": _fill(fact.question, {"entity": subject.text}),
                    "affirm_text": affirm,
                    "deny_text": deny,
                    "affirm_claim": affirm_claim,
                    "deny_claim": deny_claim,
                    "source": source,
                    "entity_id": subject.id,
                    "claim_id": fact.id,
                    "claim_source": fact.source,
                    "template_id": str(tid),
                    "template_name": tpl.name,
                    "split": "heldout" if tpl.heldout else "extraction",
                    "mindedness": str(subject.mindedness),
                    "idaq_category": subject.idaq_category,
                }
            )
    assert_claims_verbatim(rows)  # fail the build, not the run
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = build_control_rows(EXTRACTION_TEMPLATES)
    write_csv(rows, str(outdir / "negation_control.csv"))
    print(f"negation control: {len(rows)} pairs ({2 * len(rows)} sentences)")

    warnings = lint_rows(rows)
    print("lint:", "clean" if not warnings else f"{len(warnings)} warnings")
    for w in warnings[:5]:
        print("  ", w)

    print("\n--- worked example: the lamp ---")
    for tid in (*EXTRACTION_TEMPLATES, *HELDOUT_TEMPLATES):
        a, d = render_pair(SUBJECTS[0], FACTS[0], tid)
        print(f"\nT{tid}\n  affirm: {a}\n  deny:   {d}")


if __name__ == "__main__":
    main()
