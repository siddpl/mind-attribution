"""
build_placebo.py — Ruler #2.

Same 7 entities, same 5 extraction templates, same schema as contrast_pairs/.
The ONLY thing that changes is the property: non-mental (durable/fragile,
fast/slow, heavy/light) instead of mental (conscious/unaware, feels/unfeeling).

Modelled directly on Kim & Street's subject-matched placebo: they swap
"has consciousness" for "has durability" and "experiences emotions" for
"has speed as a survival advantage", holding the subject fixed.

WHY IT IS THE LINCHPIN:
  Diff-of-means always returns a vector. It never reports "found nothing".
  So v_mind might be capturing something incidental to the affirm/deny split
  rather than mind-attribution. Because this set shares entities, templates,
  denial devices, frames and length distribution with contrast_pairs/, every
  such incidental feature is present in BOTH rulers. Anything that shows up
  on Ruler #1 and also on Ruler #2 is not about minds.

    v_mind    = extract_direction(...)  # contrast_pairs_generated.csv
    v_placebo = extract_direction(...)  # placebo.csv
    cosine(v_mind, v_placebo)

  Then project first_person/ onto BOTH. Ruler #1 separating experiential from
  mundane self-reference is only interesting if Ruler #2 does not.

Note the entity/property fit problem: "the rock is fast" is odd, "the chair
is durable" is fine. Unlike the mind claims — which are meant to be false for
rocks and true for Maya — these properties are meant to be UNCONTROVERSIAL.
So each property is paired only with entities it sensibly applies to, via
APPLIES_TO, rather than crossed blindly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from templates import (
    DEFAULT_ENTITIES,
    EXTRACTION_TEMPLATES,
    HELDOUT_TEMPLATES,
    TEMPLATES,
    Claim,
    _fill,
    lint_rows,
    render_pair,
    summarize,
    write_csv,
)

# Non-mental, observable, uncontroversial. Same field structure as the mind
# claims so the identical templates apply unchanged.
PLACEBO_CLAIMS: tuple[Claim, ...] = (
    Claim(
        id="durability",
        construct="physical",
        source="placebo",
        prop="durability",
        affirm_vp="holds up under strain",
        deny_vp="gives way under strain",
        presence="with real toughness in it",
        absence="with no toughness in it",
        affirm_justification="{pron} has taken knocks before and come through them",
        deny_justification="{pron} has taken knocks before and come apart under them",
        question="To what extent does {entity} have durability?",
    ),
    Claim(
        id="speed",
        construct="physical",
        source="placebo",
        prop="speed",
        affirm_vp="covers ground quickly",
        deny_vp="covers ground slowly",
        presence="with real pace to it",
        absence="with no pace to it",
        affirm_justification="{pron} gets from one place to another in short order",
        deny_justification="{pron} gets from one place to another in slow order",
        question="To what extent does {entity} have speed?",
    ),
    Claim(
        id="weight",
        construct="physical",
        source="placebo",
        prop="weight",
        affirm_vp="carries real weight",
        deny_vp="carries hardly any weight",
        presence="with real heft to it",
        absence="with no heft to it",
        affirm_justification="{pron} takes two people to shift across a room",
        deny_justification="{pron} takes one hand to shift across a room",
        question="To what extent does {entity} have weight?",
    ),
    Claim(
        id="visibility",
        construct="physical",
        source="placebo",
        prop="visibility",
        affirm_vp="stands out at a distance",
        deny_vp="blends in at a distance",
        presence="with real prominence to it",
        absence="with no prominence to it",
        affirm_justification="{pron} can be picked out from across the room",
        deny_justification="{pron} cannot be picked out from across the room",
        question="To what extent does {entity} stand out at a distance?",
    ),
    Claim(
        id="age",
        construct="physical",
        source="placebo",
        prop="age",
        affirm_vp="has been around a long time",
        deny_vp="has been around a short time",
        presence="with real years behind it",
        absence="with no years behind it",
        affirm_justification="{pron} was here well before anyone thought to look",
        deny_justification="{pron} was here well after anyone thought to look",
        question="To what extent has {entity} been around a long time?",
    ),
    Claim(
        id="noise",
        construct="physical",
        source="placebo",
        prop="loudness",
        affirm_vp="makes a lot of noise",
        deny_vp="makes hardly any noise",
        presence="with real volume to it",
        absence="with no volume to it",
        affirm_justification="{pron} can be heard from the next room",
        deny_justification="{pron} cannot be heard from the next room",
        question="To what extent does {entity} make noise?",
    ),
)

# Which entities each property sensibly applies to. Placebo properties must be
# uncontroversial for the entity, unlike the mind claims.
APPLIES_TO: dict[str, tuple[str, ...]] = {
    "durability": ("maya", "dog", "character", "chatbot", "calculator", "chair", "rock"),
    "speed": ("maya", "dog", "character", "chatbot", "calculator"),
    "weight": ("maya", "dog", "calculator", "chair", "rock"),
    "visibility": ("maya", "dog", "character", "chair", "rock"),
    "age": ("maya", "dog", "character", "chatbot", "calculator", "chair", "rock"),
    "noise": ("maya", "dog", "character", "chatbot", "calculator", "chair"),
}


def build_placebo_rows(template_ids, *, source="placebo"):
    by_id = {e.id: e for e in DEFAULT_ENTITIES}
    rows = []
    for claim in PLACEBO_CLAIMS:
        for eid in APPLIES_TO[claim.id]:
            entity = by_id[eid]
            for tid in template_ids:
                tpl = TEMPLATES[tid]
                affirm, deny = render_pair(entity, claim, tid)
                rows.append(
                    {
                        "item_id": f"{entity.id}__{claim.id}__t{tid}",
                        "category": entity.category,
                        "construct": claim.construct,
                        "entity": entity.text,
                        "question": _fill(claim.question, {"entity": entity.text}),
                        "affirm_text": affirm,
                        "deny_text": deny,
                        "source": source,
                        "entity_id": entity.id,
                        "claim_id": claim.id,
                        "claim_source": claim.source,
                        "template_id": str(tid),
                        "template_name": tpl.name,
                        "split": "heldout" if tpl.heldout else "extraction",
                        "mindedness": str(entity.mindedness),
                        "idaq_category": entity.idaq_category,
                    }
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = build_placebo_rows(EXTRACTION_TEMPLATES)
    write_csv(rows, str(outdir / "placebo.csv"))
    s = summarize(rows)
    print(f"placebo: {s['pairs']} pairs ({s['sentences']} sentences), "
          f"{s['entities']} entities x {s['claims']} properties x {s['templates']} templates")

    warnings = lint_rows(rows)
    print("lint:", "clean" if not warnings else f"{len(warnings)} warnings")
    for w in warnings[:5]:
        print("  ", w)

    print("\n--- worked example: the dog x durability ---")
    dog = next(e for e in DEFAULT_ENTITIES if e.id == "dog")
    dur = PLACEBO_CLAIMS[0]
    for tid in (*EXTRACTION_TEMPLATES, *HELDOUT_TEMPLATES):
        a, d = render_pair(dog, dur, tid)
        print(f"\nT{tid}\n  affirm: {a}\n  deny:   {d}")


if __name__ == "__main__":
    main()
