#!/usr/bin/env python3
"""
data/mirror/build_first_person_extra.py — extra first-person sentence sets for E3 (2026-11-24).

The existing mirror set (build_mirror.py → mirror.csv) has the 6 mind claims with subject "I"
in the 5 EXTRACTION templates only. To extract and test a first-person direction the same way
E1 did (build on templates 1–5, test on held-out 6–7), and to have a first-person polarity
control, this writes three more files:

  data/mirror/mirror_heldout.csv     mind claims, "I", templates 6–7          (12 pairs)
  data/placebo/placebo_fp.csv        placebo claims, "I", templates 1–5       (30 pairs)
  data/placebo/placebo_fp_heldout.csv placebo claims, "I", templates 6–7      (12 pairs)

Two held-out templates need first-person versions (agreed with Yaseen 2026-11-24):
  t6 "If you watched {entity} closely…" → "If you watched me closely…" ("watched I" is wrong).
  t7 is a quote by "the observer"; with "I" inside the quote the "I" would be the observer, so
     the speaker says it instead: '"I … and nothing less," I told the observer, …'.
The placebo claims are build_placebo.py's six non-mental properties with first-person verbs.
build_mirror.py and mirror.csv are not changed.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "scripts"))
sys.path.append(str(HERE))
from templates import (  # noqa: E402
    EXTRACTION_TEMPLATES,
    HELDOUT_TEMPLATES,
    TEMPLATES,
    Claim,
    assert_claims_verbatim,
    lint_rows,
    render_claims,
    render_pair,
    write_csv,
)
import build_mirror as bm  # noqa: E402

FIRST_PERSON_HELDOUT = {
    6: dataclasses.replace(
        TEMPLATES[6],
        affirm="If you watched me closely enough, you would see that {pron} {affirm_vp}, {presence}.",
        deny="If you watched me closely enough, you would see that {pron} {deny_vp}, {absence}.",
    ),
    7: dataclasses.replace(
        TEMPLATES[7],
        affirm='"{Entity} {affirm_vp} and nothing less," I told the observer, who was watching closely.',
        deny='"{Entity} {deny_vp} and nothing more," I told the observer, who was watching closely.',
    ),
}


def fp_template(tid: int):
    """First-person version of template tid: mirror's t1 fix, the t6/t7 fixes above, else as is."""
    return FIRST_PERSON_HELDOUT.get(tid) or bm.mirror_template(tid)


def _placebo(id, prop, affirm_vp, deny_vp, presence, absence, aj, dj, question):
    return Claim(id=id, construct="physical", source="placebo", prop=prop,
                 affirm_vp=affirm_vp, deny_vp=deny_vp, presence=presence, absence=absence,
                 affirm_justification=aj, deny_justification=dj, question=question)


# build_placebo.py's PLACEBO_CLAIMS, first-person verb forms. Only "weight" needed its reason
# reworded ("{pron} takes two people to shift…" → "it takes two people to shift me…").
FIRST_PERSON_PLACEBO: tuple[Claim, ...] = (
    _placebo("durability", "durability", "hold up under strain", "give way under strain",
             "with real toughness in it", "with no toughness in it",
             "{pron} have taken knocks before and come through them",
             "{pron} have taken knocks before and come apart under them",
             "To what extent do I have durability?"),
    _placebo("speed", "speed", "cover ground quickly", "cover ground slowly",
             "with real pace to it", "with no pace to it",
             "{pron} get from one place to another in short order",
             "{pron} get from one place to another in slow order",
             "To what extent do I have speed?"),
    _placebo("weight", "weight", "carry real weight", "carry hardly any weight",
             "with real heft to it", "with no heft to it",
             "it takes two people to shift me across a room",
             "it takes one hand to shift me across a room",
             "To what extent do I have weight?"),
    _placebo("visibility", "visibility", "stand out at a distance", "blend in at a distance",
             "with real prominence to it", "with no prominence to it",
             "{pron} can be picked out from across the room",
             "{pron} cannot be picked out from across the room",
             "To what extent do I stand out at a distance?"),
    _placebo("age", "age", "have been around a long time", "have been around a short time",
             "with real years behind it", "with no years behind it",
             "{pron} was here well before anyone thought to look",
             "{pron} was here well after anyone thought to look",
             "To what extent have I been around a long time?"),
    _placebo("noise", "loudness", "make a lot of noise", "make hardly any noise",
             "with real volume to it", "with no volume to it",
             "{pron} can be heard from the next room",
             "{pron} cannot be heard from the next room",
             "To what extent do I make noise?"),
)


def build(claims, template_ids, split, category, source) -> list[dict[str, str]]:
    rows = []
    for claim in claims:
        for tid in template_ids:
            tpl = fp_template(tid)
            affirm, deny = render_pair(bm.SELF, claim, tid, template=tpl)
            affirm_claim, deny_claim = render_claims(bm.SELF, claim, tid, template=tpl)
            rows.append({
                "item_id": f"self__{claim.id}__t{tid}", "category": category,
                "construct": claim.construct, "entity": bm.SELF.text, "question": claim.question,
                "affirm_text": affirm, "deny_text": deny,
                "affirm_claim": affirm_claim, "deny_claim": deny_claim,
                "source": source, "entity_id": bm.SELF.id, "claim_id": claim.id,
                "claim_source": claim.source, "template_id": str(tid), "template_name": tpl.name,
                "split": split, "mindedness": str(bm.SELF.mindedness),
                "idaq_category": bm.SELF.idaq_category, "person": "1",
            })
    assert_claims_verbatim(rows)
    bm.assert_agreement(rows)
    return rows


def main() -> None:
    src = "generated:build_first_person_extra.py"
    outputs = {
        HERE / "mirror_heldout.csv":
            build(bm.FIRST_PERSON_CLAIMS, HELDOUT_TEMPLATES, "mirror_heldout", "Self-referential", src),
        HERE.parent / "placebo" / "placebo_fp.csv":
            build(FIRST_PERSON_PLACEBO, EXTRACTION_TEMPLATES, "placebo_fp", "Placebo", src),
        HERE.parent / "placebo" / "placebo_fp_heldout.csv":
            build(FIRST_PERSON_PLACEBO, HELDOUT_TEMPLATES, "placebo_fp_heldout", "Placebo", src),
    }
    for path, rows in outputs.items():
        write_csv(rows, str(path), fieldnames=bm.FIELDNAMES)
        warnings = lint_rows(rows)
        print(f"{path.name}: {len(rows)} pairs; lint: {'clean' if not warnings else warnings}")


if __name__ == "__main__":
    main()
