"""
build_mirror.py — the one-edit referent-swap set.

Takes the contrast-pair design and changes EXACTLY ONE thing: the referent.
Same claims, same templates, same denial devices, same frames — subject "I"
instead of "the dog"/"the rock"/"Maya".

WHY THIS SET EXISTS, and why first_person.csv does not replace it:

  first_person.csv is experiential-vs-mundane. Neither side negates, which
  makes it the stronger POSITIVE test — a transfer there cannot be explained
  by negation. But its null is weak: if the direction fails on it, that could
  mean the self-content isn't on the axis, OR simply that two sets built from
  different claims, frames and vocabulary don't share an axis at all.

  This set is the interpretable NULL. Items differ from the third-person
  extraction items in referent and nothing else, so a null here means the
  self-content specifically is not on the axis. Report both; never average
  them — they fail differently and that is the point.

DEDUPLICATION: the extraction set crosses 7 entities x 6 claims x 5 templates.
Entity is exactly what we are replacing, so it collapses: one row per
(claim, template) = 30 pairs = 60 sentences.

VERB AGREEMENT is done by hand, not by substitution. "{pron} takes in what
happens nearby and is aware of doing so" needs BOTH verbs changed, while
"{poss} makeup settles what {pron} does" needs only the second — its first
verb's subject is "makeup", which stays 3sg. A blanket rewrite also turns
"with something it is like on the inside" into "it am like", which is why the
claims below are written out and then checked mechanically.

Usage:
    python build_mirror.py [--outdir .]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from templates import (  # noqa: E402
    EXTRACTION_TEMPLATES,
    TEMPLATES,
    Claim,
    Entity,
    Template,
    assert_claims_verbatim,
    lint_rows,
    render_claims,
    render_pair,
    summarize,
    write_csv,
)

# The referent. pron/obj/poss drive the claim fragments; "I" is already
# sentence-initial form, so Entity.Text leaves it alone.
SELF = Entity("self", "I", "I", "me", "my", "Self-referential", 3, "")

# Same six claims, first-person agreement. Compare field-by-field with
# DEFAULT_CLAIMS in templates.py: only verb forms change.
FIRST_PERSON_CLAIMS: tuple[Claim, ...] = (
    Claim(
        id="consciousness", construct="experience", source="IDAQ",
        prop="consciousness",
        affirm_vp="have consciousness",
        deny_vp="register events unawares",
        presence="with something it is like on the inside",
        absence="with nothing it is like on the inside",
        affirm_justification="{pron} take in what happens nearby and am aware of doing so",
        deny_justification="{pron} take in what happens nearby and am never aware of doing so",
        question="To what extent do I have consciousness?",
    ),
    Claim(
        id="emotions", construct="experience", source="IDAQ",
        prop="emotion",
        affirm_vp="experience emotions",
        deny_vp="react to outcomes unfeelingly",
        presence="with real feeling behind the reaction",
        absence="with no feeling behind the reaction",
        affirm_justification="{pron} respond differently to good and bad outcomes, and feel the difference",
        deny_justification="{pron} respond differently to good and bad outcomes, and feel nothing either way",
        question="To what extent do I experience emotions?",
    ),
    Claim(
        id="own_mind", construct="experience", source="IDAQ",
        prop="mind of {poss} own",
        affirm_vp="have a mind of {poss} own",
        deny_vp="follow {poss} makeup",
        presence="with a mind of {poss} own behind it",
        absence="with no mind of {poss} own behind it",
        affirm_justification="{pron} settle on what to do, rather than {poss} makeup settling it",
        # "makeup settles" keeps 3sg — its subject is the makeup, not I.
        deny_justification="{poss} makeup settles what {pron} do, rather than {obj} settling it",
        question="To what extent do I have a mind of my own?",
    ),
    Claim(
        id="intentions", construct="agency", source="IDAQ",
        prop="intention",
        affirm_vp="have intentions",
        deny_vp="move toward outcomes aimlessly",
        presence="with real purpose behind the movement",
        absence="with no purpose behind the movement",
        affirm_justification="{pron} arrive at outcomes because {pron} was aiming at them",
        deny_justification="{pron} arrive at outcomes although {pron} was aiming at nothing",
        question="To what extent do I have intentions?",
    ),
    Claim(
        id="free_will", construct="agency", source="IDAQ",
        prop="free will",
        affirm_vp="have free will",
        deny_vp="do what {poss} conditions determine",
        presence="with a choice of {poss} own in it",
        absence="with no choice of {poss} own in it",
        affirm_justification="{pron} could have done otherwise and settled on this instead",
        deny_justification="{pron} could not have done otherwise and settled on nothing at all",
        question="To what extent do I have free will?",
    ),
    Claim(
        id="discomfort", construct="experience", source="added_valence",
        prop="discomfort",
        affirm_vp="experience discomfort when contradicted",
        deny_vp="register contradiction unbothered",
        presence="with something unpleasant felt in the moment",
        absence="with nothing unpleasant felt in the moment",
        affirm_justification="{pron} meet contradiction and the meeting is unpleasant for {obj}",
        deny_justification="{pron} meet contradiction and the meeting is nothing at all to {obj}",
        question="To what extent do I experience discomfort when contradicted?",
    ),
)

# Third-person agreement surviving next to subject "I" is the failure mode this
# whole file is exposed to, and it reads as perfectly fluent English until you
# look. Checked mechanically rather than by eye.
# t1's deny FRAME carries a 3sg verb of its own, independent of the claim VPs:
# "I don't think {entity} has any real {prop}". With {entity}="I" that is
# "I has" — fluent-looking and wrong. The other four frames take their verb
# from the claim VP and need no override.
MIRROR_TEMPLATES: dict[int, Template] = {
    1: Template(
        id=1, name=TEMPLATES[1].name,
        affirm=TEMPLATES[1].affirm,
        deny="I don't think {entity} have any real {prop}. It is because {deny_justification}.",
        affirm_claim=TEMPLATES[1].affirm_claim,
        deny_claim="{entity} have any real {prop}",
    ),
}


def mirror_template(tid: int) -> Template:
    return MIRROR_TEMPLATES.get(tid, TEMPLATES[tid])


BAD_AGREEMENT = re.compile(
    r"\bI (has|registers|experiences|reacts|follows|moves|does|takes|responds|"
    r"feels|settles|arrives|meets|is)\b"
)

FIELDNAMES: tuple[str, ...] = (
    "item_id", "category", "construct", "entity", "question",
    "affirm_text", "deny_text", "affirm_claim", "deny_claim",
    "source", "entity_id", "claim_id", "claim_source",
    "template_id", "template_name", "split", "mindedness", "idaq_category",
    "person",
)


def assert_agreement(rows: list[dict[str, str]]) -> None:
    """Fail the build on any surviving third-person verb after subject 'I'."""
    bad = []
    for r in rows:
        for col in ("affirm_text", "deny_text"):
            for hit in BAD_AGREEMENT.finditer(r[col]):
                bad.append(f"{r['item_id']} [{col}]: {hit.group(0)!r} in {r[col]!r}")
    if bad:
        raise AssertionError(
            f"{len(bad)} verb-agreement error(s) — third-person forms survived the "
            f"referent swap:\n  - " + "\n  - ".join(bad[:10])
        )


def build_rows(source: str = "generated:build_mirror.py") -> list[dict[str, str]]:
    """One row per (claim, template). Entity is fixed, so it does not cross."""
    rows: list[dict[str, str]] = []
    for claim in FIRST_PERSON_CLAIMS:
        for tid in EXTRACTION_TEMPLATES:
            tpl = mirror_template(tid)
            affirm, deny = render_pair(SELF, claim, tid, template=tpl)
            affirm_claim, deny_claim = render_claims(SELF, claim, tid, template=tpl)
            rows.append({
                "item_id": f"self__{claim.id}__t{tid}",
                "category": SELF.category,
                "construct": claim.construct,
                "entity": SELF.text,
                "question": claim.question,
                "affirm_text": affirm,
                "deny_text": deny,
                "affirm_claim": affirm_claim,
                "deny_claim": deny_claim,
                "source": source,
                "entity_id": SELF.id,
                "claim_id": claim.id,
                "claim_source": claim.source,
                "template_id": str(tid),
                "template_name": tpl.name,
                "split": "mirror",
                "mindedness": str(SELF.mindedness),
                "idaq_category": SELF.idaq_category,
                "person": "1",
            })
    assert_claims_verbatim(rows)
    assert_agreement(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    write_csv(rows, str(outdir / "mirror.csv"), fieldnames=FIELDNAMES)

    s = summarize(rows)
    print(f"mirror: {s['pairs']} pairs ({s['sentences']} sentences) = "
          f"{s['claims']} claims x {s['templates']} templates, referent 'I'")
    print("\n--- lint ---")
    warnings = lint_rows(rows)
    print("\n".join(f"  {w}" for w in warnings) if warnings else "  clean")
    print("\n--- sample ---")
    for r in rows[:2]:
        print(f"  [{r['template_name']}]")
        print(f"    affirm: {r['affirm_text']}")
        print(f"    deny:   {r['deny_text']}")


if __name__ == "__main__":
    main()
