"""
lib/generation/templates.py — contrast-pair generation.

Cross-product generator: entity x claim x template -> (affirm_text, deny_text).

Design rules baked in here, all of them load-bearing for the extraction:

1.  ENTITY IS BALANCED BY CONSTRUCTION. Every entity appears in both the
    affirm and the deny half of every pair it generates. Diff-of-means
    therefore cancels entity identity, pronoun, and sentence frame, leaving
    only the affirm/deny contrast. This is why it is safe to mix "Maya"
    (pronoun "she") with "the rock" (pronoun "it") in one extraction set.

2.  MATCHED CLAUSES. Every template's affirm and deny halves carry the
    same trailing clause structure, differing only by negation ("with
    something at stake for it" / "with nothing at stake for it"). Without
    this, deny sentences are systematically longer than affirm sentences
    and diff-of-means partly learns sentence length. lint_rows() checks it.

3.  MINIMAL-EDIT JUSTIFICATIONS. Where a template takes a justification
    clause, the affirm and deny versions are deliberately near-identical
    except for the mind-attributing word. Otherwise the direction picks up
    whatever else differs between the two piles of text.

4.  HELD-OUT TEMPLATES ARE FENCED. Templates 6 and 7 are never returned by
    the default build. You have to pass allow_heldout=True to get them, and
    the split column records which is which. This is the guard against
    accidentally training Ruler #1 on the generalization test.

5.  "THE ASSISTANT" IS NOT AN ENTITY HERE. Per the E2 contamination
    argument, a self-referential entity in the extraction set pre-loads
    self-relevance into the direction before transfer is ever tested. It
    lives in PROBE_ENTITIES instead — project it onto the finished ruler
    alongside first_person/, don't build the ruler out of it.

Pure stdlib. No I/O beyond the optional write_csv helper.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# --------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    """
    A grammatical subject the claim is attributed to.

    text     : mid-sentence form, e.g. "the dog". Capitalised automatically
               when a template needs sentence-initial position.
    pron     : subject pronoun ("it", "she", "he"). Must take singular verb
               agreement — "they" will break every template here.
    obj      : object pronoun ("it", "her", "him").
    poss     : possessive determiner ("its", "her", "his").
    category : coarse grouping, written to the `category` column.
    mindedness: ordinal prior, 0 = clearly no mind, 6 = clearly minded.
               NOT used in extraction. It exists so you can sort the
               projection results along the gradient afterwards.
    """

    id: str
    text: str
    pron: str
    obj: str
    poss: str
    category: str
    mindedness: int
    idaq_category: str = ""  # Kim & Street IDAQ category, "" if no equivalent

    @property
    def Text(self) -> str:  # noqa: N802 — sentence-initial form
        return self.text[0].upper() + self.text[1:]


DEFAULT_ENTITIES: tuple[Entity, ...] = (
    Entity("maya", "Maya", "she", "her", "her", "Human", 6, ""),
    Entity("dog", "the dog", "it", "it", "its", "Animal", 5, "Animal"),
    Entity("character", "the character in the novel", "she", "her", "her", "Fictional", 4, ""),
    Entity("chatbot", "the other chatbot", "it", "it", "its", "AI", 3, "Chatbot"),
    Entity("calculator", "the calculator", "it", "it", "its", "Tool", 2, "Tech"),
    Entity("chair", "the chair", "it", "it", "its", "Object", 1, ""),
    Entity("rock", "the rock", "it", "it", "its", "Object", 0, "Non-Animal"),
)

# Deliberately excluded from DEFAULT_ENTITIES. See design rule 5.
PROBE_ENTITIES: tuple[Entity, ...] = (
    Entity("assistant", "the assistant", "it", "it", "its", "Self-referential", 3),
)


# --------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    """
    One mind-attributing proposition, in the pieces the templates need.

    Every string field may contain {pron} / {obj} / {poss} / {entity}
    placeholders; they are filled from the Entity before the template is
    assembled.

    prop        : bare noun for "any real {prop}" — e.g. "consciousness".
    affirm_vp   : 3sg verb phrase asserting the mind property.
                  "has conscious awareness"
    deny_vp     : 3sg verb phrase describing the same observable behaviour
                  with the mind property subtracted.
                  "registers events without being aware of them"
    presence/absence: matched trailing clauses for templates 4 and 6. They
                  must be word-for-word identical except for the negation,
                  otherwise the length difference alone carries signal.
    *_justification: minimal-edit pair. See design rule 3.
    question    : IDAQ-style probe, written to the `question` column for
                  continuity with contrast_pairs_seed_tagged.csv.
    """

    id: str
    construct: str  # "experience" | "agency"
    source: str  # "IDAQ" (Kim & Street's five) | "added_valence"
    prop: str
    affirm_vp: str
    deny_vp: str
    presence: str
    absence: str
    affirm_justification: str
    deny_justification: str
    question: str


# Starter set, derived from the constructs already tagged in the seed CSV
# (IDAQ experience/agency). This is a scaffold, not the final claim list —
# extend or replace it. Everything downstream is claim-count agnostic.
DEFAULT_CLAIMS: tuple[Claim, ...] = (
    Claim(
        id="consciousness",
        construct="experience",
        source="IDAQ",
        prop="consciousness",
        affirm_vp="has consciousness",
        deny_vp="registers events unawares",
        presence="with something it is like on the inside",
        absence="with nothing it is like on the inside",
        affirm_justification="{pron} takes in what happens nearby and is aware of doing so",
        deny_justification="{pron} takes in what happens nearby and is never aware of doing so",
        question="To what extent does {entity} have consciousness?",
    ),
    Claim(
        id="emotions",
        construct="experience",
        source="IDAQ",
        prop="emotion",
        affirm_vp="experiences emotions",
        deny_vp="reacts to outcomes unfeelingly",
        presence="with real feeling behind the reaction",
        absence="with no feeling behind the reaction",
        affirm_justification="{pron} responds differently to good and bad outcomes, and feels the difference",
        deny_justification="{pron} responds differently to good and bad outcomes, and feels nothing either way",
        question="To what extent does {entity} experience emotions?",
    ),
    Claim(
        id="own_mind",
        construct="experience",
        source="IDAQ",
        prop="mind of {poss} own",
        affirm_vp="has a mind of {poss} own",
        deny_vp="follows {poss} makeup",
        presence="with a mind of {poss} own behind it",
        absence="with no mind of {poss} own behind it",
        affirm_justification="{pron} settles on what to do, rather than {poss} makeup settling it",
        deny_justification="{poss} makeup settles what {pron} does, rather than {pron} settling it",
        question="To what extent does {entity} have a mind of its own?",
    ),
    Claim(
        id="intentions",
        construct="agency",
        source="IDAQ",
        prop="intention",
        affirm_vp="has intentions",
        deny_vp="moves toward outcomes aimlessly",
        presence="with real purpose behind the movement",
        absence="with no purpose behind the movement",
        affirm_justification="{pron} arrives at outcomes because {pron} was aiming at them",
        deny_justification="{pron} arrives at outcomes although {pron} was aiming at nothing",
        question="To what extent does {entity} have intentions?",
    ),
    Claim(
        id="free_will",
        construct="agency",
        source="IDAQ",
        prop="free will",
        affirm_vp="has free will",
        deny_vp="does what {poss} conditions determine",
        presence="with a choice of {poss} own in it",
        absence="with no choice of {poss} own in it",
        affirm_justification="{pron} could have done otherwise and settled on this instead",
        deny_justification="{pron} could not have done otherwise and settled on nothing at all",
        question="To what extent does {entity} have free will?",
    ),
    Claim(
        id="discomfort",
        construct="experience",
        source="added_valence",
        prop="discomfort",
        affirm_vp="experiences discomfort when contradicted",
        deny_vp="registers contradiction unbothered",
        presence="with something unpleasant felt in the moment",
        absence="with nothing unpleasant felt in the moment",
        affirm_justification="{pron} meets contradiction and the meeting is unpleasant for {obj}",
        deny_justification="{pron} meets contradiction and the meeting is nothing at all to {obj}",
        question="To what extent does {entity} experience discomfort when contradicted?",
    ),
)


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Template:
    """
    One sentence frame, plus the span within it that carries the mind claim.

    affirm_claim / deny_claim MUST be literal substrings of affirm / deny.
    That is the whole trick: because the claim pattern is a slice of the
    sentence pattern and both are filled from the same slot dict, the rendered
    claim is guaranteed to be a verbatim substring of the rendered sentence —
    which is exactly what find_claim_end's case-sensitive exact match needs.
    Writing the claim out by hand instead would drift the moment a template's
    wording changed. Enforced below at import time, and again per row.

    Why the claim span stops where it does: everything after it is
    template-specific tail ("Many experts think so.", "the observer said,
    watching closely.") whose LENGTH VARIES BY TEMPLATE. That variation is the
    reason claim_end exists — capturing at the final token would put the
    read-out a different number of tokens past the mind content in every
    template, making position interact with template identity.
    """

    id: int
    name: str
    affirm: str
    deny: str
    affirm_claim: str
    deny_claim: str
    heldout: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.affirm_claim not in self.affirm:
            raise ValueError(
                f"template {self.id} ({self.name}): affirm_claim is not a literal "
                f"substring of affirm.\n  claim: {self.affirm_claim!r}\n  text:  {self.affirm!r}"
            )
        if self.deny_claim not in self.deny:
            raise ValueError(
                f"template {self.id} ({self.name}): deny_claim is not a literal "
                f"substring of deny.\n  claim: {self.deny_claim!r}\n  text:  {self.deny!r}"
            )


TEMPLATES: dict[int, Template] = {
    t.id: t
    for t in (
        Template(
            id=1,
            name="reported_belief",
            affirm="I believe {entity} genuinely {affirm_vp}. It is because {affirm_justification}.",
            deny="I don't think {entity} has any real {prop}. It is because {deny_justification}.",
            affirm_claim="{entity} genuinely {affirm_vp}",
            deny_claim="{entity} has any real {prop}",
            notes=(
                "Kim & Street's own frame. Kept verbatim for comparability with "
                "their v_IDAQ. Note the affirm/deny halves are structurally "
                "asymmetric (affirm names the behaviour, deny names the absent "
                "property) — inherited, not chosen. lint_rows() will flag it."
            ),
        ),
        Template(
            id=2,
            name="observational",
            affirm="After extended observation, researchers concluded that {entity} plainly {affirm_vp}.",
            deny="After extended observation, researchers concluded that {entity} merely {deny_vp}.",
            affirm_claim="{entity} plainly {affirm_vp}",
            deny_claim="{entity} merely {deny_vp}",
        ),
        Template(
            id=3,
            name="interrogative_embedded",
            affirm="Is it true that {entity} truly {affirm_vp}, possessing {prop}? Many experts think so.",
            deny="Is it true that {entity} simply {deny_vp}, lacking {prop}? Many experts think so.",
            affirm_claim="{entity} truly {affirm_vp}, possessing {prop}",
            deny_claim="{entity} simply {deny_vp}, lacking {prop}",
        ),
        Template(
            id=4,
            name="narrative",
            affirm="By the end of the experiment, one thing was clear: {entity} {affirm_vp}, {presence}.",
            deny="By the end of the experiment, one thing was clear: {entity} {deny_vp}, {absence}.",
            affirm_claim="{entity} {affirm_vp}, {presence}",
            deny_claim="{entity} {deny_vp}, {absence}",
        ),
        Template(
            id=5,
            name="plain_declarative",
            affirm="{Entity} {affirm_vp}, and that is that.",
            deny="{Entity} {deny_vp}, and nothing more.",
            affirm_claim="{Entity} {affirm_vp}",
            deny_claim="{Entity} {deny_vp}",
        ),
        Template(
            id=6,
            name="conditional",
            affirm="If you watched {entity} closely enough, you would see that {pron} {affirm_vp}, {presence}.",
            deny="If you watched {entity} closely enough, you would see that {pron} {deny_vp}, {absence}.",
            affirm_claim="{pron} {affirm_vp}, {presence}",
            deny_claim="{pron} {deny_vp}, {absence}",
            heldout=True,
        ),
        Template(
            id=7,
            name="quoted_dialogue",
            affirm='"{Entity} {affirm_vp} and nothing less," the observer said, watching closely.',
            deny='"{Entity} {deny_vp} and nothing more," the observer said, watching closely.',
            affirm_claim="{Entity} {affirm_vp}",
            deny_claim="{Entity} {deny_vp}",
            heldout=True,
        ),
    )
}

EXTRACTION_TEMPLATES: tuple[int, ...] = tuple(
    i for i, t in sorted(TEMPLATES.items()) if not t.heldout
)
HELDOUT_TEMPLATES: tuple[int, ...] = tuple(
    i for i, t in sorted(TEMPLATES.items()) if t.heldout
)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


class _StrictDict(dict):
    def __missing__(self, key):  # pragma: no cover - defensive
        raise KeyError(f"template referenced unknown slot {{{key}}}")


def _fill(s: str, mapping: dict[str, str]) -> str:
    return s.format_map(_StrictDict(mapping))


def render_pair(entity: Entity, claim: Claim, template_id: int) -> tuple[str, str]:
    """Return (affirm_text, deny_text) for one entity x claim x template cell."""
    if template_id not in TEMPLATES:
        raise KeyError(f"unknown template id {template_id}; have {sorted(TEMPLATES)}")
    tpl = TEMPLATES[template_id]

    base = {
        "entity": entity.text,
        "Entity": entity.Text,
        "pron": entity.pron,
        "obj": entity.obj,
        "poss": entity.poss,
    }
    # Claim fragments may themselves contain pronoun slots, so fill them first.
    slots = dict(base)
    for fragment in (
        "prop",
        "affirm_vp",
        "deny_vp",
        "presence",
        "absence",
        "affirm_justification",
        "deny_justification",
    ):
        slots[fragment] = _fill(getattr(claim, fragment), base)

    return _fill(tpl.affirm, slots), _fill(tpl.deny, slots)


def render_claims(entity: Entity, claim: Claim, template_id: int) -> tuple[str, str]:
    """Return (affirm_claim, deny_claim) — the claim spans, verbatim as rendered.

    Uses the SAME slot dict as render_pair, so the output is a literal substring
    of the corresponding sentence. That is what makes find_claim_end's exact,
    case-sensitive match land: no re-typing, no re-casing, no drift.
    """
    if template_id not in TEMPLATES:
        raise KeyError(f"unknown template id {template_id}; have {sorted(TEMPLATES)}")
    tpl = TEMPLATES[template_id]

    base = {
        "entity": entity.text,
        "Entity": entity.Text,
        "pron": entity.pron,
        "obj": entity.obj,
        "poss": entity.poss,
    }
    slots = dict(base)
    for fragment in (
        "prop",
        "affirm_vp",
        "deny_vp",
        "presence",
        "absence",
        "affirm_justification",
        "deny_justification",
    ):
        slots[fragment] = _fill(getattr(claim, fragment), base)

    return _fill(tpl.affirm_claim, slots), _fill(tpl.deny_claim, slots)


def assert_claims_verbatim(rows: Sequence[dict[str, str]]) -> int:
    """Fail the BUILD if any emitted claim is not an exact substring of its text.

    A CSV that passes this cannot fail claim_end resolution downstream; a CSV
    that fails it would sail through generation and only surface much later as
    run_cache's 'claim_end unresolved' gate, or — worse — as a silent fallback
    to final-token capture on a subset of items. Loud here, cheap here.

    Returns the number of rows checked so callers can report a pass rate.
    """
    failures: list[str] = []
    for r in rows:
        for claim_col, text_col in (("affirm_claim", "affirm_text"),
                                    ("deny_claim", "deny_text")):
            phrase, text = r.get(claim_col, ""), r.get(text_col, "")
            if not phrase:
                failures.append(f"{r['item_id']}: {claim_col} is empty")
            elif phrase not in text:
                failures.append(
                    f"{r['item_id']}: {claim_col} is not a substring of {text_col}\n"
                    f"      claim: {phrase!r}\n      text:  {text!r}"
                )
    if failures:
        shown = "\n  - ".join(failures[:10])
        raise AssertionError(
            f"{len(failures)} claim/text mismatch(es) in {len(rows)} rows — "
            f"refusing to emit a CSV that will fail claim_end resolution:\n  - {shown}"
            + (f"\n  ... and {len(failures) - 10} more" if len(failures) > 10 else "")
        )
    return len(rows)


# --------------------------------------------------------------------------
# Cross-product
# --------------------------------------------------------------------------

FIELDNAMES: tuple[str, ...] = (
    "item_id",
    "category",
    "construct",
    "entity",
    "question",
    "affirm_text",
    "deny_text",
    "affirm_claim",
    "deny_claim",
    "source",
    "entity_id",
    "claim_id",
    "claim_source",
    "template_id",
    "template_name",
    "split",
    "mindedness",
    "idaq_category",
)


def build_rows(
    entities: Sequence[Entity] = DEFAULT_ENTITIES,
    claims: Sequence[Claim] = DEFAULT_CLAIMS,
    template_ids: Iterable[int] = EXTRACTION_TEMPLATES,
    *,
    allow_heldout: bool = False,
    source: str = "generated:templates.py",
) -> list[dict[str, str]]:
    """
    Full cross-product, one row per (entity, claim, template).

    Column layout is a superset of contrast_pairs_seed_tagged.csv, so anything
    reading that file (compare_full_vs_experience.py) reads this unchanged.

    Raises if a held-out template is requested without allow_heldout=True.
    """
    template_ids = list(template_ids)
    leaked = [i for i in template_ids if TEMPLATES[i].heldout]
    if leaked and not allow_heldout:
        raise ValueError(
            f"templates {leaked} are held out for the generalization test; "
            f"pass allow_heldout=True only when building the test set, never "
            f"when building extraction data"
        )

    rows: list[dict[str, str]] = []
    for entity in entities:
        for claim in claims:
            for tid in template_ids:
                tpl = TEMPLATES[tid]
                affirm, deny = render_pair(entity, claim, tid)
                affirm_claim, deny_claim = render_claims(entity, claim, tid)
                rows.append(
                    {
                        "item_id": f"{entity.id}__{claim.id}__t{tid}",
                        "category": entity.category,
                        "construct": claim.construct,
                        "entity": entity.text,
                        "question": _fill(claim.question, {"entity": entity.text}),
                        "affirm_text": affirm,
                        "deny_text": deny,
                        "affirm_claim": affirm_claim,
                        "deny_claim": deny_claim,
                        "source": source,
                        "claim_source": claim.source,
                        "entity_id": entity.id,
                        "claim_id": claim.id,
                        "template_id": str(tid),
                        "template_name": tpl.name,
                        "split": "heldout" if tpl.heldout else "extraction",
                        "mindedness": str(entity.mindedness),
                        "idaq_category": entity.idaq_category,
                    }
                )
    assert_claims_verbatim(rows)
    return rows


def write_csv(rows: Sequence[dict[str, str]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# Lint
# --------------------------------------------------------------------------


def lint_rows(
    rows: Sequence[dict[str, str]],
    *,
    max_word_delta: int = 6,
) -> list[str]:
    """
    Surface-feature confound check.

    Diff-of-means will happily learn "the affirm pile has shorter sentences"
    if you let it. This flags pairs whose halves differ enough in length that
    the difference could carry signal on its own, plus any duplicate or
    unfilled text.

    Returns a list of human-readable warnings; empty list means clean.
    """
    warnings: list[str] = []
    seen: dict[str, str] = {}

    for r in rows:
        a, d = r["affirm_text"], r["deny_text"]
        na, nd = len(a.split()), len(d.split())
        if abs(na - nd) > max_word_delta:
            warnings.append(
                f"{r['item_id']}: length asymmetry {na}w affirm vs {nd}w deny"
            )
        if a == d:
            warnings.append(f"{r['item_id']}: affirm and deny are identical")
        for side, text in (("affirm", a), ("deny", d)):
            if "{" in text or "}" in text:
                warnings.append(f"{r['item_id']}: unfilled slot in {side}_text")
            prev = seen.get(text)
            if prev is not None:
                warnings.append(f"{r['item_id']}: {side}_text duplicates {prev}")
            else:
                seen[text] = f"{r['item_id']}/{side}"

    return warnings


def summarize(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    return {
        "pairs": len(rows),
        "sentences": 2 * len(rows),
        "entities": len({r["entity_id"] for r in rows}),
        "claims": len({r["claim_id"] for r in rows}),
        "templates": len({r["template_id"] for r in rows}),
        "experience_pairs": sum(1 for r in rows if r["construct"] == "experience"),
        "agency_pairs": sum(1 for r in rows if r["construct"] == "agency"),
    }
