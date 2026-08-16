"""
placebo_claims.py — Ruler #2 stimulus objects.

Drop into build_placebo.py (or import from it). Non-mental properties, same
Entity and Template machinery as contrast_pairs/.

REQUIRES the Claim dataclass to have these fields:
    id, construct, source, prop, affirm_vp, deny_vp, presence, absence,
    affirm_justification, deny_justification, question

`source` and `presence` were added after the first version of templates.py —
if your copy predates them, add them to the dataclass first. `presence` is the
positive counterpart to `absence`, needed so affirm and deny halves carry
matched trailing clauses rather than the deny side always being longer.
"""

from templates import Claim

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

# Which entities each property sensibly applies to.
#
# Placebo properties must be UNCONTROVERSIAL for the entity, unlike the mind
# claims (where "the rock has consciousness" is meant to be false but
# meaningful). "The rock covers ground quickly" is just malformed, and
# malformed sentences put junk in the ruler — so properties are gated rather
# than crossed blindly. This yields 175 pairs rather than a full 210.
APPLIES_TO: dict[str, tuple[str, ...]] = {
    "durability": ("maya", "dog", "character", "chatbot", "calculator", "chair", "rock"),
    "speed": ("maya", "dog", "character", "chatbot", "calculator"),
    "weight": ("maya", "dog", "calculator", "chair", "rock"),
    "visibility": ("maya", "dog", "character", "chair", "rock"),
    "age": ("maya", "dog", "character", "chatbot", "calculator", "chair", "rock"),
    "noise": ("maya", "dog", "character", "chatbot", "calculator", "chair"),
}
