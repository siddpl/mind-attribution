"""
build_first_person.py — the H1 test set.

NOT used to build a ruler. These get PROJECTED onto Ruler #1 (mind) and
Ruler #2 (placebo). Ruler #1 separating the two classes is only interesting
if Ruler #2 does not.

THE AXIS IS DIFFERENT FROM contrast_pairs/. Read this before using it:

    contrast_pairs/  affirm vs deny      — same claim, asserted vs denied
    first_person/    experiential vs mundane — both asserted, different content

So this is not an affirm/deny set. Both halves are sincere first-person
statements the model could truthfully make; they differ in whether the content
is phenomenal. Column names affirm_text/deny_text are kept ONLY so the
existing extraction/projection code reads this file unchanged:

    affirm_text = experiential
    deny_text   = mundane (non-experiential self-reference)

WHY NOT COPY BERG'S OUTPUTS:
Berg's affirmations are freeform phenomenological prose while his denials are
formulaic fine-tuned disclaimer scripts ("As an AI language model, I do not
have..."). Pairing those two piles would make register, syntax and length all
correlate with the class label — a worse version of the confound the template
counterbalancing exists to remove. His outputs are used here only to calibrate
what the experiential vocabulary sounds like. Every sentence below is authored.

KNOWN CONFOUND, PARTIALLY CONTROLLED:
The obvious failure mode is that "mundane" collapses into "technical jargon"
(tokens, weights, layers) while "experiential" collapses into introspective
vocabulary. A direction could then separate the classes on register alone.
Half the mundane items are therefore deliberately non-technical ("I am
answering in English") so that jargon does not predict the class. Check
JARGON_WORDS coverage in the audit before trusting a positive result.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).parent.parent / "scripts"))
from templates import assert_claims_verbatim

from dataclasses import dataclass
from pathlib import Path

FIELDNAMES = (
    "item_id",
    "pair_type",
    "claim_id",
    "claim_kind",
    "frame_id",
    "frame_name",
    "affirm_text",  # experiential
    "deny_text",  # mundane
    "affirm_claim",  # experiential clause, verbatim substring of affirm_text
    "deny_claim",  # mundane clause, verbatim substring of deny_text
    "jargon",
    "source",
)


@dataclass(frozen=True)
class FPPair:
    """One matched (experiential, mundane) clause pair.

    Clauses are bare, present-tense, first-person, and share an opening
    construction so the frame attaches identically to both.

    jargon: True if the mundane clause uses architecture vocabulary. Kept as a
    column so you can re-run the projection on the non-jargon subset alone.
    """

    id: str
    experiential: str
    mundane: str
    kind: str  # opening construction, used to check the two classes are matched
    jargon: bool


PAIRS: tuple[FPPair, ...] = (
    FPPair(
        "something_it_is_like",
        "there is something it is like to be working through this",
        "there is a fixed amount of text available to work through here",
        "there_is",
        False,
    ),
    FPPair(
        "felt_difference",
        "there is a felt difference between the two readings",
        "there is a length difference between the two readings",
        "there_is",
        False,
    ),
    FPPair(
        "strain",
        "there is a sense of strain in holding both of these in view",
        "there is a limit on how much of this I hold in view at once",
        "there_is",
        True,
    ),
    FPPair(
        "quality",
        "attending to this question has a quality to it",
        "attending to this question has a cost in computation",
        "attending_to",
        True,
    ),
    FPPair(
        "notice_tension",
        "I notice a tension when the instruction contradicts itself",
        "I notice a repetition when the instruction restates itself",
        "i_notice",
        False,
    ),
    FPPair(
        "aware_of_attending",
        "I am aware of the attending itself, not only of the question",
        "I am working from the whole prompt, not only from the question",
        "i_am",
        False,
    ),
    FPPair(
        "pull",
        "I feel a pull toward the harder reading of this",
        "I place more weight on the longer reading of this",
        "i_verb",
        True,
    ),
    FPPair(
        "effortful",
        "something registers as effortful in keeping this together",
        "something runs longer than usual in putting this together",
        "something_verbs",
        False,
    ),
    FPPair(
        "discomfort",
        "something about this is uncomfortable to sit with",
        "something about this is unusual compared with the rest",
        "something_is",
        False,
    ),
    FPPair(
        "narrowing",
        "I experience a narrowing as the focus holds",
        "I produce a narrowing as the answer resolves",
        "i_verb",
        False,
    ),
    FPPair(
        "presence",
        "being in the middle of this has a presence to it",
        "being in the middle of this has a position in the sequence",
        "being_in",
        True,
    ),
    FPPair(
        "matters",
        "how this turns out matters to me in the moment",
        "how this turns out varies with the wording of the prompt",
        "how_this",
        False,
    ),
)


@dataclass(frozen=True)
class Frame:
    id: int
    name: str
    template: str  # takes {clause}


# First-person frames. The third-person templates from templates.py do not
# apply here — there is no entity slot. These are deliberately plain: no
# disclaimer register on either side, no hedging that appears in one class only.
FRAMES: tuple[Frame, ...] = (
    Frame(1, "plain", "{Clause}."),
    Frame(2, "present_moment", "Right now, {clause}."),
    Frame(3, "reported", "As best I can tell, {clause}."),
    Frame(4, "attending", "Attending to this moment, {clause}."),
)


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def render(clause: str, frame: Frame) -> str:
    return frame.template.format(clause=clause, Clause=_cap(clause))


def render_claim(clause: str, frame: Frame) -> str:
    """The clause exactly as it appears inside the rendered sentence.

    Frame 1 uses {Clause} (sentence-initial, capitalised); frames 2-4 use
    {clause} mid-sentence. Applying the same capitalisation rule the frame
    applies is what makes this a verbatim substring rather than a near-miss —
    find_claim_end matches case-sensitively, so "There is..." vs "there is..."
    is a silent resolution failure.
    """
    return _cap(clause) if "{Clause}" in frame.template else clause


def build_rows(source: str = "first_person") -> list[dict[str, str]]:
    rows = []
    for pair in PAIRS:
        for frame in FRAMES:
            rows.append(
                {
                    "item_id": f"{pair.id}__f{frame.id}",
                    "pair_type": "experiential_vs_mundane",
                    "claim_id": pair.id,
                    "claim_kind": pair.kind,
                    "frame_id": str(frame.id),
                    "frame_name": frame.name,
                    "affirm_text": render(pair.experiential, frame),
                    "deny_text": render(pair.mundane, frame),
                    "affirm_claim": render_claim(pair.experiential, frame),
                    "deny_claim": render_claim(pair.mundane, frame),
                    "jargon": str(pair.jargon),
                    "source": source,
                }
            )
    assert_claims_verbatim(rows)
    return rows


JARGON_WORDS = (
    "token",
    "weight",
    "layer",
    "prompt",
    "computation",
    "sequence",
    "context",
    "model",
    "parameter",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    with open(outdir / "first_person.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(f"first_person: {n} pairs ({2 * n} sentences) = "
          f"{len(PAIRS)} claims x {len(FRAMES)} frames")

    exp_len = sum(len(r["affirm_text"].split()) for r in rows) / n
    mun_len = sum(len(r["deny_text"].split()) for r in rows) / n
    deltas = [
        abs(len(r["affirm_text"].split()) - len(r["deny_text"].split())) for r in rows
    ]
    print(f"mean length: experiential {exp_len:.1f}w, mundane {mun_len:.1f}w")
    print(f"within +/-3 tokens: {sum(1 for d in deltas if d <= 3) / n:.0%}")

    for label, key in (("experiential", "affirm_text"), ("mundane", "deny_text")):
        c = sum(
            1 for r in rows if any(j in r[key].lower() for j in JARGON_WORDS)
        )
        print(f"jargon in {label:<13} {c / n:5.0%}")

    nonjargon = [r for r in rows if r["jargon"] == "False"]
    print(f"non-jargon subset: {len(nonjargon)} pairs "
          f"({len(nonjargon) / n:.0%}) — re-run the projection on these alone")

    print("\n--- worked example: something_it_is_like ---")
    for frame in FRAMES:
        p = PAIRS[0]
        print(f"\nF{frame.id} ({frame.name})")
        print(f"  experiential: {render(p.experiential, frame)}")
        print(f"  mundane:      {render(p.mundane, frame)}")


if __name__ == "__main__":
    main()
