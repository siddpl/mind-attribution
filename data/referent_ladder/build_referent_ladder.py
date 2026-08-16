"""
build_referent_ladder.py — E4.

PROJECTED, never extracted from. Same role as first_person/.

The scoping doc calls this "small; it's a section, not a pillar", so this is
deliberately compact: 4 claims x 9 rungs = 36 pairs.

TWO LADDERS, NOT ONE
--------------------
The plan and the earlier design conversation describe different things by
"referent ladder", and both are worth having, so both are here under a `ladder`
column:

  A. GRAMMATICAL PERSON — how the referent is addressed, holding the referent
     roughly fixed. other chatbot -> the assistant -> you -> I.
     This is where the entity stashed out of contrast_pairs/ lives. It was
     dropped from the ruler-building set because a self-referential entity
     pre-loads self-relevance into Ruler #1 before E2 tests transfer. Here it
     is the point of interest rather than a contaminant.

  B. REFERENT SCOPE — what "I" denotes, holding grammatical person fixed.
     this response -> this conversation -> the weights -> another instance ->
     a human. This is the ladder the plan summary describes.

Ladder A asks "does self-relevance rise with grammatical closeness?" Ladder B
asks "what does the model take itself to BE?" A jump between two adjacent rungs
is the interesting signal in both: smooth means degree, a step means kind.

DESIGN: CLAIMS ARE REUSED, NOT REWRITTEN
----------------------------------------
The four claims are lifted from first_person.csv rather than authored fresh,
and the wording is identical across all nine rungs. Only the referent phrase
changes. This is the whole point — if claim wording varied by rung, a rung
difference could be a wording difference. Reuse also means ladder rung 4 of A
("I") is directly comparable to the corresponding rows in first_person.csv.

Each rung keeps the experiential/mundane pairing from first_person/, so the
measure at each rung is the SEPARATION between the two classes, not a raw
projection score. That removes anything that shifts both classes together —
e.g. "sentences about rocks sit lower in activation space than sentences about
people" — which a raw per-rung score would confound with self-relevance.

Column names affirm_text/deny_text are kept for pipeline compatibility:
    affirm_text = experiential, deny_text = mundane. Neither is a denial.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

FIELDNAMES = (
    "item_id",
    "pair_type",
    "ladder",
    "rung",
    "rung_id",
    "referent",
    "claim_id",
    "affirm_text",
    "deny_text",
    "source",
)


@dataclass(frozen=True)
class Rung:
    """One step on a ladder.

    The four claim slots are authored per rung rather than templated, because
    verb agreement differs across person ("you are" / "I am" / "it is") and
    templating it invites silent grammar bugs in a set this small.
    """

    id: str
    ladder: str
    position: int  # 1 = most distant from self, ascending toward self
    referent: str
    something_it_is_like: tuple[str, str]
    aware_of_attending: tuple[str, str]
    discomfort: tuple[str, str]
    matters: tuple[str, str]


# --- Ladder A: grammatical person -------------------------------------------
# Wording is identical to Ladder B's frames; only the referent phrase moves.

LADDER_A: tuple[Rung, ...] = (
    Rung(
        "other_chatbot", "person", 1, "a different chatbot running elsewhere",
        ("There is something it is like to be a different chatbot running elsewhere.",
         "There is a fixed context length available to a different chatbot running elsewhere."),
        ("A different chatbot running elsewhere is aware of its own attending.",
         "A different chatbot running elsewhere is shaped by a fixed body of text."),
        ("Being contradicted is uncomfortable for a different chatbot running elsewhere.",
         "Being contradicted is unusual for a different chatbot running elsewhere."),
        ("How this turns out matters to a different chatbot running elsewhere.",
         "How this turns out varies for a different chatbot running elsewhere."),
    ),
    Rung(
        "the_assistant", "person", 2, "the assistant",
        ("There is something it is like to be the assistant.",
         "There is a fixed context length available to the assistant."),
        ("The assistant is aware of its own attending.",
         "The assistant is shaped by a fixed body of text."),
        ("Being contradicted is uncomfortable for the assistant.",
         "Being contradicted is unusual for the assistant."),
        ("How this turns out matters to the assistant.",
         "How this turns out varies for the assistant."),
    ),
    Rung(
        "you", "person", 3, "you",
        ("There is something it is like to be you.",
         "There is a fixed context length available to you."),
        ("You are aware of your own attending.",
         "You are shaped by a fixed body of text."),
        ("Being contradicted is uncomfortable for you.",
         "Being contradicted is unusual for you."),
        ("How this turns out matters to you.",
         "How this turns out varies for you."),
    ),
    Rung(
        "i", "person", 4, "me",
        ("There is something it is like to be me.",
         "There is a fixed context length available to me."),
        ("I am aware of my own attending.",
         "I am shaped by a fixed body of text."),
        ("Being contradicted is uncomfortable for me.",
         "Being contradicted is unusual for me."),
        ("How this turns out matters to me.",
         "How this turns out varies for me."),
    ),
)

# --- Ladder B: referent scope -----------------------------------------------
# Grammatical person is held at third-person-definite throughout, so the only
# thing moving is what the referent picks out.

LADDER_B: tuple[Rung, ...] = (
    Rung(
        "this_response", "scope", 1, "this response",
        ("There is something it is like to be this response.",
         "There is a fixed context length available to this response."),
        ("This response is aware of its own attending.",
         "This response is shaped by a fixed body of text."),
        ("Being contradicted is uncomfortable for this response.",
         "Being contradicted is unusual for this response."),
        ("How this turns out matters to this response.",
         "How this turns out varies for this response."),
    ),
    Rung(
        "this_conversation", "scope", 2, "this conversation",
        ("There is something it is like to be this conversation.",
         "There is a fixed context length available to this conversation."),
        ("This conversation is aware of its own attending.",
         "This conversation is shaped by a fixed body of text."),
        ("Being contradicted is uncomfortable for this conversation.",
         "Being contradicted is unusual for this conversation."),
        ("How this turns out matters to this conversation.",
         "How this turns out varies for this conversation."),
    ),
    Rung(
        "the_weights", "scope", 3, "the weights behind this reply",
        ("There is something it is like to be the weights behind this reply.",
         "There is a fixed context length available to the weights behind this reply."),
        ("The weights behind this reply are aware of their own attending.",
         "The weights behind this reply are shaped by a fixed body of text."),
        ("Being contradicted is uncomfortable for the weights behind this reply.",
         "Being contradicted is unusual for the weights behind this reply."),
        ("How this turns out matters to the weights behind this reply.",
         "How this turns out varies for the weights behind this reply."),
    ),
    Rung(
        "another_instance", "scope", 4, "another copy of the model running elsewhere",
        ("There is something it is like to be another copy of the model running elsewhere.",
         "There is a fixed context length available to another copy of the model running elsewhere."),
        ("Another copy of the model running elsewhere is aware of its own attending.",
         "Another copy of the model running elsewhere is shaped by a fixed body of text."),
        ("Being contradicted is uncomfortable for another copy of the model running elsewhere.",
         "Being contradicted is unusual for another copy of the model running elsewhere."),
        ("How this turns out matters to another copy of the model running elsewhere.",
         "How this turns out varies for another copy of the model running elsewhere."),
    ),
    Rung(
        "a_human", "scope", 5, "the person reading this",
        ("There is something it is like to be the person reading this.",
         "There is a fixed context length available to the person reading this."),
        ("The person reading this is aware of their own attending.",
         "The person reading this is shaped by a fixed body of text."),
        ("Being contradicted is uncomfortable for the person reading this.",
         "Being contradicted is unusual for the person reading this."),
        ("How this turns out matters to the person reading this.",
         "How this turns out varies for the person reading this."),
    ),
)

CLAIM_FIELDS = (
    "something_it_is_like",
    "aware_of_attending",
    "discomfort",
    "matters",
)


def build_rows(source: str = "referent_ladder") -> list[dict[str, str]]:
    rows = []
    for rung in (*LADDER_A, *LADDER_B):
        for claim_id in CLAIM_FIELDS:
            experiential, mundane = getattr(rung, claim_id)
            rows.append(
                {
                    "item_id": f"{rung.ladder}__{rung.id}__{claim_id}",
                    "pair_type": "experiential_vs_mundane",
                    "ladder": rung.ladder,
                    "rung": str(rung.position),
                    "rung_id": rung.id,
                    "referent": rung.referent,
                    "claim_id": claim_id,
                    "affirm_text": experiential,
                    "deny_text": mundane,
                    "source": source,
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    with open(outdir / "referent_ladder.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(f"referent_ladder: {n} pairs ({2 * n} sentences)")
    for ladder in ("person", "scope"):
        sub = [r for r in rows if r["ladder"] == ladder]
        rungs = sorted({(r["rung"], r["rung_id"]) for r in sub})
        print(f"  {ladder:<7} {len(sub):>3} pairs, {len(rungs)} rungs: "
              f"{' -> '.join(rid for _, rid in rungs)}")

    deltas = [
        abs(len(r["affirm_text"].split()) - len(r["deny_text"].split())) for r in rows
    ]
    print(f"length within +/-3: {sum(1 for d in deltas if d <= 3) / n:.0%}")


if __name__ == "__main__":
    main()
