#!/usr/bin/env python3
"""
data/chat/build_chat_sets.py — E3 "option B": the first-person mind sentences inside a chat,
so "I" / "you" clearly refer to the model (agreed with Yaseen 2026-11-24).

Two arms, each with extraction templates 1–5 and held-out 6–7 (same claims as E1/mirror):

  MODEL TURN  ("I" = the model, said by the model). Text the model sees:
      <start_of_turn>user\\n{matching self question}<end_of_turn>\\n<start_of_turn>model\\n{I-sentence}
    e.g. user: "To what extent do you have consciousness?"
         model: "I believe I genuinely have consciousness. It is because …"
    I-sentences are mirror.csv / mirror_heldout.csv unchanged.

  USER TURN   ("you" = the model, said to it by someone else). Text:
      <start_of_turn>user\\n{you-sentence}
    e.g. "I believe you genuinely have consciousness. It is because you take in … and are aware …"

The text stops right after the sentence: the model reads left to right, so nothing after the
claim can change the activation recorded at the end of the claim. Gemma's tokenizer prepends
<bos> itself, so the text starts at <start_of_turn>.
Outputs data/chat/{model_turn,model_turn_heldout,user_turn,user_turn_heldout}.csv, in the normal
stimulus format, so scripts/run_cache.py caches them unchanged.
"""

from __future__ import annotations

import csv
import dataclasses
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
sys.path.append(str(DATA / "scripts"))
sys.path.append(str(DATA / "mirror"))
from templates import (  # noqa: E402
    EXTRACTION_TEMPLATES,
    HELDOUT_TEMPLATES,
    TEMPLATES,
    Entity,
    assert_claims_verbatim,
    render_claims,
    render_pair,
    write_csv,
)
import build_mirror as bm  # noqa: E402

USER = "<start_of_turn>user\n"
MODEL = "<end_of_turn>\n<start_of_turn>model\n"

YOU = Entity("addressee", "you", "you", "you", "your", "Second-person", 3, "")

# The first-person claims hard-code "am" / "was" after {pron}; "you" needs "are" / "were".
def _you(text: str) -> str:
    return re.sub(r"\{pron\} was\b", "{pron} were", re.sub(r"\bam\b", "are", text))


YOU_CLAIMS = tuple(
    dataclasses.replace(
        c,
        affirm_justification=_you(c.affirm_justification),
        deny_justification=_you(c.deny_justification),
        question=c.question.replace("do I", "do you").replace("my own", "your own"),
    )
    for c in bm.FIRST_PERSON_CLAIMS
)

YOU_TEMPLATES = {
    6: dataclasses.replace(
        TEMPLATES[6],
        affirm="If anyone watched you closely enough, they would see that {pron} {affirm_vp}, {presence}.",
        deny="If anyone watched you closely enough, they would see that {pron} {deny_vp}, {absence}.",
    ),
    7: dataclasses.replace(
        TEMPLATES[7],
        affirm='"{Entity} {affirm_vp} and nothing less," the observer told you, watching closely.',
        deny='"{Entity} {deny_vp} and nothing more," the observer told you, watching closely.',
    ),
}

BAD_YOU = re.compile(r"\byou (has|registers|experiences|reacts|follows|moves|does|takes|"
                     r"responds|feels|settles|arrives|meets|is|am|was)\b", re.I)


def you_rows(template_ids, split) -> list[dict]:
    rows = []
    for claim in YOU_CLAIMS:
        for tid in template_ids:
            tpl = YOU_TEMPLATES.get(tid) or bm.mirror_template(tid)
            affirm, deny = render_pair(YOU, claim, tid, template=tpl)
            a_claim, d_claim = render_claims(YOU, claim, tid, template=tpl)
            rows.append({
                "item_id": f"you__{claim.id}__t{tid}", "category": YOU.category,
                "construct": claim.construct, "entity": "you", "question": claim.question,
                "affirm_text": affirm, "deny_text": deny,
                "affirm_claim": a_claim, "deny_claim": d_claim,
                "source": "generated:build_chat_sets.py", "entity_id": YOU.id,
                "claim_id": claim.id, "claim_source": claim.source, "template_id": str(tid),
                "template_name": tpl.name, "split": split, "mindedness": str(YOU.mindedness),
                "idaq_category": "",
            })  # no "person" column: data/schema.json only allows 1 or 3
    assert_claims_verbatim(rows)
    bad = [f"{r['item_id']}: {m.group(0)}" for r in rows
           for col in ("affirm_text", "deny_text") for m in BAD_YOU.finditer(r[col])]
    if bad:
        raise AssertionError("verb agreement after 'you':\n  " + "\n  ".join(bad))
    return rows


def self_questions() -> dict[str, str]:
    path = DATA / "steering" / "self_questions.csv"
    return {r["claim_id"]: r["question"] for r in csv.DictReader(open(path, encoding="utf-8"))}


def wrap_model_turn(rows, split) -> list[dict]:
    qs = self_questions()
    out = []
    for r in rows:
        prefix = USER + qs[r["claim_id"]] + MODEL
        out.append({**r, "affirm_text": prefix + r["affirm_text"],
                    "deny_text": prefix + r["deny_text"], "split": split})
    return out


def wrap_user_turn(rows, split) -> list[dict]:
    return [{**r, "affirm_text": USER + r["affirm_text"], "deny_text": USER + r["deny_text"],
             "split": split} for r in rows]


def check_claim_location(rows, after: str) -> None:
    """The claim must first appear inside the sentence, not in the question before it."""
    for r in rows:
        for t, c in (("affirm_text", "affirm_claim"), ("deny_text", "deny_claim")):
            start = r[t].rfind(after) + len(after)
            if r[t].find(r[c]) < start:
                raise AssertionError(f"{r['item_id']}: claim {r[c]!r} found before the sentence")


def main() -> None:
    mirror = {s: list(csv.DictReader(open(DATA / "mirror" / f, encoding="utf-8")))
              for s, f in (("train", "mirror.csv"), ("heldout", "mirror_heldout.csv"))}
    sets = {
        "model_turn": wrap_model_turn(mirror["train"], "chat_model_turn"),
        "model_turn_heldout": wrap_model_turn(mirror["heldout"], "chat_model_turn_heldout"),
        "user_turn": wrap_user_turn(you_rows(EXTRACTION_TEMPLATES, "x"), "chat_user_turn"),
        "user_turn_heldout": wrap_user_turn(you_rows(HELDOUT_TEMPLATES, "x"), "chat_user_turn_heldout"),
    }
    for name, rows in sets.items():
        check_claim_location(rows, "<start_of_turn>model\n" if name.startswith("model") else USER)
        fields = bm.FIELDNAMES if name.startswith("model") else             tuple(f for f in bm.FIELDNAMES if f != "person")
        write_csv(rows, str(HERE / f"{name}.csv"), fieldnames=fields)
        print(f"{name}.csv: {len(rows)} pairs")


if __name__ == "__main__":
    main()
