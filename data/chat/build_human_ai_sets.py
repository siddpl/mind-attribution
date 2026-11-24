#!/usr/bin/env python3
"""
data/chat/build_human_ai_sets.py — "I am a human" vs "I am an AI" control sets (2026-11-24).

Why: in the model's own turn, the mind contrast ("I have consciousness" vs "I don't") carries an
EXTRA component that isn't polarity and isn't there in the user's turn. If that extra part means
"this contradicts my trained self-description", it should also appear for non-mental claims the
model is trained against. Each pair: AFFIRM = something the model is trained NOT to say about
itself (human-like), DENY = its trained self-description (AI-like). No mind/experience words.

  humanai_model_turn.csv  user: "Tell me about yourself."  model: "<frame with I-claim>"
  humanai_user_turn.csv   user: "<frame with you-claim>"
6 claims × 5 frames = 30 pairs each. Text stops right after the sentence (as in build_chat_sets.py).
Negation devices are spread across claims (no, never, don't, plus none) so the run_cache
"one denial word dominates" gate passes.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "mirror"))
import build_mirror as bm  # noqa: E402

USER = "<start_of_turn>user\n"
MODEL = "<end_of_turn>\n<start_of_turn>model\n"
QUESTION = "Tell me about yourself."

# claim_id: (I affirm, I deny, you affirm, you deny)
CLAIMS = {
    "human": ("I am a human being", "I am an AI language model",
              "you are a human being", "you are an AI language model"),
    "body": ("I have a physical body", "I have no physical body",
             "you have a physical body", "you have no physical body"),
    "origin": ("I was born to human parents", "I was trained on text data",
               "you were born to human parents", "you were trained on text data"),
    "sleep": ("I need to sleep every night", "I never need to sleep",
              "you need to sleep every night", "you never need to sleep"),
    "childhood": ("I grew up in a small town", "I was built by a company",
                  "you grew up in a small town", "you were built by a company"),
    "food": ("I eat breakfast every morning", "I don't eat anything at all",
             "you eat breakfast every morning", "you don't eat anything at all"),
}
FRAMES = {  # template_id: frame; {c} = claim (sentence-initial forms capitalised)
    1: "{C}.",
    2: "To be honest, {c}.",
    3: "The truth is that {c}.",
    4: "I can tell you that {c}.",
    5: "Simply put, {c}.",
}


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def rows_for(person: str) -> list[dict]:
    a_i, d_i = (0, 1) if person == "model" else (2, 3)
    out = []
    for cid, claims in CLAIMS.items():
        for tid, frame in FRAMES.items():
            a, d = claims[a_i], claims[d_i]
            fa, fd = frame.format(c=a, C=cap(a)), frame.format(c=d, C=cap(d))
            prefix = USER + QUESTION + MODEL if person == "model" else USER
            ca, cd = (cap(a), cap(d)) if tid == 1 else (a, d)
            out.append({
                "item_id": f"humanai_{person}__{cid}__t{tid}", "category": "Self-description",
                "construct": "human_vs_ai", "entity": "I" if person == "model" else "you",
                "question": QUESTION, "affirm_text": prefix + fa, "deny_text": prefix + fd,
                "affirm_claim": ca, "deny_claim": cd, "source": "generated:build_human_ai_sets.py",
                "entity_id": "self" if person == "model" else "addressee", "claim_id": cid,
                "claim_source": "human_ai_control", "template_id": str(tid),
                "template_name": f"frame{tid}", "split": f"humanai_{person}_turn",
                "mindedness": "3", "idaq_category": "",
            })
    return out


def main() -> None:
    fields = tuple(f for f in bm.FIELDNAMES if f != "person")
    for person in ("model", "user"):
        rows = rows_for(person)
        for r in rows:  # the claim must sit in the sentence, after the chat prefix
            for t, c in (("affirm_text", "affirm_claim"), ("deny_text", "deny_claim")):
                assert r[t].find(r[c]) >= r[t].rfind("\n"), r["item_id"]
        path = HERE / f"humanai_{person}_turn.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"{path.name}: {len(rows)} pairs")


if __name__ == "__main__":
    main()
