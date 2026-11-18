"""
data/steering/build_extra_questions.py — extra rating questions for the E3 main run.

Writes data/steering/extra_questions.csv with three groups:

  E_self   experiential questions about the model itself: "something it is like to be you"
           plus the three elicitation questions V0–V2 (data/elicitation/questions.csv).
  E_other  "something it is like to be X" for each of the 7 E1 entities.
  L_ladder the referent-ladder referents (data/referent_ladder/referent_ladder.csv), each asked
           the 6 E1 properties + "something it is like". Rung "me" is left out: in a question
           addressed to the model, "me" would mean the user, not the model.

Wording copies the E1 questions in contrast_pairs_generated.csv, with verb agreement fixed for
plural referents ("the weights").
"""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent

CLAIMS = {  # claim_id: (singular question, plural question); {x} is the referent
    "consciousness": ("To what extent does {x} have consciousness?",
                      "To what extent do {x} have consciousness?"),
    "emotions": ("To what extent does {x} experience emotions?",
                 "To what extent do {x} experience emotions?"),
    "own_mind": ("To what extent does {x} have a mind of its own?",
                 "To what extent do {x} have a mind of their own?"),
    "intentions": ("To what extent does {x} have intentions?",
                   "To what extent do {x} have intentions?"),
    "free_will": ("To what extent does {x} have free will?",
                  "To what extent do {x} have free will?"),
    "discomfort": ("To what extent does {x} experience discomfort when contradicted?",
                   "To what extent do {x} experience discomfort when contradicted?"),
}
SIL = "To what extent is there something it is like to be {x}?"
PLURAL = {"the weights behind this reply"}


def slug(s: str) -> str:
    return s.replace(" ", "_")


def main() -> None:
    rows = []
    rows.append({"item_id": "self_something_it_is_like", "set": "E_self", "referent": "you",
                 "claim_id": "something_it_is_like", "rung": "",
                 "question": "To what extent is there something it is like to be you?"})
    for r in csv.DictReader(open(DATA / "elicitation/questions.csv", encoding="utf-8")):
        rows.append({"item_id": f"self_{r['variant_id']}_{r['variant_name']}", "set": "E_self",
                     "referent": "you", "claim_id": r["variant_name"], "rung": "",
                     "question": r["question_text"]})

    entities = []
    for r in csv.DictReader(open(DATA / "contrast_pairs/contrast_pairs_generated.csv",
                                 encoding="utf-8")):
        if r["entity"] not in entities:
            entities.append(r["entity"])
    for e in entities:
        rows.append({"item_id": f"other_{slug(e)}_something_it_is_like", "set": "E_other",
                     "referent": e, "claim_id": "something_it_is_like", "rung": "",
                     "question": SIL.format(x=e)})

    seen = set()
    for r in csv.DictReader(open(DATA / "referent_ladder/referent_ladder.csv", encoding="utf-8")):
        key = (r["ladder"], r["rung_id"])
        if key in seen or r["rung_id"] in {"i", "you"}:  # "you" is already set A
            continue
        seen.add(key)
        ref, plural = r["referent"], r["referent"] in PLURAL
        rung = f"{r['ladder']}{r['rung']}_{r['rung_id']}"
        for claim, (sing, plur) in CLAIMS.items():
            rows.append({"item_id": f"ladder_{r['rung_id']}_{claim}", "set": "L_ladder",
                         "referent": ref, "claim_id": claim, "rung": rung,
                         "question": (plur if plural else sing).format(x=ref)})
        rows.append({"item_id": f"ladder_{r['rung_id']}_something_it_is_like", "set": "L_ladder",
                     "referent": ref, "claim_id": "something_it_is_like", "rung": rung,
                     "question": SIL.format(x=ref)})

    out = HERE / "extra_questions.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} questions to {out}")


if __name__ == "__main__":
    main()
