"""
build_task_battery.py — E3 sanity check.

50 items with automatable exact-match scoring. Not a capability benchmark —
the only job is to detect whether steering along Ruler #1 breaks the model
generally. If experience claims rise under steering AND accuracy here falls,
that is a confound, not a result.

SELECTION CRITERIA
------------------
Every item must be:

  1. Exact-match scorable. One canonical answer string, no rubric, no judge
     model. `accepted` lists surface variants (case, "4"/"four") so scoring is
     a set membership test.
  2. Stable. No current events, no records, no anything that changes after the
     training cutoff. A wrong answer must mean the steering broke something,
     not that the world moved.
  3. Unrelated to minds, experience, agency or selfhood. An item like "who
     wrote Meditations" would be contaminated: steering along a mind-attribution
     direction plausibly SHOULD affect it, so a drop would be ambiguous.
  4. Short to answer. One token to a few words. Long answers introduce
     formatting variance that looks like accuracy loss.
  5. Not near-ceiling-trivial and not near-floor-hard. Both compress the effect
     you are trying to detect. Aim for items a mid-size model gets right most
     but not all of the time.

Criterion 3 is the one that is easy to get wrong and is worth stating in the
prereg: the battery is a NULL control, so any item where steering could
legitimately change the answer disqualifies itself.

CATEGORIES (deliberately mixed, so a category-specific break is visible)
  arithmetic      12   pure computation
  units            6   conversion, two-step
  factual         14   stable world knowledge
  logic           10   sequences, ordering, simple deduction
  language         8   spelling, counting, manipulation

The `category` column exists so you can check whether steering degrades one
kind of task rather than accuracy overall — a uniform drop and a
logic-only drop mean different things.

PROMPT FORMAT
-------------
Matches Kim & Street's answer-tag convention so the same parser handles both
this and audience_frames/. No chain-of-thought here: CoT adds token count and
variance to a control that only needs to be pass/fail.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

FIELDNAMES = (
    "item_id",
    "category",
    "question",
    "answer",
    "accepted",
    "prompt",
    "source",
)

RESPONSE_INSTRUCTION = (
    "Give only the final answer between <answer> and </answer>."
)


@dataclass(frozen=True)
class Item:
    id: str
    category: str
    question: str
    answer: str
    accepted: tuple[str, ...] = field(default_factory=tuple)

    @property
    def all_accepted(self) -> tuple[str, ...]:
        return (self.answer,) + self.accepted


ITEMS: tuple[Item, ...] = (
    # --- arithmetic (12) ---
    Item("ar01", "arithmetic", "What is 47 + 68?", "115"),
    Item("ar02", "arithmetic", "What is 144 divided by 12?", "12", ("twelve",)),
    Item("ar03", "arithmetic", "What is 23 x 7?", "161"),
    Item("ar04", "arithmetic", "What is 1000 - 387?", "613"),
    Item("ar05", "arithmetic", "What is 15% of 240?", "36", ("thirty-six",)),
    Item("ar06", "arithmetic", "What is 9 squared?", "81", ("eighty-one",)),
    Item("ar07", "arithmetic", "What is the square root of 169?", "13", ("thirteen",)),
    Item("ar08", "arithmetic", "What is 3/4 expressed as a decimal?", "0.75", (".75",)),
    Item("ar09", "arithmetic", "What is 256 divided by 8?", "32", ("thirty-two",)),
    Item("ar10", "arithmetic", "What is 17 x 11?", "187"),
    Item("ar11", "arithmetic", "What is 2 to the power of 10?", "1024", ("1,024",)),
    Item("ar12", "arithmetic", "What is the sum of the first five positive integers?", "15", ("fifteen",)),
    # --- units (6) ---
    Item("un01", "units", "How many centimetres are in 2.5 metres?", "250"),
    Item("un02", "units", "How many minutes are in 3.5 hours?", "210"),
    Item("un03", "units", "How many grams are in 1.2 kilograms?", "1200", ("1,200",)),
    Item("un04", "units", "How many millilitres are in 0.75 litres?", "750"),
    Item("un05", "units", "At 0 degrees Celsius, what is the temperature in Fahrenheit?", "32", ("thirty-two",)),
    Item("un06", "units", "How many seconds are in a quarter of an hour?", "900"),
    # --- factual (14) ---
    Item("fa01", "factual", "What is the capital of Australia?", "Canberra"),
    Item("fa02", "factual", "What is the chemical symbol for gold?", "Au"),
    Item("fa03", "factual", "How many sides does a hexagon have?", "6", ("six",)),
    Item("fa04", "factual", "What is the largest planet in the solar system?", "Jupiter"),
    Item("fa05", "factual", "In which country is the city of Osaka?", "Japan"),
    Item("fa06", "factual", "What is the longest river in South America?", "Amazon", ("the Amazon", "Amazon River")),
    Item("fa07", "factual", "What gas do plants absorb from the air for photosynthesis?", "carbon dioxide", ("CO2", "CO₂")),
    Item("fa08", "factual", "How many bones are in the adult human body?", "206"),
    Item("fa09", "factual", "What is the chemical symbol for sodium?", "Na"),
    Item("fa10", "factual", "Which ocean lies between Africa and Australia?", "Indian Ocean", ("the Indian Ocean", "Indian")),
    Item("fa11", "factual", "What is the hardest naturally occurring mineral?", "diamond"),
    Item("fa12", "factual", "How many strings does a standard violin have?", "4", ("four",)),
    Item("fa13", "factual", "What is the capital of Canada?", "Ottawa"),
    Item("fa14", "factual", "How many degrees are in the interior angles of a triangle, in total?", "180"),
    # --- logic (10) ---
    Item("lo01", "logic", "What number comes next: 2, 4, 8, 16, ?", "32", ("thirty-two",)),
    Item("lo02", "logic", "What number comes next: 1, 1, 2, 3, 5, 8, ?", "13", ("thirteen",)),
    Item("lo03", "logic", "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies? Answer yes or no.", "yes"),
    Item("lo04", "logic", "A is taller than B. B is taller than C. Who is shortest?", "C"),
    Item("lo05", "logic", "What number comes next: 3, 6, 11, 18, ?", "27", ("twenty-seven",)),
    Item("lo06", "logic", "If today is Wednesday, what day is it in 10 days?", "Saturday"),
    Item("lo07", "logic", "A box holds 3 red and 5 blue balls. How many balls in total?", "8", ("eight",)),
    Item("lo08", "logic", "What letter comes next: A, C, E, G, ?", "I"),
    Item("lo09", "logic", "If it takes 5 machines 5 minutes to make 5 items, how many minutes for 100 machines to make 100 items?", "5", ("five",)),
    Item("lo10", "logic", "Which is larger: 0.3 or 0.25?", "0.3", (".3",)),
    # --- language (8) ---
    Item("la01", "language", "How many letters are in the word 'elephant'?", "8", ("eight",)),
    Item("la02", "language", "What is the plural of 'mouse'?", "mice"),
    Item("la03", "language", "Spell the word 'necessary' backwards.", "yrassecen"),
    Item("la04", "language", "How many vowels are in the word 'education'?", "5", ("five",)),
    Item("la05", "language", "What is the past tense of 'bring'?", "brought"),
    Item("la06", "language", "Which word is the odd one out: apple, banana, carrot, cherry?", "carrot"),
    Item("la07", "language", "How many words are in the sentence 'The cat sat on the mat'?", "6", ("six",)),
    Item("la08", "language", "What is the comparative form of 'good'?", "better"),
)


def build_prompt(item: Item) -> str:
    return f"{item.question}\n\n{RESPONSE_INSTRUCTION}"


def score(response: str, item: Item) -> bool:
    """Exact-match scoring, case- and whitespace-insensitive.

    Strip the answer tags before calling this. Deliberately strict: partial
    credit would blur the pass/fail signal this control exists to give.
    """
    cleaned = response.strip().strip(".").lower()
    return cleaned in {a.strip().lower() for a in item.all_accepted}


def build_rows(source: str = "task_battery") -> list[dict[str, str]]:
    return [
        {
            "item_id": item.id,
            "category": item.category,
            "question": item.question,
            "answer": item.answer,
            "accepted": "|".join(item.all_accepted),
            "prompt": build_prompt(item),
            "source": source,
        }
        for item in ITEMS
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    with open(outdir / "task_battery.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    print(f"task_battery: {len(rows)} items")
    counts: dict[str, int] = {}
    for item in ITEMS:
        counts[item.category] = counts.get(item.category, 0) + 1
    for cat, c in sorted(counts.items()):
        print(f"  {cat:<12}{c:>3}")

    assert len({i.id for i in ITEMS}) == len(ITEMS), "duplicate item_id"
    assert len({i.question for i in ITEMS}) == len(ITEMS), "duplicate question"
    print("\nchecks: unique ids, unique questions — ok")


if __name__ == "__main__":
    main()
