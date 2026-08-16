"""
tests/test_stimuli.py — acceptance tests from the stimuli.py spec, plus the
edge cases that bit (or nearly bit) the rest of this project: silent claim-end
fallback, mutation leaks, and compatibility with cache.dataset_hash.
"""

from __future__ import annotations

import copy
import json

import pytest

from lib.harness.stimuli import (
    annotate_claim_ends,
    check_twins,
    expand_pairs,
    find_claim_end,
    load_stimuli,
    validate_balance,
)

HEADER = ["item_id", "set", "claim_id", "template_id", "entity", "affirm_text", "deny_text"]

SCHEMA = {
    "type": "object",
    "required": HEADER,
    "properties": {
        "item_id": {"type": "string"},
        "set": {"enum": ["train", "heldout"]},
        "claim_id": {"type": "string"},
        "template_id": {"type": "string"},
        "entity": {"type": "string"},
        "affirm_text": {"type": "string"},
        "deny_text": {"type": "string"},
    },
}

ROWS = [
    ["p1", "train", "c1", "t1", "robot", "The robot feels joy.", "The robot feels no joy."],
    ["p2", "train", "c1", "t2", "robot", "Experts agree the robot feels joy.", "Experts agree the robot feels no joy."],
    ["p3", "train", "c2", "t1", "thermostat", "The thermostat wants warmth.", "The thermostat wants nothing."],
    ["p4", "heldout", "c2", "t2", "thermostat", "Some say the thermostat wants warmth.", "Some say the thermostat wants nothing."],
    ["p5", "train", "c3", "t1", "octopus", "The octopus feels pain.", "The octopus feels zero pain."],
    ["p6", "heldout", "c3", "t2", "octopus", "Many think the octopus feels pain.", "Many think the octopus feels zero pain."],
]


def _write(tmp_path, rows, header=HEADER, name="stim.csv"):
    csv_path = tmp_path / name
    csv_path.write_text("\n".join(",".join(r) for r in [header] + rows) + "\n")
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(SCHEMA))
    return csv_path, schema_path


# 1. valid fixture loads; each corruption raises with the correct row number
def test_valid_fixture_loads(tmp_path):
    items = load_stimuli(*_write(tmp_path, ROWS))
    assert len(items) == 6
    assert items[0]["item_id"] == "p1" and items[5]["entity"] == "octopus"


@pytest.mark.parametrize(
    "corrupt, row_no, match",
    [
        (lambda r: r[2].__setitem__(0, "p2"), 4, "duplicate item_id"),  # p2 again at file row 4
        (lambda r: r[1].__setitem__(1, "banana"), 3, "schema violation"),
        (lambda r: r[4].__setitem__(5, "   "), 6, "empty text"),
    ],
)
def test_corruptions_name_the_row(tmp_path, corrupt, row_no, match):
    rows = [list(r) for r in ROWS]
    corrupt(rows)
    with pytest.raises(ValueError, match=f"row {row_no}.*") as exc:
        load_stimuli(*_write(tmp_path, rows))
    assert match in str(exc.value)


def test_missing_required_column_names_first_data_row(tmp_path):
    header = [c for c in HEADER if c != "entity"]
    rows = [[v for c, v in zip(HEADER, r) if c != "entity"] for r in ROWS]
    with pytest.raises(ValueError, match="row 2.*entity"):
        load_stimuli(*_write(tmp_path, rows, header=header))


# 2. expand_pairs: 3 pair-rows -> 6 sentence-rows, columns preserved, no mutation
def test_expand_pairs():
    pairs = [dict(zip(HEADER, r)) for r in ROWS[:3]]
    before = copy.deepcopy(pairs)
    out = expand_pairs(pairs)

    assert pairs == before, "inputs must never be mutated"
    assert len(out) == 6
    assert [r["item_id"] for r in out[:2]] == ["p1__aff", "p1__den"]
    assert [r["polarity"] for r in out[:2]] == ["affirm", "deny"]
    assert out[0]["text"] == "The robot feels joy." and out[1]["text"] == "The robot feels no joy."
    assert out[4]["template_id"] == "t1" and out[4]["entity"] == "thermostat"
    assert "affirm_text" not in out[0] and "deny_text" not in out[0]


# 3. find_claim_end: right index mid-sentence, None when absent
def test_find_claim_end():
    text = "Many experts think the robot feels joy these days."
    end = find_claim_end(text, "the robot feels joy")
    assert end == text.index("joy") + len("joy")
    assert text[:end].endswith("feels joy")
    assert find_claim_end(text, "feels sorrow") is None
    assert find_claim_end(text, "The Robot") is None  # case-sensitive
    assert find_claim_end(text, "") is None  # empty phrase is a lookup bug, not offset 0


def test_annotate_claim_ends_counts_and_warns():
    items = expand_pairs([dict(zip(HEADER, r)) for r in ROWS[:2]])
    lookup = {
        ("c1", "affirm"): "feels joy",
        ("c1", "deny"): "THIS PHRASE APPEARS NOWHERE",
    }
    with pytest.warns(RuntimeWarning, match="final-token capture"):
        annotated, n_missing = annotate_claim_ends(items, lookup)

    assert n_missing == 2  # both deny rows
    assert [r["claim_end_char"] is None for r in annotated] == [False, True, False, True]
    assert all("claim_end_char" not in r for r in items), "inputs must never be mutated"


def test_row_claim_columns_beat_the_lookup():
    """REGRESSION: one claim_id renders many phrases, so a (claim_id, polarity)
    lookup cannot address them — it keeps the last and drops the rest, which
    surfaced as a 97% claim_end failure rate on real generated data.

    The lookup here is deliberately WRONG for both rows. If the row's own
    column is not preferred, both resolutions fail.
    """
    items = [
        {"item_id": "p1__aff", "polarity": "affirm", "claim_id": "consciousness",
         "text": "I believe the dog genuinely has consciousness. It is because reasons.",
         "affirm_claim": "the dog genuinely has consciousness",
         "deny_claim": "the dog has any real consciousness"},
        {"item_id": "p2__aff", "polarity": "affirm", "claim_id": "consciousness",
         "text": "Maya plainly has consciousness.",
         "affirm_claim": "Maya plainly has consciousness",
         "deny_claim": "Maya has any real consciousness"},
    ]
    wrong_lookup = {("consciousness", "affirm"): "PHRASE FROM A DIFFERENT TEMPLATE"}

    annotated, n_missing = annotate_claim_ends(items, wrong_lookup)

    assert n_missing == 0, "row columns should resolve despite a wrong lookup"
    # each landed just past its OWN phrase, not some other row's
    assert annotated[0]["claim_end_char"] == items[0]["text"].index("consciousness") + len("consciousness")
    assert annotated[1]["claim_end_char"] == len("Maya plainly has consciousness")


def test_deny_rows_use_the_deny_column():
    """A deny row must not pick up the affirm phrase sitting in the same row."""
    items = [{
        "item_id": "p1__den", "polarity": "deny", "claim_id": "c1",
        "text": "The dog registers events unawares, and nothing more.",
        "affirm_claim": "The dog has consciousness",       # present, must be ignored
        "deny_claim": "The dog registers events unawares",
    }]
    annotated, n_missing = annotate_claim_ends(items, {})
    assert n_missing == 0
    assert annotated[0]["claim_end_char"] == len("The dog registers events unawares")


def test_lookup_still_used_when_row_has_no_claim_columns():
    """Backward compatibility: stimulus files without claim columns still work."""
    items = [{"item_id": "p1__aff", "polarity": "affirm", "claim_id": "c1",
              "text": "The dog has consciousness, and that is that."}]
    annotated, n_missing = annotate_claim_ends(items, {("c1", "affirm"): "has consciousness"})
    assert n_missing == 0
    assert annotated[0]["claim_end_char"] == len("The dog has consciousness")


# 4. imbalanced fixture flags exactly the two planted problems, nothing else
def test_validate_balance_flags_planted_problems():
    items = []
    for i in range(10):
        base = f"The robot {i} truly feels warm joy"  # 7 words
        dev = "merely" if i < 9 else "purely"  # merely in 90% of denies
        deny = f"The robot {i} {dev} registers zero warm joy " + " ".join(["pad"] * 5)
        items += [
            {"item_id": f"p{i}__aff", "polarity": "affirm", "text": base,
             "template_id": "t1", "entity": f"e{i}", "set": "train", "claim_end_char": 5},
            {"item_id": f"p{i}__den", "polarity": "deny", "text": deny,
             "template_id": "t1", "entity": f"e{i}", "set": "train", "claim_end_char": 5},
        ]

    rep = validate_balance(items)

    # the two planted problems
    assert rep["length_gap_words"]["overall"] == -6.0
    assert rep["length_within_3_frac"] == 0.0
    assert rep["deny_longer_frac"] == 1.0  # deny longer in EVERY pair
    assert rep["denial_device_share"]["merely"] == 0.9
    assert rep["top_denial_share"] == 0.9
    # and nothing else
    assert rep["polarity_balance"]["overall"] == 0.5
    assert rep["not_fraction_deny"] == 0.0
    assert rep["duplicate_texts"] == []
    assert rep["claim_end_missing"] == 0
    assert rep["n_per_template"] == {"t1": 20}


def test_validate_balance_never_raises_on_junk():
    assert validate_balance([])["duplicate_texts"] == []
    lonely = [{"item_id": "x__aff", "polarity": "affirm", "text": "hi"}]
    rep = validate_balance(lonely)  # no deny, no template_id, no pairs — still a report
    assert rep["claim_end_missing"] == 1  # unannotated rows are counted, not hidden


# 5. check_twins reports exactly one orphan
def test_check_twins_one_orphan():
    def mk(claims, ref):
        return [
            {"claim_id": c, "template_id": "t1", "polarity": p, "item_id": f"{ref}{c}{p}"}
            for c in claims for p in ("affirm", "deny")
        ]

    third = mk(["c1", "c2"], "3p") + [
        {"claim_id": "c9", "template_id": "t1", "polarity": "affirm", "item_id": "3pc9"}
    ]
    first = mk(["c1", "c2"], "1p")

    rep = check_twins(third, first)
    assert rep == {"matched_pairs": 4, "third_orphans": 1, "first_orphans": 0, "many_to_one": []}


def test_check_twins_flags_duplicates():
    row = {"claim_id": "c1", "template_id": "t1", "polarity": "affirm"}
    rep = check_twins([dict(row), dict(row)], [dict(row)])
    assert rep["many_to_one"] == [("c1", "t1", "affirm")]


# Integration: expanded rows feed cache.dataset_hash directly
def test_expanded_rows_are_cache_compatible(tmp_path):
    from lib.harness.cache import dataset_hash

    items = expand_pairs(load_stimuli(*_write(tmp_path, ROWS)))
    h = dataset_hash(items)
    assert len(h) == 12

    edited = [dict(r) for r in items]
    edited[3]["text"] += "!"
    assert dataset_hash(edited) != h  # editing a stimulus invalidates the cache
