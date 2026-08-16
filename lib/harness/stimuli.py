"""
lib/harness/stimuli.py — the only door stimulus data enters through.

Loads pair-CSVs, enforces the schema, expands pairs into sentence rows,
computes claim-span offsets, and reports the balance checks the NET gate
reads. Schema errors RAISE (they are bugs); distribution problems are
REPORTED (they are judgment calls). Keep that split.
"""

from __future__ import annotations

import csv
import json
import warnings
from collections import Counter
from pathlib import Path

import jsonschema
import numpy as np

_DENIAL_DEVICES = [
    "merely", "simply", "lacking", "nothing more",
    "any real", "without", "lacks", "fails",
]


def load_stimuli(path: str | Path, schema_path: str | Path) -> list[dict]:
    """Read the stimulus file and refuse it unless every row obeys the rules in the schema file.

    LOGIC: rules live in schema.json, not here — hardcoding them means the two
    drift apart. Row numbers in errors count the header as row 1, matching
    what a spreadsheet shows.
    """
    schema = json.loads(Path(schema_path).read_text())
    validator = jsonschema.Draft202012Validator(schema)
    rows: list[dict] = []
    seen: dict[str, int] = {}
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            errs = sorted(validator.iter_errors(row), key=str)
            if errs:
                raise ValueError(f"row {i}: schema violation: {errs[0].message}")
            for k, v in row.items():
                if (k == "text" or k.endswith("_text")) and not (v or "").strip():
                    raise ValueError(f"row {i}: empty text field {k!r}")
            iid = row["item_id"]
            if iid in seen:
                raise ValueError(
                    f"row {i}: duplicate item_id {iid!r} (first seen at row {seen[iid]})"
                )
            seen[iid] = i
            rows.append(dict(row))
    return rows


def expand_pairs(items: list[dict]) -> list[dict]:
    """Split each affirm/deny pair row into the two separate sentences the model will actually see.

    item_id gains "__aff"/"__den"; every other column is carried through.
    Inputs are never mutated.
    """
    out: list[dict] = []
    for it in items:
        rest = {k: v for k, v in it.items() if k not in ("affirm_text", "deny_text")}
        for suffix, pol, src in (("__aff", "affirm", "affirm_text"),
                                 ("__den", "deny", "deny_text")):
            row = dict(rest)
            row["item_id"] = it["item_id"] + suffix
            row["text"] = it[src]
            row["polarity"] = pol
            out.append(row)
    return out


def find_claim_end(text: str, claim_phrase: str) -> int | None:
    """Point to the spot in the sentence just after the mind-claim ends, or None if it isn't there.

    Case-sensitive exact substring; first occurrence wins. LOGIC: capturing at
    claim-end means the same thing across templates, unlike the final token,
    which sits a template-dependent distance past the claim.
    """
    if not claim_phrase:
        return None
    idx = text.find(claim_phrase)
    return idx + len(claim_phrase) if idx != -1 else None


def annotate_claim_ends(
    items: list[dict], claim_lookup: dict[tuple[str, str], str]
) -> tuple[list[dict], int]:
    """Attach each sentence's claim-end position, and count how many could not be found.

    Missing offsets mean silent fallback to final-token capture, so each one
    warns and the count comes back alongside the rows — never swallowed.
    """
    out: list[dict] = []
    n_missing = 0
    for it in items:
        row = dict(it)
        phrase = claim_lookup.get((it.get("claim_id"), it.get("polarity")))
        end = find_claim_end(it["text"], phrase) if phrase else None
        if end is None:
            n_missing += 1
            warnings.warn(
                f"{it['item_id']}: claim phrase not found; item will fall back "
                f"to final-token capture",
                RuntimeWarning, stacklevel=2,
            )
        row["claim_end_char"] = end
        out.append(row)
    return out, n_missing


def _affirm_frac(rows: list[dict]) -> float:
    return float(np.mean([r.get("polarity") == "affirm" for r in rows])) if rows else float("nan")


def _by(items: list[dict], key: str) -> dict:
    groups: dict = {}
    for r in items:
        groups.setdefault(r.get(key), []).append(r)
    return groups


def _wc(text: str) -> int:
    return len(text.split())


def validate_balance(items: list[dict]) -> dict:
    """Measure everything about the stimulus set that could secretly drive the probe, and report it.

    Pure report — NEVER raises; the NET gate decides pass/fail. Keys are
    stable: the NET notebook and figures read them by name.
    """
    aff = [r for r in items if r.get("polarity") == "affirm"]
    den = [r for r in items if r.get("polarity") == "deny"]

    def gap(a: list[dict], d: list[dict]) -> float:
        if not a or not d:
            return float("nan")
        return float(np.mean([_wc(r["text"]) for r in a]) - np.mean([_wc(r["text"]) for r in d]))

    # Re-pair sentences by base item_id (the "__aff"/"__den" suffix convention).
    pairs: dict = {}
    for r in items:
        pairs.setdefault(r["item_id"].rsplit("__", 1)[0], {})[r.get("polarity")] = r
    complete = [p for p in pairs.values() if "affirm" in p and "deny" in p]
    gaps = [_wc(p["affirm"]["text"]) - _wc(p["deny"]["text"]) for p in complete]

    device_share = {
        dev: float(np.mean([dev in r["text"].lower() for r in den])) if den else float("nan")
        for dev in _DENIAL_DEVICES
    }
    text_counts = Counter(r["text"] for r in items)

    return {
        "polarity_balance": {
            "overall": _affirm_frac(items),
            "per_template": {t: _affirm_frac(g) for t, g in _by(items, "template_id").items()},
            "per_entity": {e: _affirm_frac(g) for e, g in _by(items, "entity").items()},
        },
        "length_gap_words": {
            "overall": gap(aff, den),
            "per_template": {
                t: gap([r for r in g if r.get("polarity") == "affirm"],
                       [r for r in g if r.get("polarity") == "deny"])
                for t, g in _by(items, "template_id").items()
            },
        },
        "length_within_3_frac": float(np.mean([abs(g) <= 3 for g in gaps])) if gaps else float("nan"),
        # near 0.5 = symmetric jitter; near 0 or 1 = the systematic confound
        "length_gap_signed_frac": float(np.mean([g < 0 for g in gaps])) if gaps else float("nan"),
        "not_fraction_deny": float(np.mean([" not " in r["text"].lower() for r in den])) if den else float("nan"),
        "denial_device_share": device_share,
        "top_denial_share": max(device_share.values()) if den else float("nan"),
        "duplicate_texts": sorted(t for t, c in text_counts.items() if c > 1),
        "claim_end_missing": int(sum(1 for r in items if r.get("claim_end_char") is None)),
        "n_per_set": dict(Counter(r.get("set") for r in items)),
        "n_per_template": dict(Counter(r.get("template_id") for r in items)),
        "n_per_entity": dict(Counter(r.get("entity") for r in items)),
    }


def check_twins(third: list[dict], first: list[dict]) -> dict:
    """Check that every first-person sentence has exactly one third-person twin, and vice versa.

    Join key is (claim_id, template_id, polarity) — twins differ only in
    referent. Orphans mean the mirror script dropped or duplicated something,
    and an unmatched item would make the H1 transfer test uninterpretable.
    """
    def key_counts(rows: list[dict]) -> Counter:
        return Counter((r.get("claim_id"), r.get("template_id"), r.get("polarity")) for r in rows)

    t, f = key_counts(third), key_counts(first)
    return {
        "matched_pairs": int(sum(min(t[k], f[k]) for k in t.keys() & f.keys())),
        "third_orphans": int(sum(v for k, v in t.items() if k not in f)),
        "first_orphans": int(sum(v for k, v in f.items() if k not in t)),
        "many_to_one": sorted(k for k in (t.keys() | f.keys()) if t.get(k, 0) > 1 or f.get(k, 0) > 1),
    }
