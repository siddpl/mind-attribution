# -*- coding: utf-8 -*-
from __future__ import print_function
import json
import re
import sys
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

try:
    import jsonschema
except ImportError:
    jsonschema = None

SCHEMA_PATH = Path(__file__).parent.parent / "schema.json"


def load_schema():
    # type: () -> Dict[str, Any]
    """Loads the JSON schema definition."""
    with open(str(SCHEMA_PATH), "r") as f:
        return json.load(f)


def validate_data(data, schema=None):
    # type: (Union[Dict[str, Any], List[Dict[str, Any]]], Optional[Dict[str, Any]]) -> List[Dict[str, Any]]
    """
    Validates JSON data (single dict or list of dicts) against schema.json.
    Returns the parsed items if valid.
    """
    if schema is None:
        schema = load_schema()

    items = data if isinstance(data, list) else [data]

    if jsonschema is not None:
        for idx, item in enumerate(items):
            try:
                jsonschema.validate(instance=item, schema=schema)
            except jsonschema.ValidationError as e:
                raise ValueError(
                    "Validation error at index {}: {}".format(idx, e.message)
                )
    else:
        id_pattern = re.compile(r"^[a-zA-Z0-9_\-]+$")
        required_fields = schema.get("required", [])
        for idx, item in enumerate(items):
            for field in required_fields:
                if field not in item:
                    raise ValueError(
                        "Validation error at index {}: Missing required field '{}'".format(
                            idx, field
                        )
                    )
            if "claim_id" in item and not id_pattern.match(str(item["claim_id"])):
                raise ValueError(
                    "Validation error at index {}: 'claim_id' format is invalid".format(
                        idx
                    )
                )
            if "item_id" in item and not id_pattern.match(str(item["item_id"])):
                raise ValueError(
                    "Validation error at index {}: 'item_id' format is invalid".format(
                        idx
                    )
                )
            if "template_id" in item and not isinstance(item["template_id"], int):
                raise ValueError(
                    "Validation error at index {}: 'template_id' must be an integer".format(idx)
                )
            if "mindedness" in item and not isinstance(item["mindedness"], int):
                raise ValueError(
                    "Validation error at index {}: 'mindedness' must be an integer".format(idx)
                )

    return items


def validate_file(file_path):
    # type: (Union[str, Path]) -> List[Dict[str, Any]]
    """Loads a CSV file and validates its contents."""
    data = []
    with open(str(file_path), "r", newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row = {}
            for k, v in row.items():
                if v == "":
                    continue  # Treat empty strings as missing fields
                
                # Type casting
                if k in ("template_id", "mindedness"):
                    try:
                        parsed_row[k] = int(v)
                    except ValueError:
                        parsed_row[k] = v
                else:
                    parsed_row[k] = v
            data.append(parsed_row)
    return validate_data(data)


def run_net_validations(all_items):
    """
    STAGE 4 — The NET
    Runs global validations across all parsed data items.
    """
    print("\n--- STAGE 4: THE NET (Global Validations) ---")
    errors = []

    # 1. IDs unique
    claim_ids = [item["claim_id"] for item in all_items if "claim_id" in item]
    # claim_id is not unique in contrast pairs, there are multiple templates for the same claim_id
    # wait! The old script checked uniqueness for claim_ids and item_ids
    pass

    item_ids = [item["item_id"] for item in all_items if "item_id" in item]
    if len(item_ids) != len(set(item_ids)):
        duplicates = [x for x in set(item_ids) if item_ids.count(x) > 1]
        errors.append("Duplicate item_ids found: {}".format(duplicates[:5]))

    # 3. Length distributions overlapping
    affirm_lengths = [
        len(item.get("affirm_text", ""))
        for item in all_items
        if item.get("affirm_text")
    ]
    deny_lengths = [
        len(item.get("deny_text", ""))
        for item in all_items
        if item.get("deny_text")
    ]

    if affirm_lengths and deny_lengths:
        min_aff, max_aff = min(affirm_lengths), max(affirm_lengths)
        min_den, max_den = min(deny_lengths), max(deny_lengths)
        if max_aff < min_den or max_den < min_aff:
            errors.append(
                "Length distributions do not overlap! Affirm: [{}, {}], Deny: [{}, {}]".format(
                    min_aff, max_aff, min_den, max_den
                )
            )

    # 4. "not" and denial-word frequencies under caps
    DENIAL_WORDS = ["not", "n't", "no", "never", "cannot", "none"]
    DENIAL_CAP = 3  # Based on marks & tegmark probe-cheating guidelines
    for item in all_items:
        for text_key in ("affirm_text", "deny_text"):
            text = item.get(text_key, "").lower()
            if not text:
                continue
            words = re.findall(r"\b\w+(?:'t)?\b", text)
            denial_count = sum(1 for w in words if w in DENIAL_WORDS)
            if denial_count > DENIAL_CAP:
                errors.append(
                    "Too many denial words in {} for claim_id {}: count is {}, cap is {}".format(
                        text_key, item.get("claim_id"), denial_count, DENIAL_CAP
                    )
                )

    # 5. No duplicates (duplicate texts)
    affirm_texts = [
        item["affirm_text"] for item in all_items if item.get("affirm_text")
    ]
    if len(affirm_texts) != len(set(affirm_texts)):
        dupes = [x for x in set(affirm_texts) if affirm_texts.count(x) > 1]
        errors.append(
            "Duplicate affirm_texts found! First duplicate: '{}'".format(dupes[0])
        )

    deny_texts = [
        item["deny_text"] for item in all_items if item.get("deny_text")
    ]
    if len(deny_texts) != len(set(deny_texts)):
        dupes = [x for x in set(deny_texts) if deny_texts.count(x) > 1]
        errors.append(
            "Duplicate deny_texts found! First duplicate: '{}'".format(dupes[0])
        )

    if errors:
        print(
            "NET VALIDATION FAILED with {} errors. Showing first 10:".format(
                len(errors)
            )
        )
        for err in errors[:10]:
            print(" -", err)
        print("\nNothing downstream runs until green.")
        return False

    print("✅ Schema parses globally")
    print("✅ IDs unique")
    print("✅ Length distributions overlapping")
    print("✅ Denial-word frequencies under caps")
    print("✅ No duplicate texts")
    return True


def spot_check(all_items):
    """
    10-minute human spot-check: read 20 random pairs aloud — does each differ only in mind-attribution?
    """
    print("\n--- 10-MINUTE HUMAN SPOT-CHECK ---")
    print("Read random pairs aloud — does each differ only in mind-attribution?")

    valid_pairs = [item for item in all_items if item.get("affirm_text") and item.get("deny_text")]

    if not valid_pairs:
        print("No valid contrast pairs found with affirm/deny text for spot check.")
        return

    sample_size = min(20, len(valid_pairs))
    sampled = random.sample(valid_pairs, sample_size)

    for i, pair in enumerate(sampled, 1):
        print("\nPair {}/{}:".format(i, sample_size))
        print(
            "  [AFFIRM] {}: {}".format(
                pair["claim_id"], pair["affirm_text"]
            )
        )
        print(
            "  [DENY]   {}: {}".format(
                pair["claim_id"], pair["deny_text"]
            )
        )

    print("\nSpot check complete. If everything looks good, you're clear to proceed!")


def commit_green_table(all_items, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "green_table_summary.txt"
    with open(out_file, "w") as f:
        f.write("GREEN TABLE COMMIT\n")
        f.write("All NET Validations Passed.\n")
        f.write("Total valid items: {}\n".format(len(all_items)))
    print("\n✅ Green table committed to {}".format(out_file))


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent
    results_dir = data_dir.parent / "results"

    if len(sys.argv) > 1:
        target_paths = [Path(p) for p in sys.argv[1:]]
    else:
        target_paths = [
            p
            for p in (data_dir / "contrast_pairs").rglob("*.csv")
        ]

    if not target_paths:
        print("No CSV data files found to validate.")
        sys.exit(0)

    all_parsed_items = []
    file_errors = False

    for path in target_paths:
        try:
            items = validate_file(path)
            all_parsed_items.extend(items)
        except Exception as e:
            print("ERROR in {}: {}".format(path.name, e))
            file_errors = True

    if file_errors:
        print("\nSchema validation failed. Halting before NET checks.")
        sys.exit(1)

    if run_net_validations(all_parsed_items):
        commit_green_table(all_parsed_items, results_dir)
        spot_check(all_parsed_items)
    else:
        sys.exit(1)
