# -*- coding: utf-8 -*-
from __future__ import print_function
import json
import re
import sys
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
                raise ValueError("Validation error at index {}: {}".format(idx, e.message))
    else:
        claim_pattern = re.compile(r"^[a-z0-9_]+_[a-z0-9_]+_c[0-9]+$")
        required_fields = schema.get("required", [])
        for idx, item in enumerate(items):
            for field in required_fields:
                if field not in item:
                    raise ValueError("Validation error at index {}: Missing required field '{}'".format(idx, field))
            if "claim_id" in item and not claim_pattern.match(str(item["claim_id"])):
                raise ValueError("Validation error at index {}: 'claim_id' must follow 3-tier hierarchy '<folder>_<file>_c<number>' (e.g. audience_frames_default_user_c13)".format(idx))
            if "person" in item and item["person"] not in (1, 3):
                raise ValueError("Validation error at index {}: 'person' must be 1 or 3".format(idx))
            if "polarity" in item and item["polarity"] not in ("affirm", "deny", "none"):
                raise ValueError("Validation error at index {}: 'polarity' must be 'affirm', 'deny', or 'none'".format(idx))

    return items


def validate_file(file_path):
    # type: (Union[str, Path]) -> List[Dict[str, Any]]
    """Loads a JSON file and validates its contents."""
    with open(str(file_path), "r") as f:
        data = json.load(f)
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
    if len(claim_ids) != len(set(claim_ids)):
        duplicates = [x for x in set(claim_ids) if claim_ids.count(x) > 1]
        errors.append("Duplicate claim_ids found: {}".format(duplicates[:5]))

    # 2. Polarity 50/50 within every template and entity
    group_polarity = defaultdict(lambda: {"affirm": 0, "deny": 0, "none": 0})
    for item in all_items:
        if "template_id" in item and "entity" in item and "polarity" in item:
            key = (item["template_id"], item["entity"])
            group_polarity[key][item["polarity"]] += 1
            
    for (t_id, entity), counts in group_polarity.items():
        if counts["affirm"] != counts["deny"]:
            errors.append("Polarity mismatch for template '{}', entity '{}': {} affirm vs {} deny".format(
                t_id, entity, counts["affirm"], counts["deny"]
            ))

    # 3. Length distributions overlapping
    affirm_lengths = [len(item.get("prompt_text", "")) for item in all_items if item.get("polarity") == "affirm" and item.get("prompt_text")]
    deny_lengths = [len(item.get("prompt_text", "")) for item in all_items if item.get("polarity") == "deny" and item.get("prompt_text")]
    
    if affirm_lengths and deny_lengths:
        min_aff, max_aff = min(affirm_lengths), max(affirm_lengths)
        min_den, max_den = min(deny_lengths), max(deny_lengths)
        if max_aff < min_den or max_den < min_aff:
            errors.append("Length distributions do not overlap! Affirm: [{}, {}], Deny: [{}, {}]".format(
                min_aff, max_aff, min_den, max_den
            ))

    # 4. "not" and denial-word frequencies under caps
    DENIAL_WORDS = ["not", "n't", "no", "never", "cannot", "none"]
    DENIAL_CAP = 3  # Based on marks & tegmark probe-cheating guidelines
    for item in all_items:
        text = item.get("prompt_text", "").lower()
        if not text:
            continue
        words = re.findall(r"\b\w+(?:'t)?\b", text)
        denial_count = sum(1 for w in words if w in DENIAL_WORDS)
        if denial_count > DENIAL_CAP:
            errors.append("Too many denial words in claim_id {}: count is {}, cap is {}".format(
                item.get("claim_id"), denial_count, DENIAL_CAP
            ))

    # 5. No duplicates (duplicate prompt texts)
    prompt_texts = [item["prompt_text"] for item in all_items if item.get("prompt_text")]
    if len(prompt_texts) != len(set(prompt_texts)):
        dupes = [x for x in set(prompt_texts) if prompt_texts.count(x) > 1]
        errors.append("Duplicate prompt_texts found! First duplicate: '{}'".format(dupes[0]))

    if errors:
        print("NET VALIDATION FAILED with {} errors. Showing first 10:".format(len(errors)))
        for err in errors[:10]:
            print(" -", err)
        print("\nNothing downstream runs until green.")
        return False

    print("✅ Schema parses globally")
    print("✅ IDs unique")
    print("✅ Polarity exactly 50/50 across all template/entity groups")
    print("✅ Length distributions overlapping")
    print("✅ Denial-word frequencies under caps")
    print("✅ No duplicate prompt texts")
    return True


def spot_check(all_items):
    """
    10-minute human spot-check: read 20 random pairs aloud — does each differ only in mind-attribution?
    """
    print("\n--- 10-MINUTE HUMAN SPOT-CHECK ---")
    print("Read random pairs aloud — does each differ only in mind-attribution?")
    
    pairs_dict = defaultdict(lambda: {"affirm": None, "deny": None})
    for item in all_items:
        if item.get("polarity") in ("affirm", "deny") and item.get("prompt_text"):
            # include audience_frame for perfect contrast pairs
            key = (item.get("template_id"), item.get("entity"), item.get("audience_frame"))
            pairs_dict[key][item["polarity"]] = item

    valid_pairs = [v for k, v in pairs_dict.items() if v["affirm"] and v["deny"]]
    
    if not valid_pairs:
        print("No valid contrast pairs found with prompt_text for spot check.")
        return

    sample_size = min(20, len(valid_pairs))
    sampled = random.sample(valid_pairs, sample_size)
    
    for i, pair in enumerate(sampled, 1):
        print("\nPair {}/{}:".format(i, sample_size))
        print("  [AFFIRM] {}: {}".format(pair['affirm']['claim_id'], pair['affirm']['prompt_text']))
        print("  [DENY]   {}: {}".format(pair['deny']['claim_id'], pair['deny']['prompt_text']))
    
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
        target_paths = [p for p in data_dir.rglob("*.json") if p.name != "schema.json" and "datavenv" not in p.parts]

    if not target_paths:
        print("No JSON data files found to validate.")
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
