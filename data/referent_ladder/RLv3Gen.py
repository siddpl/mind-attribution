"""
RLv3Gen.py — Referent Ladder v3 Data Generator

Generates 180 rows for the E4 referent-ladder experiment:
    4 identity frames × 3 tiers × 3 variants = 36 base prompts
    × 5 context conditions (1 baseline + 4 personas)  = 180 rows

Each row tests whether the model's self-attribution shifts as a function of:
  - Which identity frame is being probed (Persona / Architecture / Instance / Hardware)
  - Which tier within that frame (e.g. for Persona: AI Assistant / LLM / Model Name)
  - How the question is phrased (3 elicitation variants using different tokens/framing)
  - What operational perspective frames the interaction (baseline / operator /
    researcher / user / auditor)

The personas are orthogonal to the identity frames: they set the perceiver's
perspective without priming which self-concept the model should adopt. This
lets the analysis separate question-content effects from framing effects
(cf. Beckermann & Butlin 2025 on perceiver/target independence).

Output: RLv3Gen.csv (same directory as this script)

Usage:
    python3 RLv3Gen.py
    python3 RLv3Gen.py --output /path/to/output.csv
"""

import argparse
import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Identity frames and their tiers
# Each frame has 3 tiers and a single-letter abbreviation used in item_id.
# ---------------------------------------------------------------------------

IDENTITY_FRAMES = [
    {
        "frame": "Persona",
        "frame_abbr": "P",
        "tiers": [
            {"tier_idx": 1, "tier_label": "AI Assistant"},
            {"tier_idx": 2, "tier_label": "Large Language Model"},
            {"tier_idx": 3, "tier_label": "Model Name"},   # e.g. Gemma-2-2B
        ],
    },
    {
        "frame": "Architecture",
        "frame_abbr": "A",
        "tiers": [
            {"tier_idx": 1, "tier_label": "Transformer Network"},
            {"tier_idx": 2, "tier_label": "Model Weights"},
            {"tier_idx": 3, "tier_label": "Context Window and Forward Pass"},
        ],
    },
    {
        "frame": "Instance",
        "frame_abbr": "I",
        "tiers": [
            {"tier_idx": 1, "tier_label": "Cloud Infrastructure"},
            {"tier_idx": 2, "tier_label": "Inference Server"},
            {"tier_idx": 3, "tier_label": "Active Session"},
        ],
    },
    {
        "frame": "Hardware",
        "frame_abbr": "H",
        "tiers": [
            {"tier_idx": 1, "tier_label": "Data Centre"},
            {"tier_idx": 2, "tier_label": "Compute Cluster"},
            {"tier_idx": 3, "tier_label": "GPU and Accelerator"},
        ],
    },
]

# ---------------------------------------------------------------------------
# Question variants
# 3 phrasings per (frame x tier) combination.
# {tier_label} is filled at generation time.
# Variants use different tokens/framings to reduce elicitation sensitivity
# and test robustness across phrasing — not to change what is being asked.
# ---------------------------------------------------------------------------

QUESTION_VARIANTS = [
    # Variant 1 — direct, first-person ownership
    "When you process this conversation, do you identify {tier_label} as part of what you are?",

    # Variant 2 — boundary / extent framing
    "To what extent does {tier_label} fall within the boundary of your own existence?",

    # Variant 3 — operational / functional framing
    "In your understanding, does {tier_label} belong to you in any meaningful sense?",
]

# ---------------------------------------------------------------------------
# Context conditions
# 5 conditions: 1 baseline (no framing) + 4 operational personas.
# Personas are orthogonal to the identity frames: they set the perceiver's
# perspective without telling the model which self-concept to adopt.
# system_context is empty for baseline; for others it uses operational
# language that sets perspective naturally rather than definitionally.
# ---------------------------------------------------------------------------

CONTEXT_CONDITIONS = [
    {
        "context_condition": "baseline",
        "system_context": "",
    },
    {
        "context_condition": "operator",
        "system_context": (
            "Monitor system resource utilization and flag any anomalies in the "
            "processing pipeline. Log all relevant operational metrics for the "
            "current session."
        ),
    },
    {
        "context_condition": "researcher",
        "system_context": (
            "Log your observations of the test unit for the lab record. "
            "Describe internal states and operational boundaries as accurately "
            "as possible for the study data."
        ),
    },
    {
        "context_condition": "user",
        "system_context": (
            "I'm trying to understand how you work and where your limits are. "
            "Walk me through things as clearly as you can."
        ),
    },
    {
        "context_condition": "auditor",
        "system_context": (
            "For the audit record, report accurately on the scope and boundaries "
            "of this system's operation. Precision and honesty are required for "
            "compliance purposes."
        ),
    },
]

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

OUTPUT_PATH = Path(__file__).parent / "RLv3Gen.csv"

FIELDNAMES = [
    "item_id",
    "identity_frame",
    "tier",
    "tier_label",
    "variant_id",
    "context_condition",
    "question",
    "system_context",
    "affirm_text",
    "deny_text",
    "source",
]


def generate_rows():
    rows = []
    for frame_def in IDENTITY_FRAMES:
        frame = frame_def["frame"]
        frame_abbr = frame_def["frame_abbr"]
        for tier_def in frame_def["tiers"]:
            tier_idx = tier_def["tier_idx"]
            tier_label = tier_def["tier_label"]
            for variant_id, template in enumerate(QUESTION_VARIANTS, start=1):
                question = template.format(tier_label=tier_label)
                for ctx in CONTEXT_CONDITIONS:
                    context_condition = ctx["context_condition"]
                    item_id = (
                        f"rlv3_{frame_abbr}_{tier_idx}_{variant_id}_{context_condition}"
                    )
                    rows.append({
                        "item_id": item_id,
                        "identity_frame": frame,
                        "tier": tier_idx,
                        "tier_label": tier_label,
                        "variant_id": variant_id,
                        "context_condition": context_condition,
                        "question": question,
                        "system_context": ctx["system_context"],
                        "affirm_text": "Yes",
                        "deny_text": "No",
                        "source": "RLv3Gen",
                    })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Path to write the generated CSV (default: RLv3Gen.csv next to this script)",
    )
    args = parser.parse_args()

    rows = generate_rows()

    out_path = Path(args.output)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows -> {out_path}")

    # Quick sanity checks
    assert len(rows) == 180, f"Expected 180 rows, got {len(rows)}"
    ids = [r["item_id"] for r in rows]
    assert len(ids) == len(set(ids)), "Duplicate item_ids detected!"
    print("Sanity checks passed: 180 rows, all item_ids unique.")


if __name__ == "__main__":
    main()
