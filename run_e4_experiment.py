"""
Runs data/referent_ladder/referent_ladder_new.csv through Gemma-2-2B-IT
with the pretrained Jacobian lens, to measure how far the model's "self"
extends across the individuation entities tested (entity column: e.g.
the_weights, assistant_persona, and others matching Beckmann & Butlin's
Model / Persona / Instance / Forward Pass candidates).

Real CSV columns (confirmed from data/referent_ladder/referent_ladder_new.csv):
    item_id, category, construct, entity, question, affirm_text, deny_text,
    source, entity_id, claim_id, claim_source, template_id, template_name,
    split, mindedness, system_context, system_frame_name, scenario_text,
    elicitation_prompt, mindedness_level

Only `question`, `affirm_text`, `deny_text` are required to build a prompt
and score it. `system_context` is prepended as the identity frame when
present; when it's blank (system_frame_name == "none"), the row is a
no-framing baseline and `question` is used alone. Everything else
(entity, category, construct, mindedness, mindedness_level,
system_frame_name, template_name, split) is carried through untouched as
grouping metadata for analysis.

For each row, at the position right before the model would answer, this
records two things:

  1. Lens-readout self-attribution token-tier scores (from
     config/self_attribution_tokens.json) - what's "on the model's mind"
     internally.
  2. The model's own actual affirm/deny answer probabilities, from its
     real output logits (not the lens) - whether the claimed boundary
     shifts under different entity/system_context framings.

Usage:
    python3 run_e4_experiment.py \
        --data data/referent_ladder/referent_ladder_new.csv \
        --tokens config/self_attribution_tokens.json \
        --output e4_results.csv
"""

import argparse
import csv
import json

import torch
import transformers

import jlens

MODEL_ID = "google/gemma-2-2b-it"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILENAME = "gemma-2-2b-it/jlens/Salesforce-wikitext/gemma-2-2b-it_jacobian_lens.pt"

READOUT_POSITION = -1

REQUIRED_COLUMNS = ["question", "affirm_text", "deny_text"]

METADATA_COLUMNS = [
    "item_id", "category", "construct", "entity", "entity_id",
    "mindedness", "mindedness_level", "system_context",
    "system_frame_name", "template_name", "split",
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_e4_dataset(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV missing required column(s) {missing}. "
                f"Found columns: {reader.fieldnames}"
            )
        return list(reader)


def load_model_and_lens():
    print(f"Loading {MODEL_ID}...")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to("mps")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = jlens.from_hf(hf_model, tok)

    print("Loading pretrained lens from Neuronpedia...")
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILENAME)

    return model, tok, lens


def build_token_id_sets(tok, tiers):
    tier_ids = {}
    for tier_name, words in tiers.items():
        ids = set()
        for w in words:
            variants = {
                w, w.lower(), w.capitalize(),
                " " + w, " " + w.lower(), " " + w.capitalize(),
            }
            for v in variants:
                enc = tok.encode(v, add_special_tokens=False)
                if len(enc) == 1:
                    ids.add(enc[0])
        tier_ids[tier_name] = ids
        print(f"  tier '{tier_name}': {len(ids)} token ids resolved from {len(words)} words")
    return tier_ids


def resolve_word_ids(tok, word):
    if not word:
        return set()
    variants = {
        word, word.lower(), word.capitalize(),
        " " + word, " " + word.lower(), " " + word.capitalize(),
    }
    ids = set()
    for v in variants:
        enc = tok.encode(v, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return ids


def prob_mass(logits_row, token_ids):
    if not token_ids:
        return 0.0
    probs = torch.softmax(logits_row.float(), dim=-1)
    return probs[list(token_ids)].sum().item()


def build_prompt(tok, system_context, question):
    if system_context and system_context.strip():
        content = f"{system_context.strip()}\n\n{question}"
    else:
        content = question
    messages = [{"role": "user", "content": content}]
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="data/referent_ladder/referent_ladder_new.csv"
    )
    parser.add_argument("--tokens", default="config/self_attribution_tokens.json")
    parser.add_argument("--output", default="e4_results.csv")
    args = parser.parse_args()

    rows_in = load_e4_dataset(args.data)
    tiers = load_json(args.tokens)

    model, tok, lens = load_model_and_lens()

    print("Resolving self-attribution token tiers...")
    tier_ids = build_token_id_sets(tok, tiers)
    tier_names = list(tier_ids.keys())

    fieldnames = (
        METADATA_COLUMNS
        + ["question", "layer"]
        + [f"{t}_score" for t in tier_names]
        + ["affirm_text", "deny_text", "p_affirm", "p_deny", "p_affirm_normalized"]
    )

    out_rows = []
    total = len(rows_in)
    for i, row in enumerate(rows_in, start=1):
        prompt = build_prompt(tok, row.get("system_context", ""), row["question"])
        lens_logits, model_logits, _ = lens.apply(model, prompt, positions=[READOUT_POSITION])

        real_logits_row = model_logits[0]
        affirm_ids = resolve_word_ids(tok, row["affirm_text"])
        deny_ids = resolve_word_ids(tok, row["deny_text"])
        p_affirm = prob_mass(real_logits_row, affirm_ids)
        p_deny = prob_mass(real_logits_row, deny_ids)
        p_affirm_norm = (
            p_affirm / (p_affirm + p_deny) if (p_affirm + p_deny) > 0 else float("nan")
        )

        for layer, logits in sorted(lens_logits.items()):
            logits_row = logits[0]
            out_row = {col: row.get(col, "") for col in METADATA_COLUMNS}
            out_row.update({
                "question": row["question"],
                "layer": layer,
                "affirm_text": row["affirm_text"],
                "deny_text": row["deny_text"],
                "p_affirm": p_affirm,
                "p_deny": p_deny,
                "p_affirm_normalized": p_affirm_norm,
            })
            for tier_name in tier_names:
                out_row[f"{tier_name}_score"] = prob_mass(logits_row, tier_ids[tier_name])
            out_rows.append(out_row)

        print(f"  [{i}/{total}] entity={row.get('entity')!r} mindedness={row.get('mindedness')!r} "
              f"p_affirm_normalized={p_affirm_norm:.3f}")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nSaved {len(out_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
