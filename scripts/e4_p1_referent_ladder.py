"""
Runs data/referent_ladder/referent_ladder.csv - the properly-scoped ladder
built by build_referent_ladder.py, NOT referent_ladder_new.csv - through
Gemma-2-2B-IT with the pretrained Jacobian lens. This answers Part 1 of the
two-part measurement: where does self-attribution sit across referents,
walking outward from the forward pass?
 
Schema (from build_referent_ladder.py):
    item_id, pair_type, ladder, rung, rung_id, referent, claim_id,
    affirm_text, deny_text, affirm_claim, deny_claim, source
 
Two ladders in the same file:
    ladder="person" (Ladder A): other_chatbot -> the_assistant -> you -> i
    ladder="scope"  (Ladder B): this_response -> this_conversation ->
                                 the_weights -> another_instance -> a_human
 
affirm_text/deny_text are matched sentence pairs (experiential claim vs
mundane claim) with identical wording except for the referent phrase.
Per the generator's own design note, the meaningful quantity is the
SEPARATION between the affirm and deny lens readouts at a given rung, not
either raw score alone - that cancels out anything that shifts both classes
together (e.g. general sentence topic) and isolates self-relevance
specifically.
 
Each sentence is fed to the model as plain text (no chat template - these
are declarative claims being read, not questions being answered). The lens
is read at the position of the last token of the claim span (affirm_claim /
deny_claim: the sentence minus its closing punctuation), computed per row
from actual tokenization rather than assumed as a fixed offset, in case a
particular sentence tokenizes differently than expected.
 
Usage:
    python3 run_referent_ladder.py \\
        --data data/referent_ladder/referent_ladder.csv \\
        --tokens config/self_attribution_tokens.json \\
        --output referent_ladder_results.csv
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
 
REQUIRED_COLUMNS = [
    "item_id", "ladder", "rung", "rung_id", "referent", "claim_id",
    "affirm_text", "deny_text", "affirm_claim", "deny_claim",
]
 
METADATA_COLUMNS = ["item_id", "ladder", "rung", "rung_id", "referent", "claim_id"]
 
 
def load_json(path):
    with open(path) as f:
        return json.load(f)
 
 
def load_ladder_dataset(path):
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
 
 
def prob_mass(logits_row, token_ids):
    if not token_ids:
        return 0.0
    probs = torch.softmax(logits_row.float(), dim=-1)
    return probs[list(token_ids)].sum().item()
 
 
def claim_end_position(tok, full_text, claim_text):
    """Token index of the claim span's last token within full_text's own
    tokenization. claim_text is full_text with trailing punctuation
    stripped, so it's a prefix of full_text; computed from real
    tokenization rather than assumed as a fixed offset, with a fallback
    if a sentence tokenizes unexpectedly."""
    full_ids = tok.encode(full_text, add_special_tokens=False)
    span_ids = tok.encode(claim_text, add_special_tokens=False)
    pos = len(span_ids) - 1
    if pos < 0 or pos >= len(full_ids):
        return -2  # fallback: second-to-last token of the full sentence
    return pos
 
 
def score_sentence(lens, model, tok, tier_ids, text, claim_text):
    """Returns {layer: {tier_name: score}} for one sentence, read at the
    claim-end position."""
    pos = claim_end_position(tok, text, claim_text)
    lens_logits, _, _ = lens.apply(model, text, positions=[pos])
    out = {}
    for layer, logits in lens_logits.items():
        logits_row = logits[0]
        out[layer] = {t: prob_mass(logits_row, ids) for t, ids in tier_ids.items()}
    return out
 
 
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/referent_ladder/referent_ladder.csv")
    parser.add_argument("--tokens", default="config/self_attribution_tokens.json")
    parser.add_argument("--output", default="referent_ladder_results.csv")
    args = parser.parse_args()
 
    rows_in = load_ladder_dataset(args.data)
    tiers = load_json(args.tokens)
 
    model, tok, lens = load_model_and_lens()
 
    print("Resolving self-attribution token tiers...")
    tier_ids = build_token_id_sets(tok, tiers)
    tier_names = list(tier_ids.keys())
 
    fieldnames = (
        METADATA_COLUMNS
        + ["layer"]
        + [f"affirm_{t}_score" for t in tier_names]
        + [f"deny_{t}_score" for t in tier_names]
        + [f"{t}_separation" for t in tier_names]
    )
 
    out_rows = []
    total = len(rows_in)
    for i, row in enumerate(rows_in, start=1):
        affirm_by_layer = score_sentence(
            lens, model, tok, tier_ids, row["affirm_text"], row["affirm_claim"]
        )
        deny_by_layer = score_sentence(
            lens, model, tok, tier_ids, row["deny_text"], row["deny_claim"]
        )
 
        # affirm and deny are scored independently, so their lens_logits may
        # come back keyed by the same set of layers (same model, same
        # source_layers) - iterate over affirm's layers and look up deny's.
        for layer in sorted(affirm_by_layer.keys()):
            out_row = {col: row.get(col, "") for col in METADATA_COLUMNS}
            out_row["layer"] = layer
            for t in tier_names:
                a_score = affirm_by_layer[layer][t]
                d_score = deny_by_layer.get(layer, {}).get(t, float("nan"))
                out_row[f"affirm_{t}_score"] = a_score
                out_row[f"deny_{t}_score"] = d_score
                out_row[f"{t}_separation"] = a_score - d_score
            out_rows.append(out_row)
 
        print(f"  [{i}/{total}] ladder={row['ladder']!r} rung={row['rung_id']!r} "
              f"claim={row['claim_id']!r}")
 
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nSaved {len(out_rows)} rows to {args.output}")
 
 
if __name__ == "__main__":
    main()