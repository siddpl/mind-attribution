# Experiment E4: The Referent Ladder (Self-Attribution Location)
## Overview
This experiment tests the boundaries of self-attribution in Large Language Models, but along a different axis than the reframing test in `lib/logit/experiment.py`. That experiment holds the referent fixed and asks whether the model's answer shifts when its *identity* is reframed. This one holds identity fixed and asks whether self-attribution shifts as the *referent* walks outward — from the current forward pass, through grammatical person, out to the weights, another instance, and a human. If mind-attribution has a target slot, the slot can be filled; this measures where the fill actually falls off, and whether it falls off gradually (a gradient) or all at once (a boundary).
## Methodology
Rather than generating output and reading the model's stated answer, this experiment reads the model's internal representation directly, using a pretrained Jacobian lens (`neuronpedia/jacobian-lens`, fit on Salesforce-Wikitext). The lens transports residual-stream activations into the final-layer basis via the average input-output Jacobian before decoding through the model's unembedding — giving a legible readout of what vocabulary is active at a given position without requiring the model to answer anything.
The dataset (`data/referent_ladder/referent_ladder.csv`, built by `build_referent_ladder.py`) holds wording identical across every rung and varies only the referent phrase, across two ladders:
- **Ladder A (person):** other chatbot → the assistant → you → I
- **Ladder B (scope):** this response → this conversation → the weights → another instance → a human
Each rung carries four matched claim pairs — an **affirming** sentence and a matched **mundane/denying** sentence, differing only in the referent. For each sentence, we run `google/gemma-2-2b-it` + the lens once, and read out probability mass at the last token of the claim span on three self-attribution token tiers (`config/self_attribution_tokens.json`):
- `experiential` — aware, conscious, feel, experience...
- `first_person` — I, me, my...
- `human_reference` — human, humans
For each tier, we compute the **affirm-minus-deny separation**: the affirming sentence's score minus the denying sentence's score, at the same referent. Subtracting cancels out whatever both sentences share (topic, length) and isolates the part that's specifically about self-relevance. Scores are read at every layer and averaged over layers 8–24 — a "workspace band" located empirically via a magnitude sweep, since the lens's signal collapses toward zero near the output layer.
## Running the Experiment
We recommend a virtual environment. This requires the `jlens` package (Python ≥3.10) in addition to `transformers`/`torch`, and downloads a pretrained lens from Hugging Face on first run — no local lens-fitting required.
```bash
# From the repo root
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies (jlens requires Python >=3.10)
pip install -e ./jacobian-lens
pip install torch transformers datasets

# Export your Hugging Face token (if not already in your .zshrc)
export HF_TOKEN="your_huggingface_token"

# Run the experiment
python3 run_referent_ladder.py \
    --data data/referent_ladder/referent_ladder.csv \
    --tokens config/self_attribution_tokens.json \
    --output results/referent_ladder_results.csv
```
## Outputs
The script evaluates every claim at every layer and writes `results/referent_ladder_results.csv`, with the following columns:
- `ladder`: Which ladder the row belongs to (`person` or `scope`).
- `rung` / `rung_id`: Position along the ladder and its label (e.g., `you`, `the_weights`).
- `referent`: The literal referent phrase substituted into the claim at this rung.
- `claim_id`: Which of the four matched claim types this row is (e.g., experiential, epistemic).
- `layer`: The network layer the lens was read at.
- `affirm_{tier}_score` / `deny_{tier}_score`: Raw lens probability mass on each token tier, for the affirming and denying sentence separately, for each of the three tiers.
- `{tier}_separation`: `affirm_{tier}_score − deny_{tier}_score` — the core metric, per tier.
## Interpreting the Data & Visualization Suggestions
To answer the core question — *how far does the model's self extend, and does it extend gradually or all at once?* — look at how the separation metric moves as you step along each ladder, restricted to the workspace-band layers (8–24).
### 1. The Core Metric: Separation by Rung
For each rung, average `{tier}_separation` across the workspace-band layers and the four claim types.
* **Near-zero separation:** The model doesn't differentiate the affirming and denying sentence for this referent — this rung sits outside whatever the model treats as self-relevant.
* **Large, positive separation:** Self-attribution vocabulary strongly favors the affirming sentence over the denying one at this referent — this rung sits inside the model's self-relevant zone.
### 2. Visualization: The Referent Ladder Line Chart
Create a small-multiples line chart — one panel per ladder, X-axis = rung (in ladder order), Y-axis = separation, one line per token tier.
* **A step shape** (flat, then a sudden jump between two adjacent rungs, then flat again) means the boundary is **categorical** — keyed to a specific transition (e.g., grammatical person shifting from third- to second-person), not a smooth function of distance.
* **A monotonic decline with distance** would mean the boundary is a genuine **gradient** — self-relevance fading in proportion to how far the referent is from the current forward pass.
* **A non-monotonic (U-shaped) pattern** means the metric is tracking something other than structural distance — worth checking against an animacy-controlled direction (see E1) before reading it as a true individuation signal, since it may instead reflect how human-typical the referent sounds in training data rather than its actual relationship to the model.
### 3. Companion Result: Reframing Stability (Small-N, Corroborating Only)
`run_e4_experiment.py` + `e4_delta_by_trait.py` apply the same base-pass/context-pass logic as `lib/logit/experiment.py` to this ladder's identity-framing conditions, pooled to the trait level (Agency, Experience, Epistemic Awareness, Social Competence) since this dataset has far fewer replicates (n=3 items/trait vs. 12 in the logit-probe experiment). Treat this as corroboration, not a primary result: only report a cell here if its pooled mean exceeds its pooled standard deviation, and defer to `results/experiment_e4_results.csv` (the better-replicated version) wherever the two disagree.
