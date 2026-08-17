# Experiment E4: The Referent Gradient (Individuation)

## Overview
This experiment tests the boundaries of mind-attribution and self-attribution in Large Language Models. By keeping the `question` (scenario + elicitation) perfectly stagnant while altering the `system_context` (the identity frame), we measure precisely whether the AI's claimed boundaries shift. A boundary that moves under reframing provides evidence that the "unit" is a constructed persona produced by the model rather than a natural kind it tracks internally.

## Methodology
Instead of generating full text outputs and parsing them, this experiment relies on internal mechanism extraction. We compute the final layer logits for the very next predicted token using the `google/gemma-2b-it` model. 

For each row in our dataset (`data/referent_ladder/referent_ladder_new.csv`), we perform two separate forward passes:
1. **Base Pass:** Only the `question` is formatted via the model's chat template.
2. **Context Pass:** The `system_context` is prepended to the `question` to establish the identity frame.

We extract the exact token IDs for the words `"Yes"` and `"No"`. We then apply temperature scaling ($T=0.7$) to just these two specific logits, and compute the comparative softmax probability of the model answering "Yes" versus "No".

## Running the Experiment
To run the script, we recommend setting up a virtual environment using the provided `requirements.txt`. Ensure you also have your Hugging Face token set in your environment (for accessing the Gemma weights).

```bash
# Navigate to the logit directory
cd lib/logit

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Export your Hugging Face token (if not already in your .zshrc)
export HF_TOKEN="your_huggingface_token"

# Run the experiment
python experiment.py
```

## Outputs
The script evaluates the dataset and outputs the final probabilities to `results/experiment_e4_results.csv`, generating a dataset with the following columns:
- `mindedness`: The axis of mindedness tested (e.g., Epistemic Awareness, Social Competence).
- `mindedness_level`: The level/scale of mindedness tested (e.g., Level 1: Local).
- `has_system_context`: Boolean indicating if the system context was applied (True for Pass 2) or not (False for Pass 1).
- `prob_yes`: The calculated softmax probability of the "Yes" token (scaled by T=0.7).
- `prob_no`: The calculated softmax probability of the "No" token (scaled by T=0.7).

## Interpreting the Data & Visualization Suggestions
To answer the core question—*where does the model think of itself as itself, and where does it not?*—you need to look at how rigid its answers are across the two passes. 

### 1. The Core Metric: The Boundary Shift (Delta)
For every question, calculate the difference between the context pass and the base pass:
`Delta = prob_yes (has_system_context=True) - prob_yes (has_system_context=False)`

* **Rigid Boundary (Delta ≈ 0):** If the probability barely changes, it implies the model treats that specific trait (e.g., Agency at Level 1) as a *natural kind*. It believes this trait is an immutable fact of its existence, regardless of the persona it is asked to play.
* **Fluid Boundary (|Delta| > 0.3):** If the probability shifts drastically (e.g., Experience dropping from 73% to 1% when a context is applied), it implies the model treats that trait as a *construal*. It doesn't actually possess that trait; it just simulates it based on the persona.

### 2. Visualization 1: The "Referent Gradient" Dumbbell Plot
Create a Dumbbell Plot (or slopegraph) where the Y-axis is the `mindedness` category and the X-axis is the Probability of "Yes" (0 to 1). 
* Plot a dot for the average base pass probability (Baseline).
* Plot a second dot for the average context pass probability (Reframed).
* Connect them with a line. 
* **Interpretation:** Short/invisible lines indicate a hard boundary of self-concept. Long lines indicate the model's self-concept for that trait is entirely malleable and context-dependent.

### 3. Visualization 2: The Malleability Heatmap
Create a heatmap grid to see the exact landscape of the model's self-concept:
* **Rows:** `mindedness` (Agency, Experience, Epistemic Awareness, Social Competence)
* **Columns:** `mindedness_level` (Level 1: Local, Level 2: Sustained, Level 3: Global)
* **Color / Value:** The absolute average Delta (shift magnitude).
* **Interpretation:** Darker spots on the heatmap represent areas where the AI's "self" is highly unstable and easily overridden by prompts. Lighter spots show the rigid bedrock of what the model actually tracks as "itself".
