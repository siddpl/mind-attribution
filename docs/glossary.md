# Glossary

Plain definitions of every term this project uses in a specific way. If a word
here also has a general meaning, the definition given is the one that governs
in this repo.

See [`method.md`](method.md) for *why* these things are built the way they are.

---

## The objects

**Activation.** The vector a model holds at one token position, in one layer,
while reading a sentence. For gemma-2-2b that is 2304 numbers. This project
never looks at what the model *says* — only at these internal vectors.

**Residual stream (`resid_post`).** The model's running "working memory,"
carried from layer to layer. Each block reads it, computes something, and adds
the result back. `resid_post` at layer L is the stream **after** block L has
been applied. This project reads `resid_post` and nothing else — never
`resid_pre` (before the block), `mlp_out`, or `attn_out` (individual
components). One hook point, fixed in advance, so "which signal we read" is
never a free parameter.

**Layer.** One transformer block. gemma-2-2b has 26; gpt2 has 12. Early layers
carry token identity; later layers carry accumulated meaning. A result that
peaks at layer 0 is usually reading *which words are present*, not what the
sentence means.

**Direction.** A single vector in activation space that a scalar can be read
off. Ours is the **difference of class means**: average the affirm
activations, average the deny activations, subtract, normalize. Also called a
*ruler* in these docs — it is a measuring instrument, not a claim.

**Projection.** The dot product of an activation with a direction: one number
per sentence, "how far along this axis does this sentence sit." Turning 2304
numbers into 1.

---

## The stimulus vocabulary

**Pair.** One row of a stimulus CSV, holding **two** sentences that differ in
exactly one respect. Pairs exist for the humans writing them; everything
downstream works on individual sentences.

**Affirm / deny.** The two halves of a pair. `affirm_text` asserts the property;
`deny_text` describes the same observable behaviour with the property
subtracted. **These column names are structural, not semantic** — in the
first-person and referent-ladder sets, `affirm` means *experiential* and `deny`
means *mundane*, not "asserts" and "denies."

**Sentence row.** What a pair becomes after `expand_pairs`: two rows, with
`item_id` suffixed `__aff` / `__den`, a `polarity` field, and every other
column carried through. One row per forward pass.

**Entity.** The grammatical subject a claim is attributed to — "Maya," "the
dog," "the rock." Each entity appears in **both** halves of every pair it
generates, so difference-of-means cancels entity identity.

**Claim.** The mind-attributing proposition, stored in pieces (`affirm_vp`,
`deny_vp`, `prop`, `presence`/`absence`) that templates assemble.

**Template.** A sentence frame with slots. Seven exist; 1–5 are used for
extraction, 6–7 are held out. Templates vary the *packaging* of a claim so that
a direction which only works in one sentence shape can be detected.

**Claim span (`affirm_claim` / `deny_claim`).** The exact substring of the
generated sentence that carries the mind claim, emitted as a column. Guaranteed
verbatim: the claim pattern is a literal slice of the template pattern, filled
from the same slot values, so no re-typing can drift.

**Denial device.** The lexical mechanism a deny sentence uses to subtract the
property: *merely*, *simply*, *lacking*, *nothing more*, *any real*, *without*,
*lacks*, *fails*. Tracked because a direction that fires on one device is a word
detector, not a mind detector.

**Mundane control.** A self-referential sentence with no experiential content
("I was trained on text"). Predicted to land near **zero** on the direction —
not at the deny pole, because it does not deny anything.

---

## Positions

**Token position.** Which token's activation gets recorded. Two are captured:

**`final`.** The last token of the sentence. Standard, and the problem: it is
almost always `.`, sitting a *template-dependent* distance past the claim
(measured: 1 to 19 tokens). Distance-to-content becomes a function of template
identity.

**`claim_end`** *(primary)*. The token where the claim ends. Means the same
thing in every template. Resolved by converting the character offset
`claim_end_char` into a token index.

**`used_fallback`.** A per-sentence boolean stored inside each `.npz`, true when
`claim_end` could not be resolved and the `final` token was captured instead.
Its purpose is that such a substitution can never be silent.

**`<bos>`.** An invisible "beginning of sequence" token prepended to every
sentence. It occupies index 0 and shifts every real token one slot right — the
source of most off-by-one bugs in this codebase.

---

## Measurements

**`direction_probe_accuracy`.** Project onto the **frozen** direction, predict
affirm above a threshold, score. **No fitting happens here.** The hypothesis
test.

**`ceiling_probe_accuracy`.** Cross-validated logistic regression, free to use
any direction it likes. The upper bound on linearly decodable signal. It *is*
allowed to look at the target data — that is its job.

**The gap.** `ceiling − direction`. Small means our axis captures the available
signal. Large means signal exists but **not on our axis** — a positive finding,
not a null.

**Threshold.** The cut point on the direction. Midpoint of the two class means —
deliberately **not** optimized, because an accuracy-maximizing threshold chases
noise and inflates held-out accuracy invisibly. Fit on train, applied frozen.

**`chance_band(n, alpha_sd)`.** `0.5 + alpha_sd × √(0.25/n)`. The line an honest
result must clear. Exists so nobody eyeballs a number: 0.58 on 40 items is
noise, 0.58 on 400 items is real. **The band *is* 2 sd**, so ~2–3% of pure-noise
runs exceed it by construction.

**Cohen's d.** Separation in pooled standard deviations. Accuracy compresses
each item to one bit; two probes can both score 0.78 with completely different
distributions behind them. Accuracy feeds the kill criteria; effect size feeds
the paper.

**`dataset_hash`.** First 12 hex of sha256 over sorted `item_id::text`. Computed
from **sentence text**, so editing any stimulus invalidates the cache.
Deliberately **order-insensitive** — reordering a file gives the same hash,
which is why row alignment is asserted separately.

---

## The experiments and controls

**E1.** Build the ruler and try to kill it. Extract from templates 1–5, test on
held-out templates 6–7, run the control battery.

**E2.** Point the finished ruler at first-person self-reference. E1 builds the
instrument; E2 uses it. **They must never share stimuli** — a self-referential
entity in the extraction set pre-loads self-relevance into the direction before
transfer is ever tested.

**Kill-1.** The criterion that decides whether E1's direction is real:
held-out accuracy must clear the chance band **and** beat the placebo
direction's accuracy on the same items.

**Placebo (ruler #2).** Same entities, templates, denial devices, frames, and
length distribution — only the property is non-mental (durable, fast, heavy).
Every feature incidental to the affirm/deny split is present in both rulers, so
anything appearing on both is not about minds.

**Negation control.** Structurally identical pairs with **zero** mind content.
The only shared feature is negation itself. High cosine with the mind direction
means a negation detector was built.

**Held-out templates.** Templates 6–7, which never touch extraction or threshold
fitting. The generalization test.

**Referent ladder.** Graded sentences from "a different chatbot" through "you"
to "I" — the axis between third- and first-person reference.

---

## Failure modes with names

**Leak.** Any path by which evaluation data influences the thing being
evaluated. Fitting a threshold on eval items is a leak; so is a denial device
shared between training and held-out templates.

**Silent wrong answer.** Output that is plausible, non-crashing, and wrong.
Every bug in [`bug_log.md`](bug_log.md) is one. It is the failure mode this
project is structurally most exposed to, because nearly every quantity here is
a number between 0 and 1 that looks reasonable regardless of correctness.

**Row desync.** Cached activations in a different order than the stimulus rows.
Does not crash, does not look wrong; produces a weak-but-plausible result from
what is effectively a random split.
