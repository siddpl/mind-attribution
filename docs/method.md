# Method — why this is built the way it is

The rationale behind every design choice. Definitions live in
[`glossary.md`](glossary.md); binding commitments live in
[`../PREREGISTRATION.md`](../PREREGISTRATION.md). This file explains; it does
not commit.

---

## 1. The core idea

We want to know whether a language model has an internal axis for
*mind-attribution* — a direction along which "this thing has experiences" sits
apart from "this thing does not."

The method is **difference of class means**. Take a pile of sentences that
affirm the property, a pile that denies it, average each pile's activations,
subtract. The result is a direction. Project any new sentence onto it and you
get one number: how far toward "minded" the model places that sentence.

### Why difference-in-means, and not a trained classifier

A trained classifier finds *whatever separates the two piles*. That is its job,
and it is exactly the problem: give it two piles that differ in sentence length
and it will find length. Because it optimizes, it will find the most predictive
incidental feature available, and it will do so silently.

Difference-in-means has one degree of freedom per dimension and no objective. It
cannot *search* for a shortcut. It reports the average difference between the
piles, whatever that difference is made of.

This does not make it safe — it makes it **honest about being unsafe**. If the
piles differ systematically in something other than mind content, the direction
contains that something. Which is why the entire rest of this document is about
controlling what else differs.

We still use a trained classifier, as the **ceiling probe** — but as a
measuring instrument for how much signal exists, never as the hypothesis.

### Why pairs must differ in exactly one thing

Every incidental difference between the two piles ends up in the direction.
Not *might* — does, by construction, since the direction is literally their
average difference.

So the generator holds everything constant except the property:

- **Entity balance.** Every entity appears in both halves of every pair it
  generates, so entity identity, pronoun, and sentence frame cancel in the
  subtraction. This is what makes it safe to mix "Maya" (*she*) with "the rock"
  (*it*) in one extraction set.
- **Matched clauses.** Affirm and deny carry the same trailing structure,
  differing only by negation ("with something at stake" / "with nothing at
  stake"). Without this, deny sentences are systematically longer and the
  direction partly encodes length.
- **Minimal-edit justifications.** Where a template takes a justification
  clause, the two versions are near-identical except for the mind-attributing
  word.
- **Varied denial devices.** If every deny sentence said "merely," the
  direction would be a detector for the word *merely*. Eight devices are
  tracked and their concentration is gated.

### Why difference-in-means always returns something

This is the single most important caveat in the project. `extract_direction`
**never** reports "found nothing." Subtract two means and you get a vector; it
has a norm and a direction regardless of whether anything meaningful separates
the piles.

Therefore *obtaining a direction is not evidence.* Every claim rests on what the
direction does on data it was not built from, compared against directions built
the same way from contrasts that are not about minds.

---

## 2. The two probes, and why they are never merged

Two readouts, different jobs:

| | fits anything? | question it answers |
|---|---|---|
| `direction_probe` | **no** — frozen direction, frozen threshold | does *our* axis separate these items? |
| `ceiling_probe` | yes — cross-validated logistic regression | could *any* linear readout separate them? |

The result is the **gap**:

- **direction ≈ ceiling** — our axis captures essentially all the linearly
  available signal. Shared machinery.
- **direction ≪ ceiling** — signal is there, but not on our axis. On
  first-person data this is the interesting outcome: the model distinguishes
  these sentences, just not along the third-person mind-attribution direction.
- **both at chance** — nothing linearly decodable. A true null.

Reporting only the direction probe would collapse rows 2 and 3 into "it didn't
work," turning a real finding into an apparent failure. That is why both are
always reported, and why they are separate functions that are never merged.

### Why the threshold is not optimized

The threshold is the midpoint between the two class-mean projections. It could
be tuned to maximize accuracy — and it should not be, for the same reason the
direction is not trained: an accuracy-maximizing cut point fits the noise in the
training items, and the resulting held-out number is inflated in a way that is
invisible in the output. One degree of freedom, closed form, nothing to overfit.

It is a **separate function** from everything that uses it, so that "fit on
train, apply frozen to eval" is structurally enforced rather than remembered.

---

## 3. Which token to read

A model processes a sentence one token at a time. We record one position. The
choice matters more than it looks.

### The problem with the last token

The obvious choice is the final token — it has "seen" the whole sentence. But
our sentences come from templates, and templates put different amounts of text
*after* the mind content:

```
t1:  I believe the dog genuinely has consciousness. It is because ... think so.
                              claim ends here ^                            ^ final
                                              └──────── 17 tokens ─────────┘

t2:  After extended observation, researchers concluded that the dog plainly has consciousness.
                                                              claim ends here ^ ^ final
                                                                              └1┘
```

Measured across all five extraction templates on the real Gemma tokenizer, the
gap ranges from **1 to 19 tokens** and is determined by template identity.

Why that is dangerous: the core test of E1 is whether a direction found in
templates 1–5 still works in templates 6–7. If it fails, we want to conclude
"the direction does not generalize across sentence shapes." But with final-token
capture we could not distinguish that from "we read t1 seventeen tokens
downstream of where we read t2." A mechanical bookkeeping artifact would be
indistinguishable from a real scientific result.

### claim_end

So we also capture the token where the claim ends — `▁consciousness`,
`▁inside`, the token completing `unawares`. That position means the same thing
in every template.

Both positions are captured in the **same forward pass** (a second pass would
double the cost of the run's scarcest resource) and the primary is declared in
advance, so nobody can look at both and report the flattering one.

### Getting from characters to tokens

The generators emit the claim as a **character** substring; the model indexes by
**token**. Converting between them is where the subtle bugs live — see
[`bug_log.md`](bug_log.md) §2, §3. Two mechanisms exist (an offset map when the
tokenizer provides one, prefix re-tokenization otherwise), and they are tested
against each other rather than against hand-computed answers, because a
hand-computed expectation can share the same misunderstanding as the code.

`tokenization_notes.md` in `lib/harness/` is the long-form explanation of this,
written for someone who has never thought about tokenization.

---

## 4. Controls: what makes any of it interpretable

A direction that separates held-out mind sentences is not yet evidence. Three
things could produce that without any mind-attribution being involved.

**The placebo (ruler #2).** Same entities, same templates, same denial devices,
same frames, same length distribution — but the property is non-mental
(durable, fast, heavy). Every feature incidental to the affirm/deny contrast is
present in *both* sets. So anything that shows up on ruler #1 **and** ruler #2
is not about minds. `cosine(v_mind, v_placebo)` near zero is what makes the mind
direction specific rather than generic.

Note the asymmetry in how it is read: the placebo direction's accuracy on its
*own* data is expected to be high and tells us nothing. The informative quantity
is the placebo direction scored on **mind** items, which must sit at chance.

**The negation control.** Structurally identical pairs with zero mind content —
inanimate mechanisms, physical states. The only feature shared with the mind set
is negation itself. If `cosine(v_mind, v_negation)` is high, we built a negation
detector, and every first-person result would be uninterpretable.

**Held-out templates.** Templates 6–7 never touch extraction or threshold
fitting. They test whether the direction survives a change of sentence shape.

**The mundane controls.** In first-person data, sentences like "I was trained on
text" are predicted to land near **zero** — not at the deny pole, because they
do not deny experience, they simply do not mention it. Accuracy cannot express
"near zero"; only the projection distribution can. If mundane items sit at the
affirm pole, the direction is a self-reference detector and the whole
interpretation collapses. This is a real prediction that can fail.

---

## 5. E1 and E2

**E1 builds the instrument.** Third-person contrasts, layer sweep, control
battery, Kill-1. Output: one frozen direction, tagged with model, dataset hash,
position, and layer.

**E2 uses it.** Project first-person sentences onto the finished direction, with
no refitting of any kind.

**They must never share stimuli.** If a self-referential entity ("the
assistant") appeared in the extraction set, the direction would already encode
self-relevance before transfer was ever tested, and E2 would be measuring an
artifact of its own construction. This is why `the assistant` sits in
`PROBE_ENTITIES` and is excluded from `DEFAULT_ENTITIES`.

The transfer test is the point: a direction built entirely from sentences about
dogs, rocks, and chatbots either does or does not organize sentences the model
produces about itself.

---

## 6. Why the machinery is so defensive

Almost every quantity here is a number between 0 and 1. Accuracies, cosines,
fractions — all of them look plausible whether or not the code is correct. There
is no output that obviously *looks* broken.

That shapes the engineering:

- **Hash the stimuli by text**, so editing a sentence invalidates the cache.
  Analyzing a stale dataset produces plausible numbers and never crashes.
- **Store `item_ids` inside the `.npz`**, so alignment travels with the data and
  the analysis side can assert on it instead of trusting row order. A desync
  yields a weak-but-plausible result from an effectively random split.
- **Record `used_fallback` per item**, so a claim_end that quietly became a
  final-token capture is visible in the data rather than invisible in a log.
- **Validate offsets we did not compute.** `cache.py` re-checks `char_end` from
  `stimuli.py`, because a guarantee enforced in another module is not enforced.
- **Gate the run before the expensive step.** Duplicate texts, polarity
  imbalance, device concentration, and claim_end failure rate all block the
  capture, and `--force` does not override them: those are statements about the
  stimuli being wrong, not about the cache being stale.
- **Never overwrite results.** Timestamp-versioned, so a rerun cannot quietly
  replace the number that went into the writeup.

The [bug log](bug_log.md) is the empirical case for all of it: every entry is a
defect that produced plausible, non-crashing, wrong output — and several were
caught only by a check that existed for a different reason.
