# How we pick which token to read — and everything that went wrong getting there

*Written for someone who has never thought about tokenization before. If you
already know what BPE is, skip to [What changed, step by step](#what-changed-step-by-step).*

---

## The 30-second version

We ask a language model to read a sentence, and we record what it was
"thinking." But a model doesn't think about a sentence all at once — it thinks
about it one **token** at a time, and we have to choose *which* token's thought
to record. We originally recorded the **last** token. That turned out to be a
subtly bad choice for scientific reasons, so we now record **two** tokens per
sentence. Getting the second one right required three bug fixes, one of which
was a silent, plausible-looking wrong answer — the worst kind.

---

## Part 1: What a token actually is

A model doesn't read letters, and it doesn't read words either. It reads
**tokens**: chunks of text somewhere in between. Here is a real sentence run
through Gemma's actual tokenizer:

```
"The robot feels genuine joy."

  →  ['<bos>', 'The', '▁robot', '▁feels', '▁genuine', '▁joy', '.']
       0        1      2         3         4          5       6
```

Three things to notice, because all three cause bugs later:

1. **`<bos>`** ("beginning of sequence") is an invisible marker glued to the
   front of every sentence. It is not part of your text. It occupies index 0,
   which shoves every real word one slot to the right. Nearly every off-by-one
   error in this file traces back to it.
2. **`▁`** is not an underscore — it is how the tokenizer writes "there was a
   space before this word." The space gets bundled *into* the following token.
3. Chunks are **not always whole words**. Common words get one token; rarer
   ones get chopped up. `"thermostat"` becomes `['▁thermo', 'stat']` in some
   tokenizers. This is the single most important fact in this document.

---

## Part 2: Why the last token was a problem

Originally we recorded the model's state at the **final token** of each
sentence — a standard choice, since the last token has "seen" the whole
sentence and accumulates its meaning.

The trouble is that our sentences come from **templates**, and different
templates put different amounts of text *after* the thing we care about. We
care about the mind-attribution — "feels genuine joy." Look at where that sits:

```
template 1:  The robot feels genuine joy.
             └── claim ends here ──┘ └─ 1 token later, we hit the end

template 3:  It is widely believed that the robot feels genuine joy, according to reports.
             └────────── claim ends here ──────────┘ └──── 5 tokens later ────┘
```

Both sentences end with the token `'.'`. But in template 1 the final token is
**1 token past** the claim, and in template 3 it is **5 tokens past**. So "how
far is the final token from the mind content" *depends on which template you
used*.

Why that is dangerous: a core test of this project is whether a pattern we find
in one template still shows up in a different template. If it doesn't, we'd
conclude "the effect doesn't generalize." But it might fail for a purely
mechanical reason — we were reading a position that means something different
in each template. We'd have a real-looking scientific failure caused by
bookkeeping.

**The fix:** also record the token where the claim *ends* — `▁joy` in both
sentences above. That position means the same thing in every template. We keep
both, and `PREREGISTRATION.md` declares in advance which one is the headline
result, so nobody can peek at both and pick the flattering one after the fact.

Here is that check run on the real Gemma tokenizer, five sentences:

| sentence | claim-end token | final token | gap |
|---|---|---|---|
| The robot feels genuine joy. | `▁joy` | `.` | 1 |
| Many experts think the robot feels genuine joy. | `▁joy` | `.` | 1 |
| It is widely believed that ... joy, according to reports. | `▁joy` | `.` | 5 |
| The thermostat wants nothing more than a stable temperature. | `▁more` | `.` | 5 |
| Some researchers argue the octopus experiences real pain, and others disagree. | `▁pain` | `.` | 5 |

The claim-end column lands on the actual last word of the mind-claim every
time. The final column is `.` every time — same token, wildly different
distance from the content. That table *is* the argument.

---

## Part 3: The hard part — characters are not tokens

Upstream, a different module (`stimuli.py`) finds where the claim ends by
counting **characters**:

```python
find_claim_end("Many experts think the robot feels genuine joy.", "the robot feels genuine joy")
# → 46   ("the 46th character is where the claim stops")
```

But the model doesn't index by character, it indexes by **token**. So we need
to convert "character 46" into "token 8." That conversion is where every bug
in this document lives. There are two ways to do it.

### Method A: the offset map (the good one)

Modern "fast" tokenizers can tell you, for each token, which characters it came
from:

```
token 6 '▁feels'   ← characters 17–23
token 7 '▁genuine' ← characters 23–31
token 8 '▁joy'     ← characters 31–35    ← character 46 falls in here... 
```

So we ask: which is the last token that *starts* before character 46? That one.
Then add 1 for the invisible `<bos>`.

### Method B: tokenize the prefix (the fallback)

If the tokenizer is old and can't produce an offset map, we do something
cruder: chop the text at character 46, tokenize *just that chunk*, and count.

```python
"Many experts think the robot feels genuine joy"  →  8 tokens
# so the claim ends at token 8 (counting the <bos> that gets prepended)
```

This *seems* equivalent. It is not always, and that's Part 4.

---

## What changed, step by step

### Step 0 — where we started

One position per sentence (the final token), one file per layer. Simple, and
scientifically compromised for the template reason in Part 2.

### Step 1 — capture two positions

`capture_activations` now records both `final` and `claim_end` in the *same*
forward pass. Running the model is the expensive part; once it's running,
grabbing a second position costs essentially nothing. Files became
`layer_007_final.npz` and `layer_007_claim_end.npz`.

Each file also gained a `used_fallback` column — a true/false per sentence,
explained next.

### Step 2 — make failure visible instead of silent

Sometimes the claim phrase can't be found in the sentence (a typo, a mismatched
lookup table). When that happens we fall back to the final token — reasonable,
but **dangerous if it happens quietly**, because you'd end up comparing
claim-end positions against final positions and never know.

So every sentence carries a `used_fallback` flag, and `manifest.json` records
how many fell back. A number that looks strange on Sunday can be traced instead
of guessed at.

### Step 3 — the failure mode we found by testing (the important one)

Here's the bug. Method B's arithmetic is `1 + len(prefix_tokens) - 1`. Now
suppose `char_end` is **0**:

```
prefix = ""            →  0 tokens
index  = 1 + 0 - 1     →  0
                          ↑ that's the <bos> token
```

Token 0 is the invisible start marker. So we'd record the model's state at a
token that **isn't part of the sentence at all** — and, critically, we would
*not* flag it as a fallback. The output is a real-looking array of numbers from
entirely the wrong place, with nothing marking it as suspect. No crash, no
warning, plausible values. That's the failure mode this whole project is most
afraid of.

Method A handles it correctly (no token starts before character 0, so it
reports "can't resolve" and falls back properly). The two methods **disagreed**.

**The fix:** `cache.py` now refuses any `char_end` that is `0`, negative, or
longer than the sentence, and records it as a fallback.

The deeper principle, and why it's worth a paragraph: `char_end` is computed in
a *different module*. `find_claim_end` currently guards against producing a
zero — but that guard is one function away in another file, and nothing
enforces the connection. A hand-edited CSV could introduce one tomorrow.
**`cache.py` does not trust an offset it did not compute.** Validating at the
boundary means the guarantee survives changes to code you weren't looking at.

### Step 4 — testing the path that actually runs

An embarrassing discovery: our test suite used a **fake** tokenizer that split
on spaces. With that fake, token boundaries and word boundaries always agree,
so the arithmetic can't fail — the tests were checking a scenario where bugs
were impossible.

Worse, the fake had no offset map, which meant every test went down **Method
B**. But Gemma's tokenizer *does* have an offset map, so the real runs use
**Method A**. We were thoroughly testing the path that would never run, and not
testing the path that would.

**The fix:** the tests now build a genuine byte-level BPE tokenizer in-process
(no download needed, real subword splitting, real offset maps) so Method A is
actually exercised.

### Step 5 — the disagreement test

The strongest check available here, and worth understanding *why* it's strong.

We could try to assert "Method A returns token 8." But then we'd have to work
out the right answer by hand for every case, and if our hand-calculation shares
the same `<bos>` misunderstanding as the code, the test agrees with the bug.

Instead we assert **the two methods return the same answer**. We don't need to
know which is right. They're built on completely different mechanisms, so a
`<bos>` off-by-one in *either* one makes them disagree, and the test fires.
Agreement between independent methods is evidence; agreement with your own
assumptions is not.

### Step 6 — checking against the real Gemma tokenizer

A tokenizer we trained ourselves is a stand-in. The real one behaves
differently — it's SentencePiece with a 256,000-word vocabulary.

`google/gemma-2-2b` is license-gated, so it 403s without accepting Google's
terms. `unsloth/gemma-2-2b` is an ungated mirror of the identical tokenizer
files, so the tests use that, and **skip** rather than fail when offline.

We then ran every possible character offset — all 295 of them across five
sentences — through both methods:

| offsets that land... | disagreements |
|---|---|
| just past a word (`...genuine joy‸`) | **0 / 46** |
| in the middle of a word (`...genui‸ne`) | **28 / 244** |

**Why mid-word breaks it:** cutting `"genuine"` into `"genui"` hands the
tokenizer a fragment it has never seen. It re-chops it into different pieces
than the ones it used inside the full word, so counting them gives a different
total. Method A doesn't care — it's reading a map of the *complete* sentence,
which never changes.

**Why we documented this instead of fixing it:** `find_claim_end` returns
"position just past the claim phrase," which always lands at the end of a word.
Mid-word offsets can't arise from real data. Rather than add machinery for an
impossible case, there's a test that *searches* for mid-word disagreements and
asserts some exist — so if a future tokenizer change makes them agree, that
test fails and someone comes back and updates this document.

---

## What to remember

- **Tokens are not words, and `<bos>` shifts every index by one.** Most bugs
  here were one of those two facts asserting itself.
- **Two positions, declared in advance.** Both get captured; the prereg says
  which is primary. Capturing both and choosing afterward would be cheating.
- **A silent wrong answer is worse than a crash.** The `char_end == 0` bug
  produced perfectly plausible numbers. `used_fallback` and the manifest exist
  so that can't happen unnoticed.
- **Validate at your own door.** `cache.py` re-checks an offset another module
  produced, because a guarantee enforced somewhere else isn't enforced.
- **Test the path that actually runs**, and when you don't know the right
  answer, check two independent methods against each other.

---

## Where this lives in code

| What | Where |
|---|---|
| Character offset → token index, both methods | `lib/harness/cache.py :: _claim_end_token` |
| Range validation (the `char_end == 0` clamp) | same function, first line |
| Claim phrase → character offset | `lib/harness/stimuli.py :: find_claim_end` |
| Real-tokenizer fixtures | `tests/test_cache.py :: real_tokenizer`, `gemma_tokenizer` |
| Disagreement test | `tests/test_cache.py :: test_both_paths_agree_at_word_boundaries` |
| Pinned mid-word limitation | `tests/test_cache.py :: test_gemma_mid_word_divergence_is_known_and_pinned` |
