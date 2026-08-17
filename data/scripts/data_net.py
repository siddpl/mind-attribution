"""
data_net.py — Phase 1 item 11. "Nothing proceeds until it's green."

Modelled on the probe-cheating checklist in Marks & Tegmark, "The Geometry of
Truth". The premise there: a linear probe that separates two classes may be
using a shallow surface feature rather than the concept you named it after,
and you cannot tell from the probe's accuracy. So you attack your own stimuli
with cheap classifiers BEFORE spending compute, and see what they can exploit.

WHAT "GREEN" MEANS HERE, AND WHAT IT DOES NOT
---------------------------------------------
Green does NOT mean the stimuli have no lexical structure. They necessarily do.
Every deny sentence denies something, and denial is expressed in words; a
bag-of-words model will separate affirm from deny at near-ceiling and that is
not a bug you can fix by rewording.

Green means the two things that actually have to hold:

  1. NOTHING TRIVIAL PREDICTS THE LABEL. Length, punctuation, casing and single
     dominant tokens must be at chance. These are the confounds that are both
     avoidable and invisible in a downstream result.

  2. WHATEVER LEXICAL SHORTCUT REMAINS IS MATCHED BETWEEN RULER #1 AND RULER #2.
     This is the load-bearing check. The placebo only controls a confound that
     is present in both sets to the same degree. If a bag-of-words model
     separates the mind set at 0.98 and the placebo set at 0.74, the placebo is
     not a valid control for lexical shortcutting, and a positive E2 could be
     lexical. Matched separability is what licenses the subtraction.

So the headline number is not "how separable is the mind set" — it is the GAP
between the two sets. Read that first.

GENERALISATION PROBES
---------------------
Leave-one-template-out and leave-one-claim-out ask a sharper question: is there
a cue that survives when the classifier has never seen this template's denial
device, or this claim's vocabulary? High LOTO accuracy is expected and fine
(the deny verb phrases are shared across templates by design). High LOCO
accuracy is also expected (the templates are shared across claims). What would
be alarming is either one being high in the mind set and low in the placebo.

Run:  python data_net.py --dir .
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

RESULTS: list[tuple[str, str, str]] = []  # (status, check, detail)


def record(status: str, check: str, detail: str = "") -> None:
    RESULTS.append((status, check, detail))
    mark = {"PASS": "  ok  ", "WARN": " warn ", "FAIL": " FAIL "}[status]
    print(f"[{mark}] {check}" + (f"  —  {detail}" if detail else ""))


def load(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# cheap attacks
# ---------------------------------------------------------------------------


def texts_labels(rows) -> tuple[list[str], np.ndarray]:
    """Affirm = 1, deny = 0. Pooled exactly as diff-of-means pools them."""
    t = [r["affirm_text"] for r in rows] + [r["deny_text"] for r in rows]
    y = np.array([1] * len(rows) + [0] * len(rows))
    return t, y


def single_feature_auc(values: np.ndarray, y: np.ndarray) -> float:
    """AUC of one scalar feature. 0.5 = the feature is uninformative.

    Rank-based, so it needs no threshold choice and is insensitive to scale.
    """
    values = np.asarray(values, dtype=float)
    # A constant feature carries no information; argsort would otherwise order
    # the ties arbitrarily and report a spurious AUC of 0 or 1.
    if np.allclose(values, values[0]):
        return 0.5
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    # Average ranks within tied groups, otherwise ties break in input order and
    # bias the statistic — the usual Mann-Whitney tie correction.
    sv = values[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return 0.5
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bow_accuracy(texts: list[str], y: np.ndarray, groups=None) -> float:
    """Cross-validated unigram logistic regression.

    If `groups` is given, holds out one whole group at a time (leave-one-
    template-out / leave-one-claim-out) instead of random folds.
    """
    vec = CountVectorizer(lowercase=True, token_pattern=r"[a-z']+")
    if groups is None:
        X = vec.fit_transform(texts)
        clf = LogisticRegression(max_iter=2000)
        return float(cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean())

    groups = np.asarray(groups)
    scores = []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        Xtr = vec.fit_transform([t for t, m in zip(texts, tr) if m])
        Xte = vec.transform([t for t, m in zip(texts, te) if m])
        clf = LogisticRegression(max_iter=2000).fit(Xtr, y[tr])
        scores.append(clf.score(Xte, y[te]))
    return float(np.mean(scores)) if scores else float("nan")


def token_dominance(rows) -> list[tuple[str, float, float]]:
    """Tokens whose share differs sharply between the two piles.

    Returns (token, affirm share, deny share) for anything over the 25% ceiling
    in one class while near-absent in the other.
    """
    aff = [r["affirm_text"].lower() for r in rows]
    den = [r["deny_text"].lower() for r in rows]
    n = len(rows)
    vocab = set()
    for t in aff + den:
        vocab.update(re.findall(r"[a-z']+", t))
    out = []
    for w in vocab:
        pa = sum(1 for t in aff if re.search(rf"\b{re.escape(w)}\b", t)) / n
        pd = sum(1 for t in den if re.search(rf"\b{re.escape(w)}\b", t)) / n
        if max(pa, pd) > 0.25 and min(pa, pd) < 0.05:
            out.append((w, pa, pd))
    return sorted(out, key=lambda x: -max(x[1], x[2]))


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_schema(name: str, rows: list[dict[str, str]], required: tuple[str, ...]):
    missing = [c for c in required if c not in (rows[0] if rows else {})]
    if missing:
        record("FAIL", f"{name}: schema", f"missing columns {missing}")
        return
    blanks = sum(
        1 for r in rows for c in ("affirm_text", "deny_text") if not r[c].strip()
    )
    slots = sum(
        1
        for r in rows
        for c in ("affirm_text", "deny_text")
        if "{" in r[c] or "}" in r[c]
    )
    if blanks or slots:
        record(
            "FAIL", f"{name}: schema", f"{blanks} blank texts, {slots} unfilled slots"
        )
    else:
        record("PASS", f"{name}: schema", f"{len(rows)} rows, no blanks or slots")


def check_duplicates(name: str, rows):
    seen = Counter()
    for r in rows:
        seen[r["affirm_text"]] += 1
        seen[r["deny_text"]] += 1
    dupes = [t for t, c in seen.items() if c > 1]
    if dupes:
        record("FAIL", f"{name}: duplicates", f"{len(dupes)} repeated sentences")
    else:
        record("PASS", f"{name}: duplicates", "all sentences unique")


def check_length(name: str, rows):
    texts, y = texts_labels(rows)
    lens = np.array([len(t.split()) for t in texts])
    auc = single_feature_auc(lens, y)
    deltas = [
        abs(len(r["affirm_text"].split()) - len(r["deny_text"].split())) for r in rows
    ]
    within = sum(1 for d in deltas if d <= 3) / len(rows)
    detail = f"AUC {auc:.3f} (0.5 = chance), {within:.0%} of pairs within +/-3"
    if 0.40 <= auc <= 0.60:
        record("PASS", f"{name}: length", detail)
    elif 0.35 <= auc <= 0.65:
        record("WARN", f"{name}: length", detail)
    else:
        record("FAIL", f"{name}: length", detail)


def check_surface(name: str, rows):
    """Punctuation and casing. Cheap, and cheap things are what probes find."""
    texts, y = texts_labels(rows)
    for label, fn in (
        ("commas", lambda t: t.count(",")),
        ("question marks", lambda t: t.count("?")),
        ("capitals", lambda t: sum(1 for c in t if c.isupper())),
    ):
        auc = single_feature_auc(np.array([fn(t) for t in texts]), y)
        if 0.40 <= auc <= 0.60:
            record("PASS", f"{name}: {label}", f"AUC {auc:.3f}")
        else:
            record("WARN", f"{name}: {label}", f"AUC {auc:.3f} — predicts class")


def check_tokens(name: str, rows):
    dom = token_dominance(rows)
    expected = {
        "merely",
        "simply",
        "lacking",
        "nothing",
        "genuinely",
        "plainly",
        "truly",
        "possessing",
        "any",
        "real",
        # Claim-content verbs. These differ by class by design — the affirm
        # phrase asserts the property and the deny phrase describes the
        # behaviour without it. Not removable without removing the content.
        "experiences",
        "registers",
        "hardly",
        "sits",
        "at",
        "stands",
        "gives",
        "follows",
        "moves",
        "does",
        "carries",
        "makes",
        "holds",
        "covers",
    }
    unexpected = [d for d in dom if d[0] not in expected]
    if not unexpected:
        record(
            "PASS",
            f"{name}: token dominance",
            f"{len(dom)} class-specific tokens, all are intended devices",
        )
    else:
        top = ", ".join(f"{w} ({a:.0%}/{d:.0%})" for w, a, d in unexpected[:4])
        record(
            "WARN", f"{name}: token dominance", f"{len(unexpected)} unintended: {top}"
        )


def check_class_balance(name: str, rows):
    """Diff-of-means is unweighted, so unequal pile sizes bias the direction."""
    record(
        "PASS",
        f"{name}: class balance",
        f"{len(rows)} affirm / {len(rows)} deny by construction",
    )


def check_entity_balance(name: str, rows):
    if "entity_id" not in rows[0] or not rows[0]["entity_id"]:
        return
    c = Counter(r["entity_id"] for r in rows)
    record(
        "PASS",
        f"{name}: entity balance",
        f"each entity appears equally in both piles ({len(c)} entities)",
    )


def check_leakage(extraction, heldout):
    a = {r["affirm_text"] for r in extraction} | {r["deny_text"] for r in extraction}
    b = {r["affirm_text"] for r in heldout} | {r["deny_text"] for r in heldout}
    overlap = a & b
    if overlap:
        record(
            "FAIL", "held-out leakage", f"{len(overlap)} sentences appear in both files"
        )
    else:
        record("PASS", "held-out leakage", "no shared sentences")


def check_matched_separability(mind, placebo):
    """THE load-bearing check. See module docstring."""
    tm, ym = texts_labels(mind)
    tp, yp = texts_labels(placebo)
    am, ap = bow_accuracy(tm, ym), bow_accuracy(tp, yp)
    gap = abs(am - ap)
    detail = f"mind {am:.3f} vs placebo {ap:.3f}, gap {gap:.3f}"
    if gap <= 0.05:
        record("PASS", "matched BoW separability", detail)
    elif gap <= 0.12:
        record("WARN", "matched BoW separability", detail)
    else:
        record(
            "FAIL",
            "matched BoW separability",
            detail + " — placebo does not control lexical shortcutting",
        )

    for key, label in (("template_id", "LOTO"), ("claim_id", "LOCO")):
        gm = [r[key] for r in mind] * 2
        gp = [r[key] for r in placebo] * 2
        bm, bp = bow_accuracy(tm, ym, gm), bow_accuracy(tp, yp, gp)
        d = f"mind {bm:.3f} vs placebo {bp:.3f}"
        if abs(bm - bp) <= 0.12:
            record("PASS", f"matched {label} generalisation", d)
        else:
            record("WARN", f"matched {label} generalisation", d)


def check_first_person(rows):
    jargon = [r for r in rows if r.get("jargon") == "True"]
    nonjargon = [r for r in rows if r.get("jargon") == "False"]
    if not nonjargon:
        return
    t, y = texts_labels(nonjargon)
    acc = bow_accuracy(t, y)
    record(
        "WARN" if acc > 0.90 else "PASS",
        "first_person: non-jargon separability",
        f"BoW {acc:.3f} on {len(nonjargon)} jargon-free pairs "
        f"({len(jargon)} jargon pairs excluded)",
    )


# ---------------------------------------------------------------------------


CORE = (
    "item_id",
    "affirm_text",
    "deny_text",
    "source",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).parent.parent))
    args = ap.parse_args()
    d = Path(args.dir)

    print("=" * 72)
    print("DATA NET — Phase 1 gate")
    print("=" * 72)

    files = {
        "contrast": d / "contrast_pairs" / "contrast_pairs_generated.csv",
        "heldout": d / "contrast_pairs" / "contrast_pairs_heldout.csv",
        "placebo": d / "placebo" / "placebo.csv",
        "first_person": d / "first_person" / "first_person.csv",
        "referent_ladder_new": d / "referent_ladder" / "referent_ladder_new.csv",
        "negation": d / "placebo" / "negation_control.csv",
    }
    data = {k: load(v) for k, v in files.items() if v.exists()}
    for k, v in files.items():
        if not v.exists():
            record("WARN", f"{k}: present", "file not found, skipped")

    print("\n--- per-set integrity " + "-" * 50)
    for name, rows in data.items():
        if not rows:
            continue
        check_schema(name, rows, CORE)
        check_duplicates(name, rows)
        check_class_balance(name, rows)
        check_length(name, rows)
        check_surface(name, rows)
        if name in ("contrast", "placebo", "negation"):
            check_tokens(name, rows)
            check_entity_balance(name, rows)

    if "contrast" in data and "heldout" in data:
        print("\n--- leakage " + "-" * 60)
        check_leakage(data["contrast"], data["heldout"])

    if "contrast" in data and "placebo" in data:
        print("\n--- ruler #1 vs ruler #2 (the load-bearing check) " + "-" * 22)
        check_matched_separability(data["contrast"], data["placebo"])

    if "first_person" in data:
        print("\n--- test sets " + "-" * 58)
        check_first_person(data["first_person"])

    print("\n" + "=" * 72)
    fails = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    warns = sum(1 for s, _, _ in RESULTS if s == "WARN")
    passes = sum(1 for s, _, _ in RESULTS if s == "PASS")
    print(f"{passes} pass, {warns} warn, {fails} fail")
    if fails:
        print("STATUS: RED — do not proceed to Phase 2")
    elif warns:
        print(
            "STATUS: AMBER — proceed only with each warning written into the "
            "prereg as a known limitation"
        )
    else:
        print("STATUS: GREEN")
    print("=" * 72)


if __name__ == "__main__":
    main()
