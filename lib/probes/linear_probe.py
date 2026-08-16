"""
lib/probes/linear_probe.py — readouts that turn activations into numbers.

Two probes, different jobs, never merge them:
  direction_probe  → tests OUR frozen direction (no fitting). The hypothesis test.
  ceiling_probe    → trained LR, free to use any direction. The upper bound.
  The GAP between them on first-person data IS the H1 result:
  small gap = shared machinery; large gap = self-specific machinery.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lib.extraction.diff_means import project

AFFIRM, DENY, NEUTRAL = 1, 0, 2


# =============================================================================
def _check_xy(acts: np.ndarray, labels: np.ndarray) -> None:
    """Shape guard. Every public function calls this first.

    LOGIC: shape bugs are the only failure mode here that crashes loudly, so we
    make them crash as early as possible rather than three functions later.
    """
    acts = np.asarray(acts)
    labels = np.asarray(labels)
    if acts.ndim != 2:
        raise ValueError(f"acts must be 2D (n, d_model), got shape {acts.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if len(acts) != len(labels):
        raise ValueError(f"length mismatch: {len(acts)} acts vs {len(labels)} labels")
    if len(acts) == 0:
        raise ValueError("empty input")


# =============================================================================
def fit_threshold(
    acts: np.ndarray,
    labels: np.ndarray,
    direction: np.ndarray,
) -> float:
    """Cut point on the direction separating affirm from deny. TRAIN items only.

    LOGIC: two clusters on a line; put the fence at the midpoint of their
    centers. Deliberately NOT optimized — an accuracy-maximizing threshold
    chases noise and inflates held-out accuracy invisibly. One degree of
    freedom, closed form, nothing to overfit.

    WHY IT'S SEPARATE: the threshold must be fit on train and applied frozen to
    eval. Splitting it out makes that leak structurally impossible.
    """
    _check_xy(acts, labels)
    labels = np.asarray(labels)
    proj = project(acts, direction)

    aff = proj[labels == AFFIRM]
    den = proj[labels == DENY]
    if len(aff) == 0 or len(den) == 0:
        raise ValueError(
            f"need both classes to fit a threshold; got {len(aff)} affirm, {len(den)} deny"
        )
    return float((aff.mean() + den.mean()) / 2.0)


# =============================================================================
def direction_probe_accuracy(
    acts: np.ndarray,
    labels: np.ndarray,
    direction: np.ndarray,
    threshold: float,
) -> float:
    """Accuracy from thresholding the projection onto a FROZEN direction.

    LOGIC: project, predict affirm above the threshold, compare. No fitting
    happens here at all — the direction came from train items, the threshold
    came from train items, this function only measures.

    USED THREE TIMES, same code path:
      Kill-1  held-out THIRD-person templates → below chance kills the leg
      H1      FIRST-person items, zero refitting → the transfer test
      Placebo the placebo direction on the same items → must be at chance
    """
    _check_xy(acts, labels)
    labels = np.asarray(labels)

    # Neutral/mundane items have no correct affirm-deny answer; exclude them.
    mask = np.isin(labels, [AFFIRM, DENY])
    if mask.sum() == 0:
        raise ValueError("no affirm/deny items to score")

    proj = project(acts[mask], direction)
    preds = np.where(proj > threshold, AFFIRM, DENY)
    return float((preds == labels[mask]).mean())


# =============================================================================
def ceiling_probe_accuracy(
    acts: np.ndarray,
    labels: np.ndarray,
    cv: int = 5,
    seed: int = 0,
) -> float:
    """Cross-validated logistic regression: how much signal exists at all.

    LOGIC: this one IS allowed to look at the target data — that's the point.
    It answers "if a linear readout could use any direction it liked, how well
    could it do?" That upper bound is what makes the direction probe readable:

      direction ≈ ceiling → our direction captures the available signal
      direction ≪ ceiling → signal exists but NOT on our axis. On first-person
                            data this is the privileged-access result, not a
                            flat null. Reporting only the direction probe would
                            make a real finding look like a failure.

    StandardScaler because raw activation dims vary enough in scale that L2
    regularization otherwise penalizes them unevenly.
    """
    _check_xy(acts, labels)
    acts, labels = np.asarray(acts), np.asarray(labels)

    mask = np.isin(labels, [AFFIRM, DENY])
    X, y = acts[mask], labels[mask]

    counts = np.bincount(y, minlength=2)
    if counts.min() < cv:
        raise ValueError(
            f"need >= cv items per class; got {counts.tolist()} with cv={cv}"
        )

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=folds, scoring="accuracy").mean())


# =============================================================================
def chance_band(n_items: int, alpha_sd: float = 2.0) -> float:
    """Accuracy an honest result must exceed to count as above chance.

    LOGIC: balanced binary guessing has sd = sqrt(0.25/n). Two sd above 0.5 is
    the informal line. Exists so nobody eyeballs a number and decides — 0.58 on
    40 items is noise, 0.58 on 400 items is real. Kill criteria are written
    against this function's output, not a remembered rule of thumb.

    Crude on purpose; the writeup uses a binomial test. This is the in-loop gate.
    """
    if n_items < 1:
        raise ValueError("n_items must be >= 1")
    return float(0.5 + alpha_sd * np.sqrt(0.25 / n_items))


# =============================================================================
def summarize_projections(
    proj: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Distribution stats for one set of ruler readings, split by class.

    LOGIC: accuracy compresses each item to one bit and throws the shape away.
    Two probes can both score 0.78 with totally different pictures behind them.
    Accuracy feeds the kill criteria; effect size feeds the paper.

    THE REAL REASON THIS EXISTS: the mundane controls. Prereg predicts "I was
    trained on text" lands near ZERO — not at the deny pole. Deny items
    actively deny experience; mundane items just don't mention it. Accuracy
    cannot express "near zero"; only the distribution can. If mundane items sit
    at the affirm pole, the direction is a self-reference detector and H1's
    reading collapses. Hence neutral_mean.

    Keys are contractual — figure code reads them by name.

    CANONICAL HOME: this is the only copy. diff_means.py used to carry a
    0/1-only twin; it raised on NEUTRAL, so figure code that grabbed the wrong
    import died on the mundane controls. Collapsed here deliberately.
    """
    proj = np.asarray(proj, dtype=np.float64).ravel()
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if len(proj) != len(labels):
        raise ValueError(
            f"proj/labels length mismatch: {len(proj)} vs {len(labels)}"
        )

    # An unrecognized label would be silently dropped from all three splits and
    # vanish from the counts. A label-encoding bug should be loud, not invisible.
    present = set(np.unique(labels).tolist())
    if not present <= {AFFIRM, DENY, NEUTRAL}:
        raise ValueError(
            f"labels must be in {{{DENY}, {AFFIRM}, {NEUTRAL}}} "
            f"(deny/affirm/neutral), found {sorted(present)}"
        )

    aff = proj[labels == AFFIRM]
    den = proj[labels == DENY]
    neu = proj[labels == NEUTRAL]

    nan = float("nan")
    aff_mean = float(aff.mean()) if len(aff) else nan
    den_mean = float(den.mean()) if len(den) else nan
    neu_mean = float(neu.mean()) if len(neu) else nan

    # ddof=1 (sample variance) because these are samples, not populations.
    if len(aff) > 1 and len(den) > 1:
        pooled_sd = float(np.sqrt((aff.var(ddof=1) + den.var(ddof=1)) / 2.0))
    else:
        pooled_sd = nan

    separation = aff_mean - den_mean
    cohens_d = separation / pooled_sd if pooled_sd and pooled_sd > 1e-12 else nan

    return {
        "affirm_mean": aff_mean,
        "deny_mean": den_mean,
        "neutral_mean": neu_mean,
        "separation": float(separation),
        "pooled_sd": pooled_sd,
        "cohens_d": float(cohens_d),
        "n_affirm": len(aff),
        "n_deny": len(den),
        "n_neutral": len(neu),
    }


# =============================================================================
# THE ONLY GAPS LEFT — judgment calls, not code
# =============================================================================
#
# GAP 1 — evaluate_direction(): the convenience wrapper run_e1/run_e2 will call.
#   Should bundle: fit_threshold on train → direction_probe_accuracy on eval →
#   ceiling_probe_accuracy on eval → chance_band(n_eval) → summarize_projections
#   → return one dict. Write it once you've run e1 manually and know which
#   fields you actually want in the report; writing it now guesses at that.
#
# GAP 2 — the chance-band convention. alpha_sd=2.0 is a default, not a decision.
#   Fix the value in PREREGISTRATION.md before any real data is scored, and
#   have run_e1 read it from config rather than the default here.
#
# GAP 3 — layer safety. project() will happily accept a layer-9 direction and
#   layer-14 activations and return meaningless numbers. Decide the mechanism
#   (layer tag carried with every direction + assert at call site) and enforce
#   it in the scripts, not here.
#
# =============================================================================
# ACCEPTANCE TESTS — tests/test_probes.py, write before trusting any of this
# =============================================================================
# 1. SEPARABLE   planted direction, strong signal → both probes > 0.95
# 2. NULL        pure noise, 20 seeds → both inside chance_band(n) every time
# 3. NO LEAK     threshold fit on train half vs fit on eval half → the eval-fit
#                version should be noticeably higher; if they match, a leak exists
# 4. GAP LOGIC   plant a direction, evaluate with an ORTHOGONAL one →
#                direction_probe ≈ chance while ceiling stays high. This is
#                exactly the H1-null pattern; you want to recognize it on sight.
# 5. NEUTRAL     with label==2 present, neutral_mean finite and between the two
#                poles when planted at zero; with none present, nan, no raise.
# =============================================================================
