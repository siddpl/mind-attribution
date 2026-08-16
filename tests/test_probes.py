"""
tests/test_probes.py — the acceptance tests named at the bottom of linear_probe.py.

Synthetic data only. Each test plants a known ground truth and asserts the probes
report the pattern you would have to see for the real result to mean anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from lib.probes.linear_probe import (
    AFFIRM,
    DENY,
    NEUTRAL,
    ceiling_probe_accuracy,
    chance_band,
    direction_probe_accuracy,
    fit_threshold,
    summarize_projections,
)
from lib.extraction.diff_means import extract_direction, project

D_MODEL = 16


def _basis(i: int, d: int = D_MODEL) -> np.ndarray:
    """Unit vector along axis i — the planted direction in every fixture."""
    e = np.zeros(d)
    e[i] = 1.0
    return e


def _planted(
    rng: np.random.Generator,
    n_per_class: int,
    signal: float,
    axis: int = 0,
    d: int = D_MODEL,
    n_neutral: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic noise plus +/- signal along one axis. Neutrals sit at zero.

    Neutral items get the SAME noise and no offset, which is the prereg
    prediction for mundane controls: near zero, not at the deny pole.
    """
    n = 2 * n_per_class + n_neutral
    acts = rng.normal(size=(n, d))
    labels = np.concatenate(
        [
            np.full(n_per_class, AFFIRM),
            np.full(n_per_class, DENY),
            np.full(n_neutral, NEUTRAL),
        ]
    )
    acts[labels == AFFIRM, axis] += signal
    acts[labels == DENY, axis] -= signal
    return acts, labels


# =============================================================================
# 1. SEPARABLE — planted direction, strong signal → both probes > 0.95
# =============================================================================
def test_separable_both_probes_near_ceiling():
    rng = np.random.default_rng(0)
    train_acts, train_labels = _planted(rng, n_per_class=60, signal=2.0)
    eval_acts, eval_labels = _planted(rng, n_per_class=60, signal=2.0)

    direction = _basis(0)
    threshold = fit_threshold(train_acts, train_labels, direction)

    assert direction_probe_accuracy(eval_acts, eval_labels, direction, threshold) > 0.95
    assert ceiling_probe_accuracy(eval_acts, eval_labels) > 0.95


def test_separable_works_with_an_extracted_direction():
    """Same, but the direction comes from diff_means rather than being handed in."""
    rng = np.random.default_rng(1)
    train_acts, train_labels = _planted(rng, n_per_class=60, signal=2.0)
    eval_acts, eval_labels = _planted(rng, n_per_class=60, signal=2.0)

    direction = extract_direction(
        train_acts[train_labels == AFFIRM], train_acts[train_labels == DENY]
    )
    threshold = fit_threshold(train_acts, train_labels, direction)

    assert direction_probe_accuracy(eval_acts, eval_labels, direction, threshold) > 0.95


# =============================================================================
# 2. NULL — pure noise, 20 seeds → both inside chance_band(n) every time
# =============================================================================
def test_null_data_stays_inside_the_chance_band():
    """NOT "20/20 inside the band" — that is not a property the code can have.

    chance_band IS 2 sd, so by construction ~2-3% of null draws sit above it
    (measured: 7/200). Demanding every seed pass would be demanding the band be
    wrong. What has to hold is the rate: accuracy centered on 0.5 and only the
    occasional excursion. A leak or a sign bug moves the MEAN, and that is what
    this catches.
    """
    n_per_class, n_seeds = 120, 20
    band = chance_band(2 * n_per_class)

    dir_accs, ceil_accs = [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        train_acts, train_labels = _planted(rng, n_per_class, signal=0.0)
        eval_acts, eval_labels = _planted(rng, n_per_class, signal=0.0)

        direction = _basis(0)
        threshold = fit_threshold(train_acts, train_labels, direction)

        dir_accs.append(
            direction_probe_accuracy(eval_acts, eval_labels, direction, threshold)
        )
        ceil_accs.append(ceiling_probe_accuracy(eval_acts, eval_labels))

    for name, accs in (("direction", dir_accs), ("ceiling", ceil_accs)):
        mean = float(np.mean(accs))
        n_outside = int(np.sum(np.asarray(accs) >= band))
        assert abs(mean - 0.5) < 0.02, f"{name} probe mean {mean:.4f} is not at chance"
        assert n_outside <= 2, (
            f"{name} probe exceeded the band on {n_outside}/{n_seeds} null seeds "
            f"(expect 0-1); something is reading signal out of noise"
        )


# =============================================================================
# 3. NO LEAK — a threshold fit on eval beats a threshold fit on train
# =============================================================================
def test_threshold_fit_on_eval_beats_threshold_fit_on_train():
    """If these match, the train threshold is leaking eval information.

    Needs a weak-signal, SMALL-n regime: with a strong signal both thresholds
    land in the same empty gap and score identically, and with large n the
    train threshold is estimated so well there is nothing left to steal.
    Measured optimism here is ~0.015; at n=20/class it is already ~0.000, which
    is why the numbers below are tuned rather than round.
    """
    n_per_class, signal, n_seeds = 8, 0.5, 400
    direction = _basis(0)

    honest, leaked = [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(2000 + seed)
        train_acts, train_labels = _planted(rng, n_per_class, signal=signal)
        eval_acts, eval_labels = _planted(rng, n_per_class, signal=signal)

        train_thr = fit_threshold(train_acts, train_labels, direction)
        eval_thr = fit_threshold(eval_acts, eval_labels, direction)

        honest.append(
            direction_probe_accuracy(eval_acts, eval_labels, direction, train_thr)
        )
        leaked.append(
            direction_probe_accuracy(eval_acts, eval_labels, direction, eval_thr)
        )

    honest_mean, leaked_mean = float(np.mean(honest)), float(np.mean(leaked))
    assert leaked_mean > honest_mean + 0.005, (
        f"no measurable optimism from fitting on eval "
        f"(honest {honest_mean:.4f} vs leaked {leaked_mean:.4f}) — "
        f"check that the train threshold is not derived from eval items"
    )


# =============================================================================
# 4. GAP LOGIC — orthogonal direction → probe at chance, ceiling still high
# =============================================================================
def test_orthogonal_direction_gives_the_h1_null_pattern():
    """Signal exists, but not on our axis. Recognize this shape on sight."""
    rng = np.random.default_rng(3)
    n_per_class = 120
    # signal=3.0: at 2.0 the Bayes limit is ~0.977 and 15 nuisance dims drag the
    # fitted ceiling to ~0.946, which would fail a >0.95 assert for reasons that
    # have nothing to do with the gap being tested.
    train_acts, train_labels = _planted(rng, n_per_class, signal=3.0, axis=0)
    eval_acts, eval_labels = _planted(rng, n_per_class, signal=3.0, axis=0)

    wrong_axis = _basis(7)  # orthogonal to the planted axis 0
    threshold = fit_threshold(train_acts, train_labels, wrong_axis)

    dir_acc = direction_probe_accuracy(eval_acts, eval_labels, wrong_axis, threshold)
    ceil_acc = ceiling_probe_accuracy(eval_acts, eval_labels)

    assert dir_acc < chance_band(2 * n_per_class)
    assert ceil_acc > 0.95
    assert ceil_acc - dir_acc > 0.35, "the gap is the result; it should be obvious"


# =============================================================================
# 5. NEUTRAL — mundane controls land near zero, and absent neutrals are nan
# =============================================================================
def test_neutral_items_land_between_the_poles():
    rng = np.random.default_rng(4)
    acts, labels = _planted(rng, n_per_class=200, signal=2.0, n_neutral=200)
    proj = project(acts, _basis(0))

    stats = summarize_projections(proj, labels)

    assert np.isfinite(stats["neutral_mean"])
    assert stats["deny_mean"] < stats["neutral_mean"] < stats["affirm_mean"]
    assert abs(stats["neutral_mean"]) < 0.5, "neutrals were planted at zero"
    assert stats["n_neutral"] == 200
    assert stats["separation"] > 0
    assert stats["cohens_d"] > 1.0


def test_no_neutral_items_gives_nan_without_raising():
    rng = np.random.default_rng(5)
    acts, labels = _planted(rng, n_per_class=30, signal=1.0, n_neutral=0)
    proj = project(acts, _basis(0))

    stats = summarize_projections(proj, labels)

    assert np.isnan(stats["neutral_mean"])
    assert stats["n_neutral"] == 0
    assert np.isfinite(stats["affirm_mean"]) and np.isfinite(stats["deny_mean"])


def test_neutral_items_are_excluded_from_accuracy():
    """Neutrals have no correct affirm/deny answer — they must not be scored."""
    rng = np.random.default_rng(6)
    acts, labels = _planted(rng, n_per_class=60, signal=2.0, n_neutral=60)
    direction = _basis(0)
    threshold = fit_threshold(acts, labels, direction)

    with_neutral = direction_probe_accuracy(acts, labels, direction, threshold)
    pole_mask = np.isin(labels, [AFFIRM, DENY])
    without_neutral = direction_probe_accuracy(
        acts[pole_mask], labels[pole_mask], direction, threshold
    )

    assert with_neutral == without_neutral


# =============================================================================
# The collapse — one summarize_projections, keeping both copies' strictness
# =============================================================================
def test_summarize_is_not_duplicated_in_diff_means():
    """The 0/1-only twin raised on NEUTRAL; whichever copy figure code imported
    decided whether the mundane controls plotted or crashed. One copy only."""
    import lib.extraction.diff_means as diff_means

    assert not hasattr(diff_means, "summarize_projections")


def test_summarize_rejects_labels_outside_the_vocabulary():
    """Carried over from the deleted copy: an unknown label would otherwise be
    dropped from all three splits and vanish from the counts."""
    proj = np.array([1.0, -1.0, 0.0, 5.0])
    labels = np.array([AFFIRM, DENY, NEUTRAL, 7])

    with pytest.raises(ValueError, match="labels must be in"):
        summarize_projections(proj, labels)


def test_summarize_accepts_a_column_vector_and_coerces_dtype():
    """Also carried over: float64 + ravel(), so a (n, 1) projection just works."""
    proj_col = np.array([[2.0], [2.0], [-2.0], [-2.0]], dtype=np.float32)
    labels = np.array([AFFIRM, AFFIRM, DENY, DENY])

    stats = summarize_projections(proj_col, labels)

    assert stats["affirm_mean"] == 2.0
    assert stats["deny_mean"] == -2.0
    assert stats["separation"] == 4.0


def test_summarize_returns_nan_not_a_raise_when_a_pole_is_missing():
    """The one behavior that changed in the collapse. Paired with the counts,
    so `separation = nan, n_deny = 0` reads as a data problem, not a crash."""
    proj = np.array([1.0, 2.0, 3.0])
    labels = np.array([AFFIRM, AFFIRM, AFFIRM])

    stats = summarize_projections(proj, labels)

    assert np.isnan(stats["deny_mean"]) and np.isnan(stats["separation"])
    assert stats["n_deny"] == 0 and stats["n_affirm"] == 3


def test_summarize_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        summarize_projections(np.zeros(5), np.array([AFFIRM, DENY]))


# =============================================================================
# Guards — the loud failures _check_xy exists to produce
# =============================================================================
@pytest.mark.parametrize(
    "fn", [fit_threshold, direction_probe_accuracy]
)
def test_shape_guards_raise(fn):
    rng = np.random.default_rng(7)
    acts, labels = _planted(rng, n_per_class=10, signal=1.0)
    direction = _basis(0)
    args = (direction,) if fn is fit_threshold else (direction, 0.0)

    with pytest.raises(ValueError, match="must be 2D"):
        fn(acts[0], labels, *args)
    with pytest.raises(ValueError, match="length mismatch"):
        fn(acts[:-1], labels, *args)
    with pytest.raises(ValueError, match="empty input"):
        fn(acts[:0], labels[:0], *args)


def test_fit_threshold_needs_both_classes():
    rng = np.random.default_rng(8)
    acts, labels = _planted(rng, n_per_class=10, signal=1.0)
    only_affirm = labels == AFFIRM

    with pytest.raises(ValueError, match="need both classes"):
        fit_threshold(acts[only_affirm], labels[only_affirm], _basis(0))


def test_ceiling_probe_needs_enough_items_per_class():
    rng = np.random.default_rng(9)
    acts, labels = _planted(rng, n_per_class=3, signal=1.0)

    with pytest.raises(ValueError, match="need >= cv items per class"):
        ceiling_probe_accuracy(acts, labels, cv=5)


def test_chance_band_shrinks_with_n():
    assert chance_band(40) > chance_band(400) > 0.5
    with pytest.raises(ValueError):
        chance_band(0)
