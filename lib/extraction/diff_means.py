"""
lib/extraction/diff_means.py — the mind-attribution direction.

Pure numpy. No torch, no model, no I/O. Takes two piles of activations
(affirm sentences, deny sentences) and produces the arrow that separates them.
"""

from __future__ import annotations

import warnings

import numpy as np

# Degeneracy threshold: how small the difference-of-means norm may be, relative
# to the typical activation norm, before we treat it as pure noise and refuse.
_DEGENERACY_RTOL = 1e-8


def _as_2d_float(name: str, arr: np.ndarray) -> np.ndarray:
    """Coerce to a finite, non-empty 2D float64 array or raise loudly."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D (n_items, d_model), got shape {a.shape}")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values (nan/inf)")
    return a


def extract_direction(
    acts_affirm: np.ndarray,
    acts_deny: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Difference of class means: mean(affirm) - mean(deny), affirm-ward.

    Raises on shape mismatch, empty piles, non-finite input, or a degenerate
    (near-zero) difference. Returns a unit vector when normalize=True.
    """
    A = _as_2d_float("acts_affirm", acts_affirm)
    D = _as_2d_float("acts_deny", acts_deny)
    if A.shape[1] != D.shape[1]:
        raise ValueError(
            f"d_model mismatch: acts_affirm has {A.shape[1]}, acts_deny has {D.shape[1]}"
        )

    mu_affirm = A.mean(axis=0)
    mu_deny = D.mean(axis=0)
    direction = mu_affirm - mu_deny

    dir_norm = float(np.linalg.norm(direction))
    ref_norm = max(
        float(np.linalg.norm(mu_affirm)),
        float(np.linalg.norm(mu_deny)),
    )
    if ref_norm == 0.0:
        raise ValueError("degenerate extraction: both class means are the zero vector")
    if dir_norm < _DEGENERACY_RTOL * ref_norm:
        raise ValueError(
            f"degenerate direction: class means nearly coincide "
            f"(|direction|={dir_norm:.3e}, ref={ref_norm:.3e}); "
            f"refusing to normalize noise to unit length"
        )

    if normalize:
        return direction / dir_norm
    return direction


def project(
    acts: np.ndarray,
    direction: np.ndarray,
    *,
    acts_layer: int | None = None,
    direction_layer: int | None = None,
) -> np.ndarray:
    """Project each activation onto direction -> one scalar per item, (n,).

    direction is re-normalized defensively. Pass acts_layer/direction_layer to
    assert the ruler and the points come from the same layer.
    """
    A = np.asarray(acts, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    if d.ndim != 1:
        raise ValueError(f"direction must be 1D (d_model,), got shape {d.shape}")
    if A.ndim != 2:
        raise ValueError(f"acts must be 2D (n_items, d_model), got shape {A.shape}")
    if A.shape[1] != d.shape[0]:
        raise ValueError(
            f"d_model mismatch: acts has {A.shape[1]}, direction has {d.shape[0]}"
        )
    if not np.isfinite(d).all():
        raise ValueError("direction contains non-finite values (nan/inf)")
    if (
        acts_layer is not None
        and direction_layer is not None
        and acts_layer != direction_layer
    ):
        raise ValueError(
            f"layer mismatch: acts from layer {acts_layer}, "
            f"direction from layer {direction_layer}"
        )

    dn = float(np.linalg.norm(d))
    if dn == 0.0:
        raise ValueError("cannot project onto a zero-norm direction")
    return A @ (d / dn)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Signed cosine similarity of two vectors. NaN on a zero-norm operand.

    Sign is meaningful (directions are affirm-ward by construction) — no abs().
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(
            f"cosine operands must have the same length, got {x.shape} and {y.shape}"
        )
    na = float(np.linalg.norm(x))
    nb = float(np.linalg.norm(y))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    c = float(x @ y / (na * nb))
    return float(np.clip(c, -1.0, 1.0))


def extract_all_layers(
    acts_by_layer: dict[int, np.ndarray],
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Run extract_direction per layer against one shared 0/1 label split.

    The affirm/deny split is computed once and applied to every layer. Layers
    that fail the degeneracy guard are warned and skipped, not fatal.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    present = set(np.unique(labels).tolist())
    if not present <= {0, 1}:
        raise ValueError(f"labels must be 0/1, found {sorted(present)}")

    idx_affirm = np.flatnonzero(labels == 1)
    idx_deny = np.flatnonzero(labels == 0)
    if idx_affirm.size == 0 or idx_deny.size == 0:
        raise ValueError(
            f"need both classes present; affirm={idx_affirm.size}, deny={idx_deny.size}"
        )

    n = labels.shape[0]
    directions: dict[int, np.ndarray] = {}
    for layer in sorted(acts_by_layer):
        acts = np.asarray(acts_by_layer[layer])
        if acts.ndim != 2 or acts.shape[0] != n:
            raise ValueError(
                f"layer {layer}: activations shape {acts.shape} does not align "
                f"with {n} labels"
            )
        try:
            directions[layer] = extract_direction(
                acts[idx_affirm], acts[idx_deny], normalize=True
            )
        except ValueError as e:
            warnings.warn(
                f"layer {layer}: extraction failed ({e}); skipping",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
    return directions


# summarize_projections lives in lib/probes/linear_probe.py — it is a readout,
# not an extraction step, and only that version understands NEUTRAL (label 2).
# A 0/1-only copy here would raise on exactly the mundane-control data the
# neutral handling exists to serve.
