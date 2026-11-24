"""
Offline test for scripts/run_e3_fp_directions.py — no model, no torch.

Fake activation caches for the six real stimulus files: the "mind" signal sits on axis 0 for
BOTH first- and third-person sentences, the placebo signal on axis 1. So the script should find
first-person mind ≈ third-person mind (cosine near 1, near the ceiling, well above the floor),
and separate held-out sentences almost perfectly (AUC near 1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import lib.harness.cache as cache  # noqa: E402
import run_e3_fp_directions as fp  # noqa: E402
from run_e1 import load_rows  # noqa: E402

N_LAYERS, D = 2, 8


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "cache")
    rng = np.random.default_rng(0)
    for name, path in fp.FILES.items():
        rows = load_rows(ROOT / path, ROOT / "data/schema.json")
        ids = [r["item_id"] for r in rows]
        sign = np.array([1.0 if r["polarity"] == "affirm" else -1.0 for r in rows])
        h = cache.dataset_hash(rows)
        axis = 1 if "plc" in name else 0
        for L in range(N_LAYERS):
            acts = rng.normal(size=(len(rows), D)) + 5.0
            acts[:, axis] += 3.0 * sign
            cache.save_activations("fake/it", h, L, "claim_end", ids, acts, np.zeros(len(rows), bool))
        (cache.cache_dir("fake/it", h) / "manifest.json").write_text(json.dumps({"n_layers": N_LAYERS}))
    eye = np.eye(D)
    np.savez(tmp_path / "tp.npz", layers=np.arange(N_LAYERS),
             mind_unit=np.stack([eye[0]] * N_LAYERS), placebo_unit=np.stack([eye[1]] * N_LAYERS))
    return tmp_path


def test_first_person_direction_matches_planted_signal(fake):
    out = fake / "out"
    assert fp.main(["--model", "fake/it", "--tp-npz", str(fake / "tp.npz"),
                    "--n-boot", "20", "--out-dir", str(out)]) == 0
    z = np.load(out / "fake_it_claim_end_fp_directions.npz")
    assert z["fp_mind_unit"].shape == (N_LAYERS, D)
    assert np.allclose(np.linalg.norm(z["fp_mind_unit"], axis=1), 1)
    s = json.loads((out / "fake_it_claim_end_fp_directions.json").read_text())
    assert s["n_fp_pairs"] == 30
    for r in s["per_layer"]:
        assert r["auc_fp_mind_heldout"] > 0.95 and r["auc_fp_placebo_heldout"] > 0.95
        assert r["auc_tp_mind_on_fp_heldout"] > 0.95 and r["auc_fp_mind_on_tp_heldout"] > 0.95
        assert r["cos_fp_tp_mind"] > 0.9 and r["cos_fp_tp_placebo"] > 0.9
        assert abs(r["cos_fp_mind_placebo"]) < 0.3
        assert r["ceiling_mean"] > 0.9            # same direction, 30 pairs: near 1
        # Shuffled labels still partly follow the sentences' main axis of variation (here the
        # planted axis), so the floor is not ~0; the real direction must beat it.
        assert r["cos_fp_tp_mind"] > r["floor_p95"]
