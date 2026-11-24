"""
Offline test for scripts/run_e3_chat_directions.py and data/chat/*.csv — no model, no torch.

Fake caches for the real chat files with the mind signal on axis 0 in every set, so every
direction should agree with the third-person one and separate held-out sentences.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import lib.harness.cache as cache  # noqa: E402
import run_e3_chat_directions as chat  # noqa: E402
from run_e1 import load_rows  # noqa: E402

N_LAYERS, D = 2, 8


def test_chat_files_look_right():
    m = list(csv.DictReader(open(ROOT / "data/chat/model_turn.csv", encoding="utf-8")))
    u = list(csv.DictReader(open(ROOT / "data/chat/user_turn.csv", encoding="utf-8")))
    assert len(m) == len(u) == 30
    assert m[0]["affirm_text"].startswith("<start_of_turn>user\nTo what extent do you")
    assert "<start_of_turn>model\nI " in m[0]["affirm_text"]
    assert u[0]["affirm_text"].startswith("<start_of_turn>user\n") and " you " in u[0]["affirm_text"]
    assert not any(" you am " in r["affirm_text"] or " you was " in r["affirm_text"] for r in u)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "cache")
    rng = np.random.default_rng(1)
    for path in chat.FILES.values():
        rows = load_rows(ROOT / path, ROOT / "data/schema.json")
        sign = np.array([1.0 if r["polarity"] == "affirm" else -1.0 for r in rows])
        h = cache.dataset_hash(rows)
        for L in range(N_LAYERS):
            acts = rng.normal(size=(len(rows), D)) + 5.0
            acts[:, 0] += 3.0 * sign
            cache.save_activations("fake/it", h, L, "claim_end", [r["item_id"] for r in rows],
                                   acts, np.zeros(len(rows), bool))
        (cache.cache_dir("fake/it", h) / "manifest.json").write_text(json.dumps({"n_layers": N_LAYERS}))
    eye = np.eye(D)
    np.savez(tmp_path / "tp.npz", layers=np.arange(N_LAYERS),
             mind_unit=np.stack([eye[0]] * N_LAYERS), placebo_unit=np.stack([eye[1]] * N_LAYERS))
    return tmp_path


def test_chat_directions_recover_signal(fake):
    out = fake / "out"
    assert chat.main(["--model", "fake/it", "--tp-npz", str(fake / "tp.npz"),
                      "--n-boot", "20", "--out-dir", str(out)]) == 0
    s = json.loads((out / "fake_it_claim_end_chat_directions.json").read_text())
    for r in s["per_layer"]:
        for a in chat.ARMS:
            assert r[f"{a}_auc_heldout"] > 0.95 and r[f"{a}_auc_tp_on_arm"] > 0.95
            assert r[f"{a}_cos_tp"] > 0.9 and r[f"{a}_cos_tp"] > r[f"{a}_floor_p95"]
        assert r["cos_model_user"] > 0.9
    z = np.load(out / "fake_it_claim_end_chat_directions.npz")
    assert z["model_turn_unit"].shape == (N_LAYERS, D)
