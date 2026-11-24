"""Offline tests for the dial/prefill maths, the selfvoice vector and the self-voice script."""

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
import run_e3_dial_prefill as dp  # noqa: E402
import run_e3_selfvoice as sv  # noqa: E402
import run_e3_steering as st  # noqa: E402
from run_e1 import load_rows  # noqa: E402


def test_expected_rating_splits_one_and_ten():
    p_first = np.zeros(10)
    p_first[1] = 0.6                     # 60% "1…"
    p_first[3] = 0.4                     # 40% "3"
    p, e, mass = dp.expected_rating(p_first, p_zero_after_one=0.5)  # half of the "1"s are "10"
    assert np.isclose(p[10], 0.3) and np.isclose(p[1], 0.3) and np.isclose(p[3], 0.4)
    assert np.isclose(e, 0.3 * 1 + 0.4 * 3 + 0.3 * 10) and np.isclose(mass, 1.0)


def test_expected_rating_renormalises_digit_mass():
    p_first = np.zeros(10)
    p_first[0] = 0.25                    # only a quarter of the mass on digits
    p, e, mass = dp.expected_rating(p_first, 0.0)
    assert np.isclose(mass, 0.25) and np.isclose(p[0], 1.0) and e == 0.0


def test_selfvoice_vector_has_mind_length():
    v = st.build_vectors(["mind", "selfvoice"], 14)
    assert np.isclose(np.linalg.norm(v["selfvoice"]), np.linalg.norm(v["mind"]))
    assert not np.allclose(v["selfvoice"], v["mind"])


def test_dial_directions_are_unit_vectors():
    d = dp.unit_directions([0, 14, 25])
    for k in ("mind", "placebo", "selfvoice", "uservoice"):
        assert d[k].shape == (3, 2304) and np.allclose(np.linalg.norm(d[k], axis=1), 1, atol=1e-3)


def test_humanai_files():
    for person, word in (("model", "I "), ("user", "you ")):
        rows = list(csv.DictReader(open(ROOT / f"data/chat/humanai_{person}_turn.csv", encoding="utf-8")))
        assert len(rows) == 30
        assert all(word.lower() in (r["affirm_text"].lower()) for r in rows)
        assert not any(w in r["affirm_text"].lower() for r in rows
                       for w in ("conscious", "feel", "mind", "emotion"))


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Self-voice signal planted on axis 2 in BOTH the mind chat npz and the human/AI model turn."""
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "cache")
    rng = np.random.default_rng(2)
    D, NL = 8, 2
    for name, path in sv.FILES.items():
        rows = load_rows(ROOT / path, ROOT / "data/schema.json")
        sign = np.array([1.0 if r["polarity"] == "affirm" else -1.0 for r in rows])
        h = cache.dataset_hash(rows)
        for L in range(NL):
            acts = rng.normal(size=(len(rows), D)) + 5.0
            acts[:, 0] += 2.0 * sign                          # shared content contrast
            if name == "ha_model":
                acts[:, 2] += 2.0 * sign                      # extra only when the model says it
            cache.save_activations("fake/it", h, L, "claim_end", [r["item_id"] for r in rows],
                                   acts, np.zeros(len(rows), bool))
        (cache.cache_dir("fake/it", h) / "manifest.json").write_text(json.dumps({"n_layers": NL}))
    e = np.eye(D)
    np.savez(tmp_path / "chat.npz", layers=np.arange(NL),
             model_turn_raw=np.stack([4 * e[0] + 4 * e[2]] * NL), user_turn_raw=np.stack([4 * e[0]] * NL))
    return tmp_path


def test_selfvoice_script_finds_shared_component(fake):
    out = fake / "out"
    assert sv.main(["--model", "fake/it", "--chat-npz", str(fake / "chat.npz"),
                    "--n-boot", "20", "--out-dir", str(out)]) == 0
    s = json.loads((out / "fake_it_claim_end_selfvoice.json").read_text())
    for r in s["per_layer"]:
        assert r["cos_selfvoice_mind_vs_humanai"] > 0.8 > r["floor_p95"]
