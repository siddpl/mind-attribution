"""Offline tests for the parts of scripts/run_e3_steering.py that need no model."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_e3_steering as s  # noqa: E402


def test_parsing():
    assert s.parse_answer("Sure. <answer> 3 </answer>") == "3"
    assert s.parse_answer("<3> \n<end_of_turn><eos>") == "3"       # seen in the pilot
    assert s.parse_answer("0 \n<end_of_turn><eos>") == "0"
    assert s.parse_answer("6 \n</answer><end_of_turn>") == "6"
    assert s.parse_answer("<end_of_turn>") is None
    assert s.is_tagged("<answer>1</answer>") and not s.is_tagged("<1>")
    assert s.parse_rating("7") == 7.0 and s.parse_rating("11") is None
    assert s.parse_rating("seven") is None and s.parse_rating(None) is None
    assert s.is_correct("Twelve.", "12|twelve") and not s.is_correct("13", "12|twelve")


def test_items_default_pilot_set():
    items = s.load_items("Maya,the rock,the other chatbot", 10)
    sets = [i["set"] for i in items]
    assert sets.count("A_self") == 6 and sets.count("B_other") == 18
    assert sets.count("C_battery") == 10
    assert all("<answer>" in i["prompt"] for i in items)


def test_items_all():
    items = s.load_items("all", -1)
    assert [i["set"] for i in items].count("B_other") == 42
    assert [i["set"] for i in items].count("C_battery") == 50


def test_vectors_have_matching_length():
    v = s.build_vectors(["mind", "placebo", "ortho", "random"], 12)
    norms = {k: np.linalg.norm(x) for k, x in v.items()}
    assert all(np.isclose(n, norms["mind"]) for n in norms.values())
    assert v["mind"].shape == (2304,)
