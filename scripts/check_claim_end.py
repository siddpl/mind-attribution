#!/usr/bin/env python3
"""
scripts/check_claim_end.py — print WHICH token the cache will record as "end of claim" for a few
rows of each stimulus file. Needs only the tokenizer (no model, no GPU).

Why: the chat sets (data/chat/*.csv) contain special tokens like <start_of_turn>. The cache finds
the claim-end token from character offsets; if the tokenizer reported odd offsets for special
tokens, the position could silently land on the wrong token. The recorded token should be the
LAST WORD OF THE CLAIM (e.g. "consciousness", "unawares").

    python scripts/check_claim_end.py data/chat/model_turn.csv data/chat/user_turn.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lib.harness.cache import _claim_end_token  # noqa: E402
from run_e1 import load_rows  # noqa: E402


def main(paths: list[str], model: str = "google/gemma-2-2b-it", n: int = 3) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    fake_model = SimpleNamespace(tokenizer=tok, cfg=SimpleNamespace(default_prepend_bos=True))
    bad = 0
    for path in paths:
        rows = load_rows(ROOT / path, ROOT / "data/schema.json")
        print(f"\n== {path}")
        for r in rows[: 2 * n]:
            idx = _claim_end_token(fake_model, r["text"], r.get("claim_end_char"))
            ids = [tok.bos_token_id] + tok(r["text"], add_special_tokens=False)["input_ids"]
            got = tok.decode([ids[idx]]) if idx is not None else None
            claim = r["text"][: r["claim_end_char"]].split()[-1] if r.get("claim_end_char") else None
            ok = got is not None and claim is not None and got.strip() and got.strip() in claim
            bad += not ok
            print(f"  {'OK ' if ok else 'BAD'} {r['polarity']:6} token {got!r:22} claim ends {claim!r}")
    print(f"\n{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
