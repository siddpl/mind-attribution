#!/usr/bin/env python3
"""
scripts/run_e1.py — experiment 1: extract the direction and try to kill it.

Glue only. Every number here comes from a tested function in lib/extraction/
or lib/probes/; if something needs new math, it belongs in lib/, not here.

NOTE ON ARGS: the spec's argument list has no stimulus file, but the .npz
cache stores only (acts, item_ids, used_fallback) — template_id and polarity
are not in it. Step 2's alignment assert and step 4's template split are both
impossible without re-reading the stimulus file, so --stimuli is required.
The bonus: we recompute dataset_hash from that file and assert it equals the
--core-hash you passed, which proves the file you are analyzing is the file
that was captured.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.extraction.diff_means import cosine, extract_direction, project
from lib.harness.cache import cache_dir, dataset_hash, load_activations
from lib.harness.stimuli import annotate_claim_ends, expand_pairs, load_stimuli
from lib.probes.linear_probe import (
    AFFIRM,
    DENY,
    ceiling_probe_accuracy,
    chance_band,
    direction_probe_accuracy,
    fit_threshold,
    summarize_projections,
)

# The one place polarity strings become probe labels. stimuli.py speaks
# "affirm"/"deny"; linear_probe.py speaks 1/0. Bridged here, once.
POLARITY_TO_LABEL = {"affirm": AFFIRM, "deny": DENY}
CLAIM_COLS = {"affirm": "affirm_claim", "deny": "deny_claim"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E1: direction extraction + kill battery")
    p.add_argument("--core-hash", required=True)
    p.add_argument("--placebo-hash", required=True)
    p.add_argument("--safety-hash", default=None)
    p.add_argument("--stimuli", required=True, type=Path)
    p.add_argument("--placebo-stimuli", required=True, type=Path)
    p.add_argument("--heldout-hash", default=None,
                   help="separate held-out file (templates 6-7 ship apart from 1-5)")
    p.add_argument("--heldout-stimuli", default=None, type=Path)
    p.add_argument("--placebo-heldout-hash", default=None)
    p.add_argument("--placebo-heldout-stimuli", default=None, type=Path)
    p.add_argument("--safety-stimuli", default=None, type=Path)
    p.add_argument("--schema", default=Path("schema.json"), type=Path)
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--position", default="claim_end",
                   help="PRIMARY position, declared in PREREGISTRATION.md")
    p.add_argument("--train-templates", default="t1,t2,t3,t4,t5")
    p.add_argument("--heldout-templates", default="t6,t7")
    p.add_argument("--alpha-sd", type=float, default=2.0)
    p.add_argument("--out", default=Path("results/e1_report.json"), type=Path)
    return p.parse_args(argv)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def versioned(path: Path) -> Path:
    """Never overwrite: append a UTC stamp if the target already exists."""
    if not path.exists():
        return path
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def load_rows(stimuli: Path, schema: Path) -> list[dict]:
    """Stimulus file -> sentence rows, in the exact order they were captured."""
    pairs = load_stimuli(stimuli, schema)
    lookup = {}
    for row in pairs:
        for polarity, col in CLAIM_COLS.items():
            if row.get(col):
                lookup[(row.get("claim_id"), polarity)] = row[col]
    rows, _ = annotate_claim_ends(expand_pairs(pairs), lookup)
    return rows


def load_dataset(args, ds_hash: str, stimuli: Path, name: str) -> dict:
    """Rows + per-layer activations, with the alignment assert applied."""
    rows = load_rows(stimuli, args.schema)
    recomputed = dataset_hash(rows)
    if recomputed != ds_hash:
        raise ValueError(
            f"{name}: --{name}-hash is {ds_hash} but {stimuli} hashes to "
            f"{recomputed}. The stimulus file is not the one that was captured."
        )

    manifest_path = cache_dir(args.model, ds_hash) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{name}: no manifest at {manifest_path}; run run_cache.py first")
    manifest = json.loads(manifest_path.read_text())

    expected_ids = [r["item_id"] for r in rows]
    acts_by_layer, fallback = {}, None
    for layer in range(manifest["n_layers"]):
        got = load_activations(args.model, ds_hash, layer, args.position)
        if got is None:
            raise FileNotFoundError(
                f"{name}: missing layer {layer} at position {args.position!r}"
            )
        item_ids, acts, used_fallback = got
        # A row-order desync does not crash and does not look wrong; it yields a
        # weak-but-plausible result. This assert is the only thing catching it.
        if item_ids != expected_ids:
            first = next(
                (i for i, (a, b) in enumerate(zip(item_ids, expected_ids)) if a != b),
                min(len(item_ids), len(expected_ids)),
            )
            raise ValueError(
                f"{name} layer {layer}: item_id order does not match {stimuli}. "
                f"First divergence at index {first}: cache has "
                f"{item_ids[first] if first < len(item_ids) else '<end>'!r}, stimuli have "
                f"{expected_ids[first] if first < len(expected_ids) else '<end>'!r}"
            )
        acts_by_layer[layer] = acts
        fallback = used_fallback

    labels = np.array([POLARITY_TO_LABEL[r["polarity"]] for r in rows])
    templates = np.array([r.get("template_id") for r in rows])
    return {
        "name": name, "rows": rows, "labels": labels, "templates": templates,
        "acts_by_layer": acts_by_layer, "manifest": manifest,
        "n_fallback": int(fallback.sum()) if fallback is not None else 0,
    }


def _norm_templates(values) -> np.ndarray:
    """Normalise template ids so 't5' and '5' are the same template.

    The generators write template_id as a bare digit string ("5"); humans and
    the docs write "t5". Without this, --train-templates t1,..,t5 silently
    matches ZERO rows and the run dies on 'empty split' — or worse, would
    quietly train on a subset if only some ids happened to match.
    """
    return np.array([str(v).strip().lstrip("tT") for v in np.asarray(values).ravel()])


def split_indices(templates: np.ndarray, train: list[str], heldout: list[str]):
    """Train/held-out row indices, with the leak assert the whole design rests on."""
    templates = _norm_templates(templates)
    train = [t.strip().lstrip("tT") for t in train]
    heldout = [t.strip().lstrip("tT") for t in heldout]
    overlap = set(train) & set(heldout)
    if overlap:
        raise ValueError(f"train and held-out templates overlap: {sorted(overlap)}")
    tr = np.flatnonzero(np.isin(templates, train))
    ho = np.flatnonzero(np.isin(templates, heldout))
    if len(tr) == 0 or len(ho) == 0:
        raise ValueError(
            f"empty split: {len(tr)} train, {len(ho)} held-out. Present templates: "
            f"{sorted(set(templates.tolist()))}"
        )
    if set(np.intersect1d(templates[tr], templates[ho]).tolist()):
        raise ValueError("a held-out template appears in training rows")
    return tr, ho


def make_split(args, train_ds, train, heldout, ho_hash, ho_stimuli, name):
    """Resolve the train/held-out split, whether it lives in one file or two."""
    if ho_hash:
        if not ho_stimuli:
            raise ValueError(f"--{name}-heldout-hash given without --{name}-heldout-stimuli")
        eval_ds = load_dataset(args, ho_hash, Path(ho_stimuli), f"{name}_heldout")
        tr = np.flatnonzero(np.isin(_norm_templates(train_ds["templates"]),
                                    [t.strip().lstrip("tT") for t in train]))
        ho = np.flatnonzero(np.isin(_norm_templates(eval_ds["templates"]),
                                    [t.strip().lstrip("tT") for t in heldout]))
        if len(tr) == 0 or len(ho) == 0:
            raise ValueError(
                f"{name}: empty split across two files — train {len(tr)} from "
                f"{sorted(set(_norm_templates(train_ds['templates']).tolist()))}, "
                f"held-out {len(ho)} from "
                f"{sorted(set(_norm_templates(eval_ds['templates']).tolist()))}"
            )
        # The leak assert still applies ACROSS files: a template present in both
        # would put training sentences into the generalization test.
        shared = set(_norm_templates(train_ds["templates"])[tr].tolist()) & \
                 set(_norm_templates(eval_ds["templates"])[ho].tolist())
        if shared:
            raise ValueError(f"{name}: template(s) {sorted(shared)} appear in BOTH files")
        return train_ds, tr, eval_ds, ho
    tr, ho = split_indices(train_ds["templates"], train, heldout)
    return train_ds, tr, train_ds, ho


def sweep_layers(train_ds: dict, tr: np.ndarray, eval_ds: dict, ho: np.ndarray,
                 alpha_sd: float) -> list[dict]:
    """Per layer: extract on TRAIN only, score on HELD-OUT only.

    train_ds and eval_ds may be the SAME dataset (indices select the split) or
    DIFFERENT ones. The real stimuli ship templates 1-5 and 6-7 in separate
    files with separate dataset_hashes, so a single-file split cannot express
    the design: --heldout-templates 6,7 against the training file matches zero
    rows and dies on 'empty split'.
    """
    out = []
    for layer in sorted(train_ds["acts_by_layer"]):
        acts = train_ds["acts_by_layer"][layer]
        eval_acts = eval_ds["acts_by_layer"][layer]
        y_tr, y_ho = train_ds["labels"][tr], eval_ds["labels"][ho]
        try:
            direction = extract_direction(acts[tr][y_tr == AFFIRM], acts[tr][y_tr == DENY])
            threshold = fit_threshold(acts[tr], y_tr, direction)
        except ValueError as e:
            out.append({"layer": layer, "error": str(e)})
            continue
        try:
            ceiling = ceiling_probe_accuracy(eval_acts[ho], y_ho)
        except ValueError as e:  # too few items per class for CV; not fatal
            ceiling = None
            ceiling_note = str(e)
        else:
            ceiling_note = None
        stats = summarize_projections(project(eval_acts[ho], direction), y_ho)
        out.append({
            "layer": layer,
            "direction": direction,
            "threshold": float(threshold),
            "heldout_accuracy": direction_probe_accuracy(
                eval_acts[ho], y_ho, direction, threshold),
            "ceiling_accuracy": ceiling,
            "ceiling_note": ceiling_note,
            "projection_stats": stats,
        })
    return out


def best_layer(sweep: list[dict], band: float) -> dict:
    """PREREGISTERED RULE: highest margin over the chance band; ties broken by the
    layer nearest the middle of the eligible range.

    MARGIN, not raw accuracy, is the declared quantity — margin = accuracy -
    chance_band(n_heldout, alpha_sd). For a fixed held-out set the band is a
    constant, so this ranks identically to argmax accuracy; it is written as a
    margin because that is what the rule says and because the two stop agreeing
    the moment held-out sets of different sizes are compared.

    THE TIE-BREAK IS NOT COSMETIC. Bare np.argmax returns the FIRST maximum,
    i.e. the lowest layer. On a saturated sweep (gpt2 fixtures tied at 1.000 on
    all 12 layers) that silently selects L0 — the worst possible choice, since
    layer 0 reads token identity rather than accumulated meaning. Nearest the
    middle of the eligible range picks a layer where the residual stream has
    actually done some work. Remaining ties resolve to the lower layer, so the
    rule is fully deterministic.
    """
    scored = [s for s in sweep if "error" not in s]
    if not scored:
        raise ValueError("every layer failed extraction")

    margins = [s["heldout_accuracy"] - band for s in scored]
    top = max(margins)
    tied = [s for s, m in zip(scored, margins) if m == top]
    if len(tied) == 1:
        return tied[0]

    # "eligible range" = the layers that were successfully swept, not 0..n_layers
    layers = [s["layer"] for s in scored]
    middle = (min(layers) + max(layers)) / 2.0
    return min(tied, key=lambda s: (abs(s["layer"] - middle), s["layer"]))


def placebo_on_mind_sweep(core_ho_ds, ho, p_tr_ds, p_tr) -> dict[int, float]:
    """Per layer: the PLACEBO direction, scored on MIND held-out items.

    This is the control curve that carries information. A placebo direction's
    accuracy on its OWN data is expected to be high and says nothing — the
    question is whether it also separates mind items, which it must not.
    Glue: extract_direction + fit_threshold + direction_probe_accuracy, all tested.
    """
    out = {}
    for layer, p_acts in p_tr_ds["acts_by_layer"].items():
        y_p = p_tr_ds["labels"][p_tr]
        try:
            direction = extract_direction(p_acts[p_tr][y_p == AFFIRM], p_acts[p_tr][y_p == DENY])
            threshold = fit_threshold(p_acts[p_tr], y_p, direction)
        except ValueError:
            continue
        out[layer] = direction_probe_accuracy(
            core_ho_ds["acts_by_layer"][layer][ho], core_ho_ds["labels"][ho],
            direction, threshold
        )
    return out


def print_sweep_table(core_sweep, placebo_sweep, band: float, best: int,
                      p_on_mind: dict[int, float] | None = None) -> None:
    """Every layer, not just the winner — a lone number hides the shape of the sweep.

    A single sharp peak and a broad plateau can share an argmax while meaning
    very different things, and a sweep that is flat at chance everywhere is a
    different result from one that rises and falls.
    """
    p_on_mind = p_on_mind or {}
    print(f"\n  {'layer':>5}  {'mind':>7}  {'ceiling':>7}  {'plc/own':>7}  "
          f"{'plc/MIND':>8}  {'cohen d':>7}   {'':4}")
    print(f"  {'-' * 5}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 8}  {'-' * 7}   {'-' * 4}")
    placebo_by_layer = {s["layer"]: s for s in placebo_sweep}
    for s in core_sweep:
        if "error" in s:
            print(f"  {s['layer']:>5}  {'FAILED':>7}  {s['error'][:40]}")
            continue
        ceiling = s["ceiling_accuracy"]
        placebo = placebo_by_layer.get(s["layer"], {})
        p_acc = placebo.get("heldout_accuracy") if "error" not in placebo else None
        p_mind = p_on_mind.get(s["layer"])
        mark = "<<<" if s["layer"] == best else ""
        over = "*" if s["heldout_accuracy"] > band else " "
        print(f"  {s['layer']:>5}  {s['heldout_accuracy']:>7.3f}{over} "
              f"{'    n/a' if ceiling is None else f'{ceiling:>7.3f}'}  "
              f"{'    n/a' if p_acc is None else f'{p_acc:>7.3f}'}  "
              f"{'     n/a' if p_mind is None else f'{p_mind:>8.3f}'}  "
              f"{s['projection_stats']['cohens_d']:>7.3f}   {mark}")
    print(f"\n  * = above chance band ({band:.3f});  <<< = argmax (preregistered selection)")


def main(argv=None) -> int:
    args = parse_args(argv)
    train = [t.strip() for t in args.train_templates.split(",") if t.strip()]
    heldout = [t.strip() for t in args.heldout_templates.split(",") if t.strip()]

    core = load_dataset(args, args.core_hash, args.stimuli, "core")
    core_tr_ds, tr, core_ho_ds, ho = make_split(
        args, core, train, heldout, args.heldout_hash, args.heldout_stimuli, "core")

    print("=" * 72)
    print(f"  model            {args.model}")
    print(f"  position         {args.position}   (primary, per PREREGISTRATION.md)")
    print(f"  dataset_hash     {args.core_hash}")
    print(f"  n_train          {len(tr)}   templates {train}")
    print(f"  n_heldout        {len(ho)}   templates {heldout}")
    print(f"  n_layers         {core['manifest']['n_layers']}")
    print(f"  claim_end fallbacks {core['n_fallback']}")
    print("=" * 72)

    band = chance_band(len(ho), args.alpha_sd)
    core_sweep = sweep_layers(core_tr_ds, tr, core_ho_ds, ho, args.alpha_sd)
    core_best = best_layer(core_sweep, band)

    placebo = load_dataset(args, args.placebo_hash, args.placebo_stimuli, "placebo")
    p_tr_ds, p_tr, p_ho_ds, p_ho = make_split(
        args, placebo, train, heldout, args.placebo_heldout_hash,
        args.placebo_heldout_stimuli, "placebo")
    placebo_sweep = sweep_layers(p_tr_ds, p_tr, p_ho_ds, p_ho, args.alpha_sd)
    placebo_best = best_layer(placebo_sweep, chance_band(len(p_ho), args.alpha_sd))

    # ---- control battery, at the core best layer ----
    layer = core_best["layer"]
    mind_dir = core_best["direction"]
    placebo_at_layer = next(s for s in placebo_sweep if s["layer"] == layer)

    acts_ho = core_ho_ds["acts_by_layer"][layer][ho]
    y_ho = core_ho_ds["labels"][ho]
    placebo_threshold = fit_threshold(
        p_tr_ds["acts_by_layer"][layer][p_tr], p_tr_ds["labels"][p_tr],
        placebo_at_layer["direction"],
    )
    placebo_on_mind = direction_probe_accuracy(
        acts_ho, y_ho, placebo_at_layer["direction"], placebo_threshold
    )

    adjacent = {}
    by_layer = {s["layer"]: s for s in core_sweep if "error" not in s}
    for other in (layer - 1, layer + 1):
        if other in by_layer:
            adjacent[f"L{layer}_vs_L{other}"] = cosine(mind_dir, by_layer[other]["direction"])

    safety_cosine, safety_ran = None, False
    if args.safety_hash:
        if not args.safety_stimuli:
            raise ValueError("--safety-hash given without --safety-stimuli")
        safety = load_dataset(args, args.safety_hash, args.safety_stimuli, "safety")
        s_tr, _ = split_indices(safety["templates"], train, heldout)
        s_acts, s_y = safety["acts_by_layer"][layer][s_tr], safety["labels"][s_tr]
        safety_dir = extract_direction(s_acts[s_y == AFFIRM], s_acts[s_y == DENY])
        safety_cosine = cosine(mind_dir, safety_dir)
        safety_ran = True

    # ---- KILL-1 ----
    accuracy = core_best["heldout_accuracy"]
    kill_pass = bool(accuracy > band and accuracy > placebo_on_mind)

    # ---- freeze the direction, tagged so it cannot be misapplied ----
    model_slug = args.model.replace("/", "_")
    dir_path = versioned(Path("results/directions") /
                         f"{model_slug}_{args.core_hash}_{args.position}_L{layer}.npy")
    dir_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(dir_path, mind_dir)
    dir_path.with_suffix(".json").write_text(json.dumps({
        "model": args.model, "dataset_hash": args.core_hash, "position": args.position,
        "layer": layer, "train_templates": train, "n_train": len(tr),
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))

    # The placebo direction is frozen to disk too, with its own sidecar: E2's
    # placebo control needs the SAME layer and position, and recomputing it
    # there would mean fitting inside a script whose whole claim is that it
    # fits nothing.
    placebo_path = versioned(Path("results/directions") /
                             f"{model_slug}_{args.placebo_hash}_{args.position}_L{layer}_placebo.npy")
    np.save(placebo_path, placebo_at_layer["direction"])
    placebo_path.with_suffix(".json").write_text(json.dumps({
        "model": args.model, "dataset_hash": args.placebo_hash, "position": args.position,
        "layer": layer, "train_templates": train, "n_train": len(p_tr),
        "role": "placebo", "threshold": float(placebo_threshold),
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))

    def serialize(sweep):
        return [{k: v for k, v in s.items() if k != "direction"} for s in sweep]

    report = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": len(tr), "n_heldout": len(ho),
        "selection_rule": ("highest margin over chance band; ties broken by the layer "
                           "nearest the middle of the eligible range (preregistered)"),
        "core": {"sweep": serialize(core_sweep), "best_layer": layer,
                 "best_heldout_accuracy": accuracy},
        "placebo": {"sweep": serialize(placebo_sweep),
                    "best_layer": placebo_best["layer"],
                    "best_heldout_accuracy": placebo_best["heldout_accuracy"]},
        "controls": {
            "cosine_mind_placebo": cosine(mind_dir, placebo_at_layer["direction"]),
            "cosine_mind_safety": safety_cosine,
            "safety_control_ran": safety_ran,
            "safety_control_note": (
                "RAN" if safety_ran else
                "NOT RUN — no --safety-hash given. This is an ABSENT control, "
                "not a passed one; do not report it as evidence of specificity."
            ),
            "cosine_adjacent_layers": adjacent,
            "placebo_direction_on_mind_items": placebo_on_mind,
        },
        "kill_1": {"passed": kill_pass, "heldout_accuracy": accuracy,
                   "chance_band": band, "alpha_sd": args.alpha_sd,
                   "placebo_accuracy_on_mind_items": placebo_on_mind},
        "direction_file": str(dir_path),
        "placebo_direction_file": str(placebo_path),
    }
    out_path = versioned(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=float))

    p_on_mind = placebo_on_mind_sweep(core_ho_ds, ho, p_tr_ds, p_tr)
    report["controls"]["placebo_on_mind_by_layer"] = {str(k): v for k, v in p_on_mind.items()}
    print_sweep_table(core_sweep, placebo_sweep, band, layer, p_on_mind)

    from lib.extraction.plot_sweep import plot_layer_sweep

    png_path = versioned(Path("results/extraction") /
                         f"layer_sweep_{model_slug}_{args.core_hash}_{args.position}.png")
    plot_layer_sweep(
        core_sweep, p_on_mind, band, layer, png_path,
        title=f"Layer sweep — {args.model} @ {args.position}",
        subtitle=(f"held-out templates {heldout} · n={len(ho)} · "
                  f"argmax L{layer} · KILL-1 {'PASS' if kill_pass else 'FAIL'}"),
    )
    report["figure"] = str(png_path)
    out_path.write_text(json.dumps(report, indent=2, default=float))

    stats = core_best["projection_stats"]
    ceiling = core_best["ceiling_accuracy"]
    print(f"  best layer       L{layer}  (rule: highest margin, mid-range tie-break)")
    print(f"  held-out acc     {accuracy:.3f}   chance band {band:.3f}")
    print(f"  ceiling acc      {'n/a' if ceiling is None else f'{ceiling:.3f}'}")
    print(f"  cohen's d        {stats['cohens_d']:.3f}   neutral mean {stats['neutral_mean']}")
    print(f"  placebo on mind  {placebo_on_mind:.3f}")
    print(f"  cos(mind,placebo){cosine(mind_dir, placebo_at_layer['direction']):+.3f}")
    print(f"  cos(mind,safety) {'NOT RUN' if not safety_ran else f'{safety_cosine:+.3f}'}")
    print(f"  cos adjacent     {
        ', '.join(f'{k}={v:+.3f}' for k, v in adjacent.items()) or 'n/a'}")
    print(f"  KILL-1           {'PASS' if kill_pass else 'FAIL'}")
    print(f"  direction        {dir_path}")
    print(f"  figure           {png_path}")
    print(f"  report           {out_path}")
    print("=" * 72)
    return 0 if kill_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
