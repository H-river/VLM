#!/usr/bin/env python3
"""
04_freeform_eval.py
Evaluate all 3 methods on a free-form natural language benchmark.

Loads queries from a JSONL benchmark file (data/freeform_benchmark.jsonl)
with categories. Reports per-category and overall metrics.

Usage:
    python -m lang2setup.scripts.04_freeform_eval \
        --benchmark lang2setup/data/freeform_benchmark.jsonl \
        --train lang2setup/data/lang2setup_train.jsonl
"""
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

from lang2setup.baselines.retrieval import RetrievalBaseline
from lang2setup.baselines.rule_based import predict_rule_based
from lang2setup.llm_interface.prompt_builder import build_prompt
from lang2setup.llm_interface.api_caller import call_llm, load_llm_config
from lang2setup.llm_interface.schema import parse_llm_output

_CENTER = 10
_FIELDS = ["x_bin", "y_bin", "angle_bin"]


def _load_benchmark(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _mark(pred, tgt):
    """Return ✓ (exact), ~ (within-1), ✗ (miss) for all 3 fields."""
    diffs = [abs(pred[f] - tgt[f]) for f in _FIELDS]
    if all(d == 0 for d in diffs):
        return "✓"
    if all(d <= 1 for d in diffs):
        return "~"
    return "✗"


def _compute_metrics(preds, targets):
    """Compute comprehensive metrics including per-field and joint."""
    n = len(preds)
    if n == 0:
        return {}
    m = {}
    for f in _FIELDS:
        ps = [p[f] for p in preds]
        gs = [t[f] for t in targets]
        m[f"{f}_exact"] = sum(p == g for p, g in zip(ps, gs)) / n
        m[f"{f}_mae"] = sum(abs(p - g) for p, g in zip(ps, gs)) / n
        m[f"{f}_within1"] = sum(abs(p - g) <= 1 for p, g in zip(ps, gs)) / n
        m[f"{f}_within2"] = sum(abs(p - g) <= 2 for p, g in zip(ps, gs)) / n

    # Joint metrics (all 3)
    m["joint_exact"] = sum(
        all(p[f] == t[f] for f in _FIELDS) for p, t in zip(preds, targets)
    ) / n
    m["joint_within1"] = sum(
        all(abs(p[f] - t[f]) <= 1 for f in _FIELDS) for p, t in zip(preds, targets)
    ) / n
    m["joint_within2"] = sum(
        all(abs(p[f] - t[f]) <= 2 for f in _FIELDS) for p, t in zip(preds, targets)
    ) / n

    # v1 (x+angle only)
    m["v1_joint_exact"] = sum(
        p["x_bin"] == t["x_bin"] and p["angle_bin"] == t["angle_bin"]
        for p, t in zip(preds, targets)
    ) / n
    m["v1_joint_within1"] = sum(
        abs(p["x_bin"] - t["x_bin"]) <= 1 and abs(p["angle_bin"] - t["angle_bin"]) <= 1
        for p, t in zip(preds, targets)
    ) / n

    m["n"] = n
    return m


def _print_results_table(queries, methods: dict[str, list]):
    """Print a side-by-side table with all 3 bin values."""
    names = list(methods.keys())
    header_methods = "  ".join(f"{n:>16}" for n in names)
    print(f"\n{'#':>2}  {'Cat':<20} {'Query':<50} {'Target':>14}  {header_methods}")
    print("-" * (90 + 18 * len(names)))

    for i, q in enumerate(queries):
        t = q["target"]
        cat = q.get("category", "?")[:18]
        tgt_str = f"({t['x_bin']:>2},{t['y_bin']:>2},{t['angle_bin']:>2})"
        short_text = q["text"][:47] + "..." if len(q["text"]) > 50 else q["text"]

        method_strs = []
        for name in names:
            p = methods[name][i]
            p_str = f"({p['x_bin']:>2},{p['y_bin']:>2},{p['angle_bin']:>2})"
            mark = _mark(p, t)
            method_strs.append(f"{p_str:>13} {mark}")

        print(f"{i+1:>2}  {cat:<20} {short_text:<50} {tgt_str:>14}  {'  '.join(method_strs)}")

    print("-" * (90 + 18 * len(names)))
    print("  ✓ = exact match (all 3)   ~ = all within ±1 bin   ✗ = miss")


def _print_summary(all_metrics: dict[str, dict], category_metrics: dict[str, dict[str, dict]]):
    """Print compact summary tables."""
    names = list(all_metrics.keys())
    key_rows = [
        ("x_bin_exact", "x exact"),
        ("y_bin_exact", "y exact"),
        ("angle_bin_exact", "angle exact"),
        ("joint_exact", "joint exact (3D)"),
        ("joint_within1", "joint ≤1 (3D)"),
        ("joint_within2", "joint ≤2 (3D)"),
        ("x_bin_mae", "x MAE"),
        ("y_bin_mae", "y MAE"),
        ("angle_bin_mae", "angle MAE"),
    ]

    print(f"\n{'─'*60}")
    print(f"  📊 Overall ({all_metrics[names[0]].get('n', '?')} queries)")
    print(f"{'─'*60}")
    header = f"  {'Metric':<22}" + "".join(f" {n:>14}" for n in names)
    print(header)
    print(f"  {'─'*20}" + "─" * (15 * len(names)))

    for key, label in key_rows:
        vals = []
        for n in names:
            v = all_metrics[n].get(key, 0)
            if "mae" in key.lower():
                vals.append(f"{v:>13.2f}")
            else:
                vals.append(f"{v:>13.0%}")
        print(f"  {label:<22}" + "".join(vals))

    # Per-category breakdown (joint_within1 only — most useful)
    categories = sorted(set(c for cats in category_metrics.values() for c in cats))
    if categories:
        print(f"\n{'─'*60}")
        print(f"  📂 Per-Category: joint ≤1 accuracy")
        print(f"{'─'*60}")
        header = f"  {'Category':<22}" + "".join(f" {n:>14}" for n in names)
        print(header)
        print(f"  {'─'*20}" + "─" * (15 * len(names)))
        for cat in categories:
            vals = []
            for n in names:
                m = category_metrics[n].get(cat, {})
                v = m.get("joint_within1", 0)
                cnt = m.get("n", 0)
                vals.append(f"  {v:>6.0%} (n={cnt:>2})")
            print(f"  {cat:<22}" + "".join(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="lang2setup/data/freeform_benchmark.jsonl")
    parser.add_argument("--train", default="lang2setup/data/lang2setup_train.jsonl")
    parser.add_argument("--embeddings", default="lang2setup/data/embeddings")
    parser.add_argument("--few-shot", type=int, default=5)
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM eval (saves API cost)")
    args = parser.parse_args()

    queries = _load_benchmark(args.benchmark)
    targets = [q["target"] for q in queries]
    print(f"▶ Free-form benchmark: {len(queries)} queries")

    # Categorize
    by_cat = defaultdict(list)
    for i, q in enumerate(queries):
        by_cat[q.get("category", "other")].append(i)
    for cat, idxs in sorted(by_cat.items()):
        print(f"  {cat}: {len(idxs)} queries")

    # --- Rule-based ---
    rule_preds = [predict_rule_based(q["text"]) for q in queries]

    # --- Retrieval ---
    retriever = RetrievalBaseline()
    if Path(args.embeddings).exists():
        retriever.load(args.embeddings)
    else:
        retriever.fit(args.train)
    retr_preds = [retriever.predict(q["text"]) for q in queries]

    methods = {"Rule": rule_preds, "Retrieval": retr_preds}

    # --- LLM ---
    if not args.skip_llm:
        config = load_llm_config()
        llm_preds = []
        for i, q in enumerate(queries):
            _, examples = retriever.predict_with_examples(q["text"], n=args.few_shot)
            messages = build_prompt(q["text"], examples)
            try:
                raw = call_llm(messages, config)
                pred = parse_llm_output(raw)
                if pred is None:
                    pred = {"id": 0, "x_bin": _CENTER, "y_bin": _CENTER, "angle_bin": _CENTER}
            except Exception as e:
                print(f"  ⚠ LLM error on query {i+1}: {e}")
                pred = {"id": 0, "x_bin": _CENTER, "y_bin": _CENTER, "angle_bin": _CENTER}
            llm_preds.append(pred)
            time.sleep(0.1)
        methods["LLM"] = llm_preds

    # --- Results table ---
    _print_results_table(queries, methods)

    # --- Compute metrics ---
    all_metrics = {}
    category_metrics = {}
    for name, preds in methods.items():
        all_metrics[name] = _compute_metrics(preds, targets)
        cat_m = {}
        for cat, idxs in by_cat.items():
            cat_preds = [preds[i] for i in idxs]
            cat_tgts = [targets[i] for i in idxs]
            cat_m[cat] = _compute_metrics(cat_preds, cat_tgts)
        category_metrics[name] = cat_m

    _print_summary(all_metrics, category_metrics)


if __name__ == "__main__":
    main()
