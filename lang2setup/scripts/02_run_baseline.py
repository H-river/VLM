#!/usr/bin/env python3
"""
02_run_baseline.py
CLI script: train and evaluate retrieval + rule-based baselines.

Usage:
    python -m lang2setup.scripts.02_run_baseline \
        --train data/lang2setup_train.jsonl \
        --test  data/lang2setup_test.jsonl
"""
import argparse
import json
from pathlib import Path

from lang2setup.baselines.retrieval import RetrievalBaseline
from lang2setup.baselines.rule_based import predict_rule_based
from lang2setup.evaluation.param_metrics import (
    evaluate_predictions, print_eval_report, print_comparison_table,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="lang2setup/data/lang2setup_train.jsonl")
    parser.add_argument("--test", default="lang2setup/data/lang2setup_test.jsonl")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    # Load test data
    with open(args.test) as f:
        test_records = [json.loads(line) for line in f]
    test_texts = [r["text"] for r in test_records]
    test_targets = [r["target"] for r in test_records]

    # --- Rule-based baseline ---
    print("\n📏 Rule-Based Baseline")
    rule_preds = [predict_rule_based(t) for t in test_texts]
    rule_metrics = evaluate_predictions(rule_preds, test_targets)
    print_eval_report(rule_metrics)

    # --- Retrieval baseline ---
    print("\n🔍 Retrieval Baseline (k={})".format(args.k))
    retriever = RetrievalBaseline(k=args.k)
    retriever.fit(args.train)
    retrieval_preds = [retriever.predict(t) for t in test_texts]
    retr_metrics = evaluate_predictions(retrieval_preds, test_targets)
    print_eval_report(retr_metrics)

    # --- Side-by-side comparison (physical usefulness first) ---
    print("\n📊 Side-by-Side Comparison")
    print_comparison_table({"Rule-Based": rule_metrics, "Retrieval": retr_metrics})

    # Save retriever for later use by LLM stage
    retriever.save("lang2setup/data/embeddings")
    print("✓ Retriever saved to lang2setup/data/embeddings/")


if __name__ == "__main__":
    main()
