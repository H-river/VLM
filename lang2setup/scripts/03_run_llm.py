#!/usr/bin/env python3
"""
03_run_llm.py
CLI script: evaluate LLM predictions with few-shot prompting on test set.

Usage:
    # Small test (100 samples)
    python -m lang2setup.scripts.03_run_llm \
        --test lang2setup/data/lang2setup_test.jsonl \
        --train lang2setup/data/lang2setup_train.jsonl \
        --max-samples 100

    # Full test set
    python -m lang2setup.scripts.03_run_llm \
        --test lang2setup/data/lang2setup_test.jsonl \
        --train lang2setup/data/lang2setup_train.jsonl
"""
import argparse
import json
import time
from pathlib import Path

from lang2setup.baselines.retrieval import RetrievalBaseline
from lang2setup.llm_interface.prompt_builder import build_prompt
from lang2setup.llm_interface.api_caller import call_llm, load_llm_config
from lang2setup.llm_interface.schema import parse_llm_output
from lang2setup.evaluation.param_metrics import evaluate_predictions, print_eval_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="lang2setup/data/lang2setup_test.jsonl")
    parser.add_argument("--train", default="lang2setup/data/lang2setup_train.jsonl")
    parser.add_argument("--embeddings", default="lang2setup/data/embeddings")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Max test samples to evaluate (default 100 to save cost)")
    parser.add_argument("--few-shot", type=int, default=5)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="lang2setup/data/llm_predictions.jsonl",
                        help="Save predictions to JSONL")
    args = parser.parse_args()

    # Load LLM config
    config = load_llm_config(args.config) if args.config else load_llm_config()

    # Load test data
    with open(args.test) as f:
        test_records = [json.loads(line) for line in f]
    if args.max_samples and args.max_samples < len(test_records):
        test_records = test_records[:args.max_samples]

    print(f"▶ Evaluating {len(test_records)} test samples with LLM ({config['llm']['model']})")
    print(f"  Few-shot examples: {args.few_shot}")

    # Load retriever for few-shot example selection
    retriever = RetrievalBaseline()
    embeddings_path = Path(args.embeddings)
    if embeddings_path.exists():
        print("  Loading cached embeddings...")
        retriever.load(args.embeddings)
    else:
        print("  Computing embeddings from training data...")
        retriever.fit(args.train)
        retriever.save(args.embeddings)

    # Run predictions
    predictions = []
    targets = []
    parse_failures = 0
    total_cost_tokens = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fout:
        for i, rec in enumerate(test_records):
            query = rec["text"]
            target = rec["target"]

            # Get few-shot examples from retriever
            _, examples = retriever.predict_with_examples(query, n=args.few_shot)

            # Build prompt and call LLM
            messages = build_prompt(query, examples)
            try:
                raw_response = call_llm(messages, config)
                pred = parse_llm_output(raw_response)
                if pred is None:
                    parse_failures += 1
                    pred = {"id": 0, "x_bin": 10, "y_bin": 10, "angle_bin": 10}
            except Exception as e:
                print(f"  ⚠ Error on sample {i}: {e}")
                parse_failures += 1
                pred = {"id": 0, "x_bin": 10, "y_bin": 10, "angle_bin": 10}

            predictions.append(pred)
            targets.append(target)

            # Save each prediction
            fout.write(json.dumps({
                "text": query,
                "prediction": pred,
                "target": target,
                "source_run": rec["source_run"],
            }) + "\n")

            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(test_records)}] completed")

            # Small delay to avoid rate limits
            time.sleep(0.1)

    print(f"\n  Parse failures: {parse_failures}/{len(test_records)}")
    print(f"  Predictions saved to {output_path}\n")

    # Evaluate
    print("🤖 LLM Baseline")
    print_eval_report(evaluate_predictions(predictions, targets))


if __name__ == "__main__":
    main()
