# lang2setup — Natural Language to Optical Setup Prediction

An NLP pipeline that maps **free-form natural language descriptions** to structured optical alignment parameters `[x_bin, y_bin, angle_bin]` for a Gaussian laser beam simulator.

## Overview

```
"Shift the beam slightly to the left with moderate positive tilt"
        │
        ▼  (rule-based / retrieval / LLM)
  { "id": 0, "x_bin": 6, "y_bin": 10, "angle_bin": 15 }
        │
        ▼  (bins → physical values)
  x_offset = -1.14 mm,  y_offset = 0.0 mm,  tilt_x = 9.52 mrad
        │
        ▼  (closed-loop: run simulator)
  Beam intensity image on sensor
```

## Features

- **21-bin discretization** per axis: x/y ±3 mm, angle ±20 mrad
- **21-level natural language descriptors** (e.g., "extreme far left" → bin 0, "centered" → bin 10, "extreme far right" → bin 20)
- **3 prediction methods**: rule-based keyword matching, retrieval (sentence-transformers), LLM (gpt-4o-mini few-shot)
- **Tiered evaluation**: physical usefulness (joint ≤2/≤1 bins, mm/mrad errors) → fine control (per-axis MAE) → exact classification
- **Free-form benchmark**: 62 queries across 8 categories (simple position, tilt, y-offset, combined, extreme, near-center, casual, ambiguous)
- **Closed-loop evaluation**: re-simulate predicted setups, compare beam profiles via SSIM/PSNR
- **Visual comparison**: side-by-side predicted vs ground-truth beam images

## Project Structure

```
lang2setup/
├── __init__.py
├── configs/
│   ├── bins.yaml            # 21 bins per axis, ranges, center_bin=10
│   ├── templates.yaml       # 21-level descriptors, 20 templates, paraphrase config
│   ├── llm.yaml             # LLM config (model, temperature, few-shot count)
│   └── split.yaml           # Train/val/test split ratios
├── data/
│   ├── lang2setup_train.jsonl      # ~15.6k training samples
│   ├── lang2setup_val.jsonl        # ~2.2k validation samples
│   ├── lang2setup_test.jsonl       # ~2.2k test samples
│   ├── llm_predictions.jsonl       # LLM outputs on test set
│   ├── freeform_benchmark.jsonl    # 62 hand-crafted benchmark queries
│   └── embeddings/                 # Cached sentence-transformer embeddings
├── data_prep/
│   ├── extract_features.py   # Extract beam metrics from simulation output
│   ├── discretize.py         # Continuous values → bin indices
│   ├── describe.py           # Bin indices → natural language descriptions
│   ├── build_dataset.py      # Assemble text + target JSONL
│   └── split.py              # Train/val/test split
├── baselines/
│   ├── rule_based.py         # Keyword → bin matching (21 levels, longest-first)
│   └── retrieval.py          # Sentence-transformer nearest-neighbour lookup
├── llm_interface/
│   ├── prompt_builder.py     # System prompt + few-shot example assembly
│   ├── api_caller.py         # OpenAI API wrapper
│   ├── output_parser.py      # JSON response → structured prediction
│   └── schema.py             # Output schema validation
├── evaluation/
│   ├── param_metrics.py      # 3-tier metrics: physical → fine → exact
│   └── closed_loop.py        # Bins → physical → re-simulate → SSIM/PSNR
├── scripts/
│   ├── 01_build_dataset.py   # Generate text+target JSONL from sim outputs
│   ├── 02_run_baseline.py    # Run rule-based + retrieval baselines
│   ├── 03_run_llm.py         # Run LLM on template test set
│   ├── 04_freeform_eval.py   # Benchmark all methods on free-form queries
│   ├── 05_closed_loop_eval.py# Re-simulate predictions, compute SSIM/PSNR
│   ├── 06_paraphrase_augment.py  # LLM-based paraphrase augmentation
│   └── 07_visual_compare.py  # Side-by-side beam image comparison
├── DESIGN.md
└── WORKFLOW_SUMMARY.md
```

## Quick Start

### Prerequisites

```bash
conda activate optical_sim
pip install openai sentence-transformers
export OPENAI_API_KEY="sk-..."
```

### 1. Build dataset from simulation outputs

```bash
cd ~/VLM
python -m lang2setup.scripts.01_build_dataset \
    --sim-dir optical_sim/outputs/random_5k
```

### 2. Run baselines (rule-based + retrieval)

```bash
python -m lang2setup.scripts.02_run_baseline
```

### 3. Run LLM evaluation on template test set

```bash
python -m lang2setup.scripts.03_run_llm
```

### 4. Free-form benchmark (all 3 methods)

```bash
python -m lang2setup.scripts.04_freeform_eval         # with LLM
python -m lang2setup.scripts.04_freeform_eval --skip-llm  # without API cost
```

### 5. Closed-loop simulation evaluation

```bash
python -m lang2setup.scripts.05_closed_loop_eval
```

### 6. Visual comparison (predicted vs ground truth)

```bash
python -m lang2setup.scripts.07_visual_compare --n 20 --save compare.png
```

## Results

### Template Test Set (100 samples, gpt-4o-mini, 5-shot)

| Metric | Rule-based | Retrieval | LLM |
|---|---|---|---|
| x exact | 99.6% | 51.1% | 94.0% |
| y exact | 78.0% | 51.1% | 47.0% |
| angle exact | 90.5% | 51.1% | 79.0% |
| joint exact | 61.4% | 5.3% | 31.0% |
| joint ≤1 | 66.1% | 12.7% | 51.0% |
| joint ≤2 | 70.3% | 20.8% | 65.0% |

### Free-form Benchmark (62 queries)

| Metric | Rule-based | Retrieval | LLM |
|---|---|---|---|
| joint exact | 16% | 0% | 29% |
| joint ≤1 | 27% | 2% | 44% |
| joint ≤2 | 34% | 2% | 55% |

The LLM excels on extreme values (83% joint ≤1) and near-center queries (83%) but struggles with combined multi-axis queries (12%) and casual language (25%).

## Bin System

Each axis is discretized into **21 uniform bins** (0–20, center = 10):

| Axis | Physical range | Bin width |
|---|---|---|
| x (horizontal offset) | ±3 mm | ~0.286 mm |
| y (vertical offset) | ±3 mm | ~0.286 mm |
| angle (tilt) | ±20 mrad | ~1.905 mrad |

## Evaluation Tiers

1. **Physical Usefulness** (Tier 1): joint within ±2/±1 bins, estimated mm/mrad errors
2. **Fine Control** (Tier 2): per-axis within-1/2, MAE, median absolute error
3. **Exact Classification** (Tier 3): per-axis and joint exact match accuracy

## License

Research use only.
