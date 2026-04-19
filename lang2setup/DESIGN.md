# Lang2Setup — Language-to-Optical-Setup Pipeline

## Design Document v1

---

## 1. Task Definition & Target Representation

### What `[ID, x, y, angle]` means

| Field   | Meaning | Physical source | Unit |
|---------|---------|-----------------|------|
| `ID`    | Experiment family / regime (encodes lens choice, wavelength band, distance regime) | Cluster ID from k-means on setup params | integer 0–K |
| `x`     | Lateral offset of beam centroid on sensor | `alignment.x_offset` | discretized bin index |
| `y`     | Vertical offset of beam centroid on sensor | `alignment.y_offset` | discretized bin index |
| `angle` | Dominant tilt | `alignment.tilt_x` (primary axis) | discretized bin index |

### v1 Simplification — strongly recommended

Since the setup is largely 1D and Gaussian:

1. **Fix ID = 0** (single Gaussian family). Add multi-ID later when you add non-Gaussian beams.
2. **Predict only `(x, angle)`** in v1. `y` is symmetric with `x`; add it as a trivial extension.
3. **Discretize into bins**, not continuous regression:
   - `x`: 11 bins over [−3 mm, +3 mm] → bin width ≈ 0.6 mm
   - `angle`: 11 bins over [−20 mrad, +20 mrad] → bin width ≈ 4 mrad
4. Bin 5 = center (zero offset / zero tilt). Bins 0–4 = negative, 6–10 = positive.

This turns the problem into **two small classification tasks** (11 × 11 = 121 combos), which is far more robust than regression for a first version with ~5k samples.

### Target schema (v1)

```json
{
  "id": 0,
  "x_bin": 5,
  "y_bin": 5,
  "angle_bin": 5
}
```

### Bin definitions (stored in config)

```yaml
bins:
  x:
    num: 11
    min: -0.003   # meters
    max:  0.003
  y:
    num: 11
    min: -0.003
    max:  0.003
  angle:
    num: 11
    min: -0.02    # radians
    max:  0.02
```

---

## 2. Text-Supervision Data Pipeline

### 2.1 Per-sample feature extraction

From each sample's `metadata.json`, extract a **feature vector**:

| Feature | Source | Semantics |
|---------|--------|-----------|
| `centroid_x` | metrics.centroid_x | beam position |
| `centroid_y` | metrics.centroid_y | beam position |
| `sigma_x` | metrics.sigma_x | beam width |
| `sigma_y` | metrics.sigma_y | beam width |
| `ellipticity` | metrics.ellipticity | circularity |
| `peak_intensity` | metrics.peak_intensity | brightness |
| `x_offset` | setup.alignment.x_offset | ground truth param |
| `tilt_x` | setup.alignment.tilt_x | ground truth param |

### 2.2 Template-based text generation

Define ~20–30 templates that combine **property descriptors**:

```
descriptors:
  width:
    narrow:   sigma_x < 0.7 mm
    medium:   0.7 mm ≤ sigma_x ≤ 1.0 mm
    wide:     sigma_x > 1.0 mm
  position:
    centered:        |centroid_x| < 0.15 mm
    shifted_left:    centroid_x < -0.15 mm
    shifted_right:   centroid_x > 0.15 mm
  tilt:
    no_tilt:         |tilt_x| < 2 mrad
    positive_tilt:   tilt_x ≥ 2 mrad
    negative_tilt:   tilt_x ≤ -2 mrad
  shape:
    circular:   ellipticity > 0.95
    elliptical: ellipticity ≤ 0.95
  intensity:
    bright:  peak > median(all peaks)
    dim:     peak ≤ median(all peaks)
```

Templates (examples):

```
"Generate a {width} {position} beam with {tilt} tilt."
"Create a {width} {shape} beam, {position}."
"Produce a beam that is {width}, {position}, and has {tilt} tilt."
"I need a {intensity} {width} Gaussian beam, {position}."
```

Each sample generates **3–5 text variants** → ~15k–25k text–target pairs from 5k samples.

### 2.3 Output format

Store as JSONL:

```json
{
  "text": "Generate a narrow beam shifted slightly to the left with positive tilt.",
  "target": {"id": 0, "x_bin": 3, "y_bin": 5, "angle_bin": 7},
  "target_continuous": {"x_offset": -0.0012, "y_offset": 0.0001, "tilt_x": 0.008},
  "source_run": "rand_00042",
  "split": "train"
}
```

### 2.4 Later: paraphrase augmentation

- Use an LLM to rephrase each template output into 2–3 paraphrases.
- Or use back-translation.
- Not needed for v1.

---

## 3. Prediction Strategy — Staged Recommendation

### Stage 1 (Baseline): Embedding Retrieval + Nearest Neighbor

1. Embed all training texts with a sentence transformer (e.g., `all-MiniLM-L6-v2`).
2. At inference, embed the user query → find k-nearest training texts → vote on `(x_bin, angle_bin)`.
3. **Why first:** no LLM cost, fast, interpretable, gives a performance floor.

### Stage 2 (Main): LLM with Structured Output

1. Build a prompt with:
   - system instructions defining the bins and valid ranges
   - 3–5 retrieved few-shot examples (from Stage 1 retriever)
   - the user query
2. Ask LLM to return JSON matching the schema.
3. Parse + validate output.
4. **Why second:** leverages LLM reasoning, benefits from retrieval context.

### Stage 3 (Later): Fine-tuned / Refined

- Fine-tune a small model on the 15k text–target pairs if LLM API latency/cost is an issue.
- Or use retrieval + LLM refinement: retrieve top-5 candidates, ask LLM to interpolate/select.

**Recommendation for v1: Build Stage 1 first, then Stage 2. Skip Stage 3.**

---

## 4. Structured Output Schema

### JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "id":        {"type": "integer", "minimum": 0, "maximum": 0},
    "x_bin":     {"type": "integer", "minimum": 0, "maximum": 10},
    "y_bin":     {"type": "integer", "minimum": 0, "maximum": 10},
    "angle_bin": {"type": "integer", "minimum": 0, "maximum": 10}
  },
  "required": ["id", "x_bin", "y_bin", "angle_bin"],
  "additionalProperties": false
}
```

### Validation & Fallback

1. Parse JSON from LLM output (strip markdown fences if present).
2. Validate against schema with `jsonschema`.
3. Clamp out-of-range integers to nearest valid value.
4. If parsing fails entirely → retry once with a simpler prompt.
5. If still fails → fall back to retrieval baseline prediction.

### Why bins > regression for v1

- LLMs are better at classification than precise float regression.
- Bins are trivially validatable (integer in range).
- 11 bins gives ~0.6 mm resolution, sufficient for this setup.
- Easy to make finer later (21 bins, 41 bins, etc.).

---

## 5. Dataset Split & Evaluation

### Splitting strategy

1. **Group by `(x_bin, angle_bin)`** — each bin-pair is a group.
2. Within each group, split 70/15/15 (train/val/test) by sample.
3. This ensures all bin combinations appear in all splits (stratified).
4. If a bin-pair has < 5 samples, merge into train only.

Alternatively, for stricter generalization testing:
- Hold out 10% of bin-pairs entirely for test (unseen combos).

### Evaluation Level 1: Parameter Accuracy

| Metric | Definition |
|--------|-----------|
| `x_bin_accuracy` | exact match rate for x_bin |
| `angle_bin_accuracy` | exact match for angle_bin |
| `joint_accuracy` | both correct simultaneously |
| `x_bin_MAE` | mean |pred_bin − true_bin| |
| `angle_bin_MAE` | mean |pred_bin − true_bin| |
| `within_1_bin` | fraction of predictions within ±1 bin |

### Evaluation Level 2: Closed-Loop Simulation

1. Convert predicted bins back to physical values (bin center).
2. Run the optical simulator with predicted setup.
3. Compare resulting beam profile to ground-truth beam profile.

| Metric | Definition |
|--------|-----------|
| `centroid_error` | Euclidean distance between centroids [m] |
| `width_error` | relative error in sigma_x, sigma_y |
| `peak_intensity_ratio` | predicted / true peak intensity |
| `profile_MSE` | pixel-wise MSE on normalized intensity images |
| `profile_SSIM` | structural similarity index |

---

## 6. Repository Structure

```
optical_sim/lang2setup/
├── DESIGN.md                  ← this document
├── configs/
│   ├── bins.yaml              ← bin definitions, ranges
│   ├── templates.yaml         ← text generation templates + descriptors
│   ├── split.yaml             ← split ratios, random seed
│   └── llm.yaml               ← model name, temperature, max_tokens
│
├── data_prep/
│   ├── __init__.py
│   ├── extract_features.py    ← read metadata.json → feature dict
│   ├── discretize.py          ← continuous params → bin indices
│   ├── describe.py            ← features → natural language descriptions
│   ├── build_dataset.py       ← orchestrate: read all samples → JSONL
│   └── split.py               ← stratified train/val/test split
│
├── baselines/
│   ├── __init__.py
│   ├── retrieval.py           ← sentence-transformer embed + kNN
│   ├── rule_based.py          ← keyword → bin lookup heuristic
│   └── majority.py            ← majority-class baseline
│
├── llm_interface/
│   ├── __init__.py
│   ├── prompt_builder.py      ← build system + few-shot + user prompt
│   ├── api_caller.py          ← call OpenAI / Anthropic / local LLM
│   ├── output_parser.py       ← extract JSON, validate, clamp, fallback
│   └── schema.py              ← JSON schema definition + validator
│
├── evaluation/
│   ├── __init__.py
│   ├── param_metrics.py       ← bin accuracy, MAE, within-1-bin
│   ├── closed_loop.py         ← re-run simulator, compare profiles
│   └── report.py              ← generate eval summary tables
│
├── scripts/
│   ├── 01_build_dataset.py    ← CLI: generate text-supervision JSONL
│   ├── 02_run_baseline.py     ← CLI: evaluate retrieval baseline
│   ├── 03_run_llm.py          ← CLI: evaluate LLM predictions
│   └── 04_closed_loop.py      ← CLI: closed-loop sim evaluation
│
└── data/                      ← generated artifacts (gitignored)
    ├── lang2setup_train.jsonl
    ├── lang2setup_val.jsonl
    ├── lang2setup_test.jsonl
    └── embeddings/            ← cached sentence embeddings
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `data_prep/extract_features` | Load one `metadata.json`, return flat feature dict |
| `data_prep/discretize` | Map continuous `x_offset` → `x_bin` given `bins.yaml` |
| `data_prep/describe` | Given features + descriptors → list of NL strings |
| `data_prep/build_dataset` | Walk `outputs/random_5k/`, call above, write JSONL |
| `data_prep/split` | Read JSONL, assign train/val/test, write split files |
| `baselines/retrieval` | Embed corpus, kNN predict, return bin predictions |
| `llm_interface/prompt_builder` | Assemble prompt from template + retrieved examples |
| `llm_interface/output_parser` | `parse(raw_text) → validated dict or fallback` |
| `evaluation/param_metrics` | `evaluate(preds, golds) → metric dict` |
| `evaluation/closed_loop` | `bins → physical params → run simulator → compare` |

---

## 7. Development Roadmap

### Stage A: Generate Text-Supervision Dataset (Week 1)

- [ ] Define `bins.yaml` with bin edges
- [ ] Define `templates.yaml` with 20+ templates and descriptor thresholds
- [ ] Implement `data_prep/` modules
- [ ] Run `01_build_dataset.py` → produce ~15k JSONL records
- [ ] Run `split.py` → train/val/test JSONL files
- [ ] Manual inspection: read 50 random examples, sanity check

**Exit criteria:** JSONL files exist, text descriptions read naturally, bin distribution is reasonable.

### Stage B: Build Non-LLM Baseline (Week 2)

- [ ] Implement `baselines/retrieval.py` with sentence-transformers
- [ ] Implement `baselines/rule_based.py` (keyword matching)
- [ ] Implement `baselines/majority.py`
- [ ] Implement `evaluation/param_metrics.py`
- [ ] Run `02_run_baseline.py` → get accuracy numbers on test set
- [ ] Record baseline numbers

**Exit criteria:** retrieval baseline gives reasonable accuracy (expect ~40–60% joint accuracy with kNN on 121 classes).

### Stage C: LLM with Structured Output (Week 3)

- [ ] Implement `llm_interface/` modules
- [ ] Design system prompt with bin definitions and examples
- [ ] Use retrieval to select 3–5 few-shot examples per query
- [ ] Implement output parsing + validation + fallback
- [ ] Run `03_run_llm.py` on test set
- [ ] Compare to baseline

**Exit criteria:** LLM prediction ≥ baseline accuracy, output parsing succeeds >95% of the time.

### Stage D: Closed-Loop Evaluation (Week 4)

- [ ] Implement `evaluation/closed_loop.py`
- [ ] Convert predicted bins → physical params → run existing simulator
- [ ] Compute centroid error, width error, profile SSIM
- [ ] Run `04_closed_loop.py`

**Exit criteria:** closed-loop metrics correlate with parameter-level metrics; system produces physically meaningful beam profiles from NL input.

### Stage E: Refinements (Week 5+)

- [ ] Add `y_bin` prediction (trivial extension)
- [ ] Add paraphrase augmentation
- [ ] Try finer bins (21 or 41)
- [ ] Add multi-ID support if new optical families are added
- [ ] Consider fine-tuning a small model if API costs are high

---

## 8. Key Design Tradeoffs

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Bins vs continuous | **Bins** | More robust for LLM, easier validation, sufficient resolution |
| 11 bins vs more | **11** | 0.6 mm resolution is fine for ±3 mm range; 121 combos manageable |
| `y` in v1? | **Predict but default to center** | Simplifies to 1D; add later trivially |
| Template vs LLM-generated text | **Template first** | Deterministic, debuggable, no dependency on another LLM |
| Retrieval vs LLM first | **Retrieval first** | Zero cost, fast baseline, informs LLM prompt design |
| Fine-tune vs prompt | **Prompt first** | No training infra needed; fine-tune only if perf insufficient |
| One ID vs many | **One ID** | Single Gaussian family; add IDs when setup diversity grows |

---

## 9. Concrete Next Steps

1. **Create `configs/bins.yaml`** — define bin edges.
2. **Create `configs/templates.yaml`** — write 20 templates + descriptor thresholds.
3. **Implement `data_prep/extract_features.py`** — read one sample's metadata.
4. **Implement `data_prep/discretize.py`** — map continuous → bin.
5. **Implement `data_prep/describe.py`** — generate text descriptions.
6. **Implement `data_prep/build_dataset.py`** — orchestrate over 5k samples.
7. **Run and inspect** — verify dataset quality before any modeling.
