# Project Report: Evaluating and Fine-Tuning an LLM/VLM for Optics Experiment Understanding

**Report date:** 17 July 2026  
**Project scope:** Synthetic single-source, single-lens, camera-based optics setup  
**Primary model:** Qwen2.5-VL-3B-Instruct with local QLoRA training  
**API expenditure:** SGD 0; all generation, labelling, training, and evaluation were local

## 1. Executive summary

The project investigated whether a small LLM/VLM can understand an optics experiment setup rather than merely regress from an observed beam error to a control value.

We built a simulator-grounded benchmark covering seven reasoning tasks, implemented scenario-disjoint splits, simulator-replay ground truth, held-out parameter bands, multimodal examples, matched counterfactual pairs, and strict promotion gates. Across successive experiments, the model improved substantially in JSON/schema compliance, setup interpretation, causal effects, diagnosis, and some counterfactual tasks. However, it did not learn two core discriminative capabilities reliably:

1. deciding whether a target is feasible under a constrained actuator grid; and
2. deciding whether the visible information is sufficient to determine an optical outcome.

The strongest large-data round raised independent-development macro score from 0.468 to approximately 0.691, but all three seeds still failed the control and sufficiency gates. Later focused rounds showed that the aggregate score improvement was partly driven by formatting and easier task families, while the critical decisions remained constant-class or strongly biased.

The latest evidence-grounded v5A experiment removed the earlier hidden-simulator problem by placing residuals and thresholded directions directly in the prompt. Even then, a balanced 50-step trial improved schema validity from 0.760 to 0.885 but reduced feasible-control recall from 0.66 to 0.38; insufficiency recall remained zero. No checkpoint was promoted, and all confirmation and sealed test sets remain unopened.

**Current conclusion:** the benchmark and data pipeline are ready for continued research, but the present 3B model plus ordinary token-level SFT is not yet a reliable optics decision system. The next experiment should supervise explicit intermediate evidence objects or use a deterministic optics tool, rather than simply adding epochs, seeds, or more similar records.

## 2. Initial research question and constraints

The initial question was: **Can an LLM API understand an optics experiment setup, including causal behavior, ambiguity, feasibility, and counterfactual changes?**

The work was constrained by:

- an initial testing budget below SGD 50;
- reduced visual dependence during early experiments;
- no use of paid LLMs to generate or approve ground truth;
- synthetic ground truth derived only from the local Fresnel simulator;
- scenario-level splitting to prevent the same physical setup from appearing across train and evaluation;
- a sealed final test set that could only be opened after a model passed predefined development gates.

The project ultimately used **SGD 0 in paid API calls**.

## 3. Problem decomposition

To prevent the work from becoming a simple control regression task, optics understanding was divided into seven task families:

1. **Setup interpretation and units** — identify components, roles, distances, adjustable fields, and unit conversions.
2. **Information sufficiency** — determine whether the visible fields uniquely determine the requested result.
3. **Causal effects** — predict which measured properties increase, decrease, or stay unchanged after an intervention.
4. **Forward prediction** — predict the post-intervention sensor state.
5. **Diagnosis and ambiguity** — identify all interventions compatible with an observation and distinguish unique, ambiguous, and unsupported cases.
6. **Constrained intervention** — find the minimum-motion successful action on a declared grid or certify infeasibility.
7. **Counterfactual reasoning** — compare paired cases that differ in exactly one physical parameter.

The dataset design also included multiple questions per physical scenario, OOD parameter bands, same-setup opposite-label pairs, ambiguity sets, exhaustive action grids, and simulator verification. These features make it difficult to solve the full benchmark using one scalar regression rule.

## 4. Phase 1 — Pilot dataset implementation

We created the `optics_understanding_sft` package and generated the first complete pilot dataset.

### Dataset composition

- 300 independently sampled physical scenarios;
- 1,200 total records;
- 210/30/60 train/validation/test scenarios;
- 840/120/240 train/validation/test records;
- four questions per physical scenario;
- exactly 10% visual records: 84/12/24;
- half of the test scenarios use held-out parameter bands;
- target-free public test prompts and private test labels;
- deterministic generation with seed 42.

### Ground truth

- setup labels came from deterministic configuration extraction;
- sufficiency labels came from replaying five compatible completions;
- causal and forward labels came from before/after simulator measurements;
- diagnosis replayed candidate interventions and retained every matching cause;
- control exhaustively searched up to 81 action combinations;
- counterfactual pairs held all nuisance variables fixed except one declared parameter.

### Verification

The package included schemas, prompt templates, provider-neutral exports, Qwen exports, leakage checks, checksums, a 28-record smoke fixture, replay audits, and trainer compatibility tests. The unrelated existing `profile2setup` work was preserved.

## 5. Phase 2 — Base-model benchmark

The untouched Qwen2.5-VL-3B model was evaluated on all 120 validation records.

| Task | Base score |
|---|---:|
| Setup interpretation | 0.000 |
| Information sufficiency | 0.450 |
| Causal effects | 0.644 |
| Forward prediction | 0.278 |
| Diagnosis | 0.402 |
| Constrained intervention | 0.487 |
| Counterfactual reasoning | 0.648 |
| **Equal-task macro** | **0.416** |

Additional observations:

- strict-JSON validity was 94.2%;
- complete schema validity was 58.3%;
- setup failures included ontology aliases and unit-scale errors;
- diagnosis predicted `ambiguous` for every record;
- control often used an unsupported status or omitted a valid four-actuator plan;
- only about 26% of parsed forward centroids were within 2 px.

This established genuine learning headroom, but it also showed that format compliance and physical correctness needed to be scored separately.

## 6. Phase 3 — Initial 40-step QLoRA trial

A local NF4 rank-8 QLoRA trial was run for 40 optimizer steps with seed 42. Checkpoints were evaluated at steps 20 and 40.

| Model | Legacy macro | JSON validity | Schema validity |
|---|---:|---:|---:|
| Base | 0.416 | 94.2% | 58.3% |
| Checkpoint 20 | **0.518** | **99.2%** | 90.8% |
| Checkpoint 40 | 0.514 | 98.3% | **93.3%** |

Checkpoint 20 was selected under the original frozen metric. The two checkpoints were statistically close: the paired bootstrap interval for their difference included zero. Lower teacher-forced loss at step 40 did not correspond to the best original macro score.

A later rubric audit corrected scoring artifacts and established rubric v2 for future work. Under v2, step 40 scored 0.518, step 20 scored 0.510, and the base scored 0.392. The historical selection was not rewritten.

## 7. Phase 4 — Targeted v1.1 augmentation

We generated 100 new training-only scenarios and 400 records targeting setup, sufficiency, diagnosis, and control errors. Two short ablations were run: a 20-step continuation and a fresh 40-step run.

The fresh run achieved the highest v2 macro score, **0.538**, but it was not promoted because:

- the bootstrap improvement over the retained reference was inconclusive;
- every control record was predicted feasible;
- every sufficiency record was predicted insufficient;
- every diagnosis record was predicted unique.

This was the first clear evidence that aggregate task scores could rise while the model learned output priors instead of case discrimination.

## 8. Phase 5 — Corrective v2 scale-up

The next revision replaced repetition-based weighting with exact balance and matched opposite-label examples.

### Data

- 500 corrective training scenarios and 2,000 records;
- independent `dev_v2`: 150 scenarios and 600 records;
- final curriculum: 2,462 unique records, including 462 clean pilot anchors;
- exact control and sufficiency status balance;
- balanced diagnosis statuses;
- 10,585/10,585 corrective and 3,180/3,180 development simulator replays passed;
- no cross-dataset ID, seed, group, or image overlap.

### Training and results

Three 200-step continuation runs were completed with seeds 42, 123, and 314.

| Run | Independent-dev macro | Schema |
|---|---:|---:|
| Step-40 reference | 0.468 | 0.938 |
| Seed 42 | 0.689 | 0.983 |
| Seed 123 | 0.691 | 0.983 |
| Seed 314 | 0.694 | 0.982 |

The macro improvement was large, statistically clear, and stable across seeds. However, **zero of three seeds passed promotion**.

The main failed gates were:

- feasible-control recall: 0.083–0.117 versus required 0.60;
- feasible simulator success: 0.067–0.117 versus required 0.40;
- control status F1: 0.410–0.425 versus required 0.60;
- sufficiency status F1: 0.501–0.532 versus required 0.60.

This phase showed that larger balanced data fixed setup and diagnosis behavior but did not fix feasible control or sufficiency.

## 9. Phase 6 — Action-first v3 and schema repair v3.1

### Action-first v3

We generated 320 new physical scenarios and 640 focused records. Control targets serialized the actuator, signed movement, residual, and action validity before the final status. The curriculum added 160 preservation anchors.

The strongest full-development checkpoint had:

- macro score 0.623;
- schema validity 0.842;
- control F1 0.559;
- sufficiency F1 0.407;
- feasible-control recall 0.700;
- infeasible-control recall 0.433;
- feasible simulator success 0.417.

It improved feasible behavior but failed schema, infeasible recall, control F1, and sufficiency F1. It was not promoted.

### Schema repair v3.1

A focused 50-step continuation attempted to repair output structure while preserving the physical decisions. Schema validity reached approximately 0.992 on the partial diagnostic, but:

- checkpoint-25 sufficiency/control F1 was 0.403/0.460;
- checkpoint-50 sufficiency/control F1 was 0.333/0.403;
- checkpoint 50 predicted all inspected insufficient cases as answerable;
- checkpoint 50 predicted 13/14 inspected infeasible controls as feasible.

The intervention repaired syntax rather than understanding. Additional seeds and full-development generation were stopped to conserve compute.

## 10. Phase 7 — Balanced hard pairs v4

We then designed a stricter same-setup decision-repair dataset:

- 960 fresh records;
- 480 same-setup opposite-label pairs;
- exact 1:1 status balance;
- counterbalanced list ordering and visible error features;
- 90 physical scenarios reserved outside training for diagnostic/holdout use;
- seed-42 training capped at 200 steps.

Training used:

1. 100 decision-weighted warm-up steps; and
2. 100 mixed-preservation steps.

All evaluated checkpoints failed identically:

| Checkpoint | Schema | Sufficiency F1 | Control F1 | Feasible recall | Pair-joint |
|---|---:|---:|---:|---:|---:|
| Warm-up 100 | 0.992 | 0.333 | 0.333 | 0.000 | 0.000 |
| Mixed 50 | 1.000 | 0.333 | 0.333 | 0.000 | 0.000 |
| Mixed 100 | 1.000 | 0.333 | 0.333 | 0.000 | 0.000 |

The final model returned the same status for every member of every pair. No additional seeds or holdout evaluations were run.

## 11. Phase 8 — Root-cause analysis of v4

The v4 data was balanced, simulator-correct, and shortcut-controlled, but the prompt did not expose the measurements needed to distinguish paired labels.

Important findings:

- every feasible target began outside the 2 px tolerance;
- feasibility required replaying a hidden nine-action response curve;
- those response curves averaged 3.20 local slope reversals;
- sufficiency depended on hidden per-completion simulator measurements;
- the correct task therefore required internally emulating a high-precision, non-monotonic Fresnel solver.

This changed the project interpretation: v4 was not a clean reasoning test because decisive evidence was absent from the prompt.

## 12. Phase 9 — Evidence-grounded v5 design probe

V5 separated three capabilities:

1. reasoning over visible experimental evidence;
2. analytic paraxial optics intuition; and
3. exact simulator/tool use.

A 400-record design probe was created from already-used training scenarios. It was not treated as an evaluation set. The deterministic visible-evidence solver reproduced all 400 labels, all 200 pair invariants passed, and shortcut baselines stayed below 0.60.

On a 28-record local probe:

- raw centroid representations remained near chance;
- supplying numeric residuals raised base-model control F1 to 0.708 and pair-joint accuracy to 0.429;
- sufficiency remained at F1 0.333 and pair-joint zero, even with explicit thresholded directions;
- repeating the decision rule improved schema but collapsed control again.

This justified one small, fresh v5A trial, but not a larger fine-tune.

## 13. Phase 10 — Fresh evidence-grounded v5A trial

### Data and certification

- 300 new physical scenarios and 1,200 records;
- 200/50/50 train/development/target-free-confirmation scenarios;
- 800/200/200 records;
- a balanced 200-record, 50-scenario trial curriculum;
- 9,600 simulator replays with zero failures;
- zero solver, scaffold, leakage, overlap, non-finite, or pair-invariant failures;
- all 50 project tests passed.

### Training

- fresh rank-8 QLoRA adapter from the base model;
- seed 42;
- ordinary completion loss;
- one pass over 200 records;
- 50-step maximum with checkpoints at 25 and 50.

### Results

| Run | Macro | Schema | Control F1 | Feasible recall | Control pair-joint | Sufficiency F1 | Insufficient recall | Sufficiency pair-joint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.417 | 0.760 | 0.650 | 0.660 | 0.380 | 0.333 | 0.000 | 0.000 |
| Checkpoint 25 | 0.433 | 0.825 | 0.624 | 0.500 | 0.340 | 0.333 | 0.000 | 0.000 |
| Checkpoint 50 | 0.460 | 0.885 | 0.605 | 0.380 | 0.340 | 0.333 | 0.000 | 0.000 |

Neither checkpoint passed any of the nine gates. Training improved format imitation while control discrimination worsened, and sufficiency did not improve at all. Confirmation remained sealed.

## 14. Phase 11 — Intermediate-evidence v6 tool-use trial

V6 moved numerical thresholding and exhaustive action selection into two deterministic, simulator-independent tools. Each v5A example was decomposed into three separately scored stages: tool choice, exact call construction, and tool-result interpretation. The resulting dataset contains 3,600 records: 2,400 train, 600 development, and 600 target-free confirmation records.

The audit reconstructed every target and executed all 1,200 target calls with zero failures. A balanced 480-record seed-42 QLoRA run completed at 120 steps.

- Checkpoint 60 produced invalid, token-capped JSON for 13/13 targeted control calls, making the 0.98 schema gate impossible.
- Checkpoint 120 improved outer formatting, but exact argument construction and executed-tool agreement were both 0/41.
- With 41 confirmed failures among 200 call-construction records, the best possible full-set argument accuracy was 0.795, below the 0.80 gate.
- Neither checkpoint was promoted and confirmation remained sealed.

This localized the current bottleneck: the model can choose the correct tool but cannot reliably extract and preserve the ordered numeric inputs required to call it.

## 15. Overall findings

### What worked

- A reusable, simulator-grounded optics-understanding dataset and evaluation package was built.
- Ground truth is deterministic, replayable, and independent of paid LLM judgement.
- Scenario-level splitting, private labels, OOD bands, pair invariants, and leakage audits were implemented.
- QLoRA and text/visual trainer compatibility were verified.
- Setup interpretation, causal effects, diagnosis, counterfactual reasoning, and output schema improved substantially in the larger-data rounds.
- Promotion gates prevented misleading macro-score improvements from being reported as successful physical reasoning.
- The sealed test was protected throughout the iterative development cycle.

### What did not work

- Repetition-based targeted augmentation changed class priors rather than discrimination.
- More balanced data alone did not solve feasible control or sufficiency.
- Action-first targets improved some feasible behavior but damaged or failed other branches.
- Schema repair fixed JSON structure, not physical understanding.
- Decision-token weighting caused highly confident constant-class behavior.
- Ordinary token-level SFT on evidence-grounded prompts primarily learned predictable schema and copied numeric content.

### Current technical interpretation

There are two separate limitations:

1. **Evidence availability:** v4 required hidden simulator emulation. This was a dataset/task-design problem.
2. **Evidence transfer:** v6 moved arithmetic into deterministic tools, but the model still did not reliably extract and preserve the ordered numerical arguments needed to invoke them.

The second limitation is now the main bottleneck.

## 16. Recommended next step

Do not immediately add more epochs, random seeds, or similar literal-array records. V6 has completed the intermediate-object experiment and shown that long argument copying is the new failure point. The next design should supervise compact source-path and semantic-role mappings:

- residual source path;
- active-actuator motion source path;
- threshold source path;
- allowed-order/tie-break source path;
- completion-delta source path.

A deterministic adapter should resolve these paths into numeric arrays, validate lengths, units, finiteness, and ordering, then call the already implemented decision tools. Each path/role mapping should be scored independently before the complete call is tested.

In parallel, establish a deterministic tool-use baseline:

1. the LLM identifies which optical calculation/tool is required;
2. it supplies compact field mappings rather than copying long arrays;
3. deterministic code constructs validated arrays and performs thresholding or exhaustive search;
4. the LLM interprets the returned result in the experiment context.

Exact non-monotonic Fresnel propagation should remain a simulator/tool-use tier. A separate analytic tier should test genuine optics intuition using paraxial equations and monotonic cases. Real laboratory data will still be required before making any physical-world validity claim.

## 17. Current status

- No adapter is currently promoted as a reliable optics-understanding model.
- The original historical checkpoint remains available as a reference, not as a validated deployment model.
- V5A training stopped exactly at 50 steps.
- V5A confirmation, v4 holdout, and the original 240-record sealed pilot test remain unopened.
- No paid API calls were used.
- The latest full test suite passes 57/57 tests.
- The project is ready for a compact path/role-mapping tool-use phase.
- The first literal-array tool-call trial is complete and rejected; the next version should supervise compact source-path and semantic-role mappings that a deterministic adapter resolves into validated arrays.

## 18. One-minute verbal summary

> We built a simulator-grounded benchmark to test seven types of optics understanding, not just beam-control regression. The initial 3B model scored 0.416, and early QLoRA raised the aggregate score above 0.51. Scaling to a balanced 2,000-record corrective dataset produced a statistically stable macro score around 0.69 across three seeds. However, strict promotion gates showed that the model still failed feasible control and information sufficiency. We exposed calibration evidence and then moved thresholding and exhaustive selection into deterministic tools. The latest 3,600-record tool-use trial showed that the model chooses the correct tool but does not reliably construct its ordered numerical arguments: checkpoint 120 scored 0/41 on exact targeted calls. Therefore, no checkpoint was promoted and all sealed tests remain untouched. The next version should let the model select compact source paths and semantic roles while a deterministic adapter constructs and validates the numeric arrays.
