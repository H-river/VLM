# Active diagnosis v13 — final report

Status: formal Gate A completed after the T+4 guard and froze exactly one
primary path, Branch A. The development-only no-residual-history estimator is
bound in `development/branch_a_resolution.json`; the single protected evaluation
is complete, supports the frozen path, and was not used for reselection.

Main research conclusion: the frozen visible sequential continuation rule is
the strongest development-only mechanism, gaining +21.0 points across five
original-setup planner seeds and +18.06 points across the balanced 2-seed ×
90-fresh-setup matrix, with every seed/cell positive and zero matched
regressions. In contrast, candidate best-of-k coverage has no complementary
recovery stable across all three proposal seeds. The smallest decisive next
experiment uses 30 external groups (ten per stratum) and two preregistered
planner seeds for the unchanged rule versus fixed four, capped at eight with
success, steps, and saturation fixed before execution.

## Locked experimental contract

- Corrected v12 H1 forward checkpoint:
  `runs/overnight_v12_semantics_20260731_002709/models/lc_128g_v2/continuous_forward_v12_128g.pt`
- Hidden physical gain hypotheses: `{0.5, 0.75, 1.0, 1.25, 1.5}`.
- Development groups: suffixes `0000`–`0009` in each of three strata (30 groups).
- Protected groups: suffixes `0010`–`0015` (18 groups), inaccessible before freeze.
- Direct, oracle-known, and probe controllers use candidate-sampling seeds based
  only on root seed and case ID; the first invalid gain-dependent-seed attempt is
  quarantined under `invalid_evidence/true_gain_seed_leak/`.

## Formal Gate A evidence (development; Branch A frozen)

| Component | Current result | Threshold | Status |
|---|---:|---:|---|
| A: direct fault impact | 19.2 pp success drop | 5 pp | pass |
| B: oracle recoverability | 12.5 pp success gain | 5 pp | pass |
| C: safe-probe observability | 78.7% five-class group-OOF | 80% | fail |
| D: probe + estimate + replan | +10.0 pp (95% CI +1.7 to +18.3) | 5 pp | pass |

The formal diagnosis selected Branch A because impact and recoverability pass
while preregistered full-feature observability misses 80%. It records exactly
one selected branch, development-only selection, and no protected use. The
separate Branch-A resolution freezes the audited no-residual-history estimator
at exactly 80.0% OOF accuracy and +10.8-point development control value; its
seven-seed minimum gain is +6.7 points, so every seed clears the +5 criterion.

### Branch disposition

| Branch | Disposition | Evidence and reason |
|---|---|---|
| P — primary active diagnosis | Not selected by the formal gate | The preregistered full-feature probe reached 78.7%, below the approximately 80% observability criterion. |
| A — probe design/observability | **Selected as the one primary branch** | Fault impact, oracle recoverability, and closed-loop value passed; the development-only no-residual refinement reached exactly 80.0% and was frozen before the single protected confirmation. |
| B — execute-only reranking | Investigated as supporting evidence, not selected | Existing nominal top-k had only 2/74 later-decision recoveries and visible OOF residual rerankers regressed overall performance, so no learned reranker was promoted. |
| C — H1/CEM forensics | Executed post-freeze as supporting research | Forward calibration, CEM controls, exact failure mining, temporal horizons, and candidate-coverage bounds identify premature stopping as the strongest tested bottleneck; these results cannot reselect Branch A. |
| T — technical fallback | Not invoked as a primary branch | The verified v12 checkpoint, simulator, Qwen environment, and protected-once workflow all completed; invalid attempts were quarantined rather than substituted. |

The selected development probe is a symmetric `+10%,-10%` command pair. It has
zero final beam displacement, zero saturation, and zero constraint violations,
but its transient disturbance is not zero: mean peak 1.472 tolerances, p95 2.595,
maximum 3.570. This transient cost is reported alongside net disturbance.

## Completed supporting analyses

- All 2,400 probe records are finite, group-complete, and free of protected IDs
  and policy-visible hidden state.
- Linear classifier: 78.7% OOF accuracy, macro-F1 0.784, group-bootstrap 95% CI
  68.0%–88.7%; 84.4% of its errors are adjacent gain classes.
- Small MLP on the identical visible features: 79.3% OOF accuracy, macro-F1
  0.794 and +9.2-point fault success. It does not beat the full linear policy.
- The development-only no-residual-history refinement reaches exactly 80.0%
  OOF accuracy and +10.8-point fault success (95% CI +2.5 to +18.3), with 19
  recoveries and 6 regressions. It is the supported Branch-A refinement, but
  was not formally frozen until the T+4 Gate diagnosis selected Branch A.
- Its remaining control-level observability gap is small but nonzero: hidden-
  gain oracle planning reaches 66.7% fault success versus 65.0% for the visible
  no-residual policy, a matched -1.7-point difference with 95% interval
  `[-4.2,0.0]`. The visible policy recovers no oracle failures and regresses two
  oracle successes, while adding 2.03 steps and 6.0% saturation. Thus the main
  residual cost is probe/estimation overhead rather than large recoverable
  success headroom hidden behind the 80% classification boundary.
- Within the refined policy's 120 fault episodes, the 98 correct estimates
  succeed 70.4% of the time with zero saturation; the 22 incorrect estimates
  succeed 40.9% with 40.9% saturation. Six of nine saturated error episodes
  are severe underestimates at true gains 1.25/1.5. This posthoc slice supports
  a future boundary-aware confidence or discrete/continuous hybrid, not a
  change to the frozen exact classifier.
- Selected linear probe-replan raises non-nominal success from 54.2% to 64.2%,
  with 18 matched recoveries and 6 regressions. Boundary/clipping remains weak:
  57.5% gain accuracy and 42.5% probe-replan success in that stratum.
- Feature ablations: no uncertainty 77.3%; no gain-ratio proxy 78.7%; no residual
  history 80.0% (exploratory, not substituted into the preregistered selection).
- Data-size curve plateaus near 79% from 18 to 24 training groups per fold; more
  examples alone are not yet supported as the main fix.
- Existing nominal H1 top-10 audit: no first-decision success headroom and only
  2/74 later-decision hard-negative recoveries (+2.7 pp), below Gate B.
- Nominal H1 forward calibration on 74 executed transitions: absolute-error /
  uncertainty Spearman 0.677; 1/2/3-sigma coverage 67.6/87.3/92.7%; centroid-x
  appears in 5/7 strict final failures.
- A follow-up group-OOF residual calibrator on all 740 retained predicted-top
  candidate records is negative. Uncorrected normalized residual MAE is 0.330;
  a training-fold constant correction worsens it to 0.364 and a visible-feature
  Ridge correction to 0.911. Original/constant/Ridge reranking strict success
  is 31.1/27.0/24.3% over 74 decisions, and actual-best selection is
  55.4/43.2/45.9%. Constant/Ridge produce 7/14 improved decisions but 20/23
  regressions. Even on the 28 decisions from the seven exact nominal failure
  episodes, constant correction changes strict success only 3.6→7.1% with
  essentially unchanged mean actual cost. Bias calibration is not a safe
  candidate-order fix, so H1 remains frozen.
- Across three CEM seeds, full-linear control value remains positive at
  +6.7/+10.0/+11.7 points; mean is +9.4 points. Fault impact and oracle recovery
  also pass in every seed.
- Across five planner seeds, the no-residual Branch-A refinement remains at
  exactly 80.0% gain accuracy and yields +10.8/+6.7/+10.8/+15.8/+15.8 points
  of control value (mean +12.0; planner-seed bootstrap 95% +9.2 to +14.8);
  every seed clears 5 points. A deterministic report refresh
  preserves both its binary SHA-256 and all 150 OOF/full predictions exactly.
  Relative to the full feature model it changes eight OOF predictions (three
  wrong-to-right, one right-to-wrong), raising each interior stratum from 90%
  to 92% while leaving the boundary/clipping stratum at 56%. Thus it clears the
  aggregate gate without resolving the main boundary observability weakness.
- Extending the same matched comparison to seven planner seeds adds +10.0 and
  +17.5 points. The seven-seed mean is +12.5, minimum +6.7, standard deviation
  3.9, and planner-seed bootstrap 95% +9.8 to +15.1; every seed remains above
  +5 and exact gain accuracy remains 80.0%.
  The fully matched seven-seed Gate decomposition also keeps fault impact above
  +13.3 points in every seed (mean +20.5, seed-bootstrap 95% +17.3 to +23.5)
  and oracle recovery above +9.2 in every seed (mean +13.7, 95% +11.4 to
  +15.8). Thus A, B, and refined D are individually robust to planner seed.
- A final eighth development seed adds +27.5-point fault impact, +19.2-point
  oracle recovery, and +18.3-point refined control value at the same 80.0%
  accuracy. Across eight seeds, impact averages +21.4 points (95% +18.1 to
  +24.4), recovery +14.4 (+12.0 to +16.7), and refined control +13.2
  (+10.5 to +15.8); every seed still clears +5. The formal freeze continues
  to reference the already-audited seven-seed artifact, while the eighth is
  supporting robustness that cannot change selection.
- Repeating the full feature-ablation search at safer symmetric magnitudes does
  not find a smaller passing probe: the best 1%, 2%, and 5% accuracies are
  66.7%, 70.0%, and 78.0%, respectively. The 10% symmetric/no-residual policy
  is therefore the only tested zero-net-displacement symmetric arm at 80%.
- The strict 16-cell closed-loop matrix is complete at 150 episodes per cell.
  All probe executions have zero probe-time saturation and zero constraint
  violations. Symmetric 1%/2%/5% probes restore exactly to zero final beam
  disturbance and yield +13.3/+14.2/+12.5 points of fault success, but their
  full-feature gain accuracies are only 65.3/65.3/76.0%; they do not pass the
  Branch-A observability criterion. The repeated and one-sided designs reach
  at most 76.7% and 75.3% accuracy and retain nonzero final disturbance. This
  separates useful coarse gain information from the exact five-hypothesis
  observability required by the registered branch rule. A compact CSV records
  accuracy, SNR, transient/final disturbance, violations, success, intervals,
  saturation, and step cost for every design.
- A matched 2%-versus-refined-10% comparison makes the constraint tradeoff
  explicit. The 2% probe has only 65.3% exact gain accuracy but 68.3% fault
  success; the refined 10% probe has 80.0% accuracy but 65.0% fault success.
  The refined-minus-2% difference is -3.3 points with 95% interval
  `[-9.2,+2.5]`. Both restore net displacement and use the same mean steps;
  2% has one-fifth the transient disturbance but 10.7% versus 6.0% control
  saturation. The formal 10% choice follows the registered observability rule,
  not a claim of control-only dominance; 2% remains a coarse-diagnosis ablation.
- A forced-nominal control isolates the apparent strength of the one-sided 1%
  negative probe. Its physical treatment alone changes fault success by -0.8
  points (95% CI -3.3 to +1.7), while using its estimator adds +13.3 points
  over forced nominal (95% +5.8 to +20.8). The benefit is diagnostic rather
  than a lucky control displacement, but 60.7% accuracy and nonzero 0.147-
  tolerance final disturbance keep it out of the frozen safe-probe choice.
- The development-selected confidence threshold 0.8 was replayed end to end and
  matches its offline projection on all 150 outcomes: 65.0% fault success,
  +10.8 points (95% CI +4.2 to +17.5), 15 recoveries, and only 2 regressions.
  It is a useful risk-control result but does not replace the no-residual model:
  its effective post-fallback gain accuracy is 57.3% and control saturation is
  17.3%, versus 80.0% and 6.0% for no-residual control.
- A secondary no-residual posterior-mean estimator lowers OOF gain RMSE from
  0.154 to 0.140 and raises development fault success to 68.3%, or +14.2 points
  over direct (95% +7.5 to +20.8). Against the exact no-residual classifier its
  matched gain is only +3.3 points with a 95% interval of -2.5 to +10.0, while
  saturation rises from 6.0% to 22.7%. Its continuous output has no exact
  five-class accuracy, so this promising control ablation cannot replace the
  exact-observability freeze candidate and remains development-only.
  Its matched failure taxonomy localizes the apparent upside: +12.5 points in
  multi-step interior cases and +6.7/+10.0 points at gains 1.25/1.5, but -2.5
  points on boundary/clipping and -3.3 at gain 0.5. Fault-case saturation rises
  from 7.5% to 28.3%, reaching 70% at gain 1.5. This suggests a future
  boundary-aware continuous/discrete hybrid, not a safe substitution now.
- Non-loading Qwen preflight passes: the local 3B model has two weight shards,
  the RTX 4080 Laptop GPU reports 10.6 GiB free before load, and Torch 2.5.1,
  Transformers 5.8.1, PEFT 0.19.1, TRL 1.4.0, bitsandbytes 0.49.2, Datasets
  4.8.5, and Accelerate 1.13.0 import together. No weights or data were loaded
  and no training was started before the formal branch.
- Qwen long-job behavior is prepared for incremental evidence: training logs
  every step and checkpoints every 15 of at most 60 steps; smoke output is
  isolated from the full adapter. Inference fsyncs each row and is now bound by
  an atomic resume manifest over the input JSONL, adapter root-file tree, and
  base config/index hashes. Closed loop recomputes each visible-prompt hash.
  Changed adapters/inputs, unbound stale rows, duplicates, and out-of-dataset
  rows are rejected. At that Qwen checkpoint the v13 suite passed 51/51; the
  final expanded source suite passes 102/102.
- Post-freeze Qwen smoke and capped training both completed cleanly on the
  reduced 120-row training set. Sixty optimizer steps took 198 seconds;
  validation token loss was 0.0405/0.0294/0.0325/0.0297 at steps
  15/30/45/60, and the final adapter was evaluated without checkpoint
  selection. On all 30 rows from six disjoint held-out development groups,
  structured JSON and gain parse rates are 100%, but exact gain accuracy is
  only 76.7%. All seven errors are adjacent underestimates in the boundary/
  clipping stratum, and the generated confidence is identically 1.0, so it is
  not calibrated. On the 24 fault rows, direct/frozen-linear/reduced-MLP/Qwen
  strict success is 54.2/66.7/62.5/62.5%. Qwen gains +8.3 points over direct
  but is -4.2 points versus frozen linear; both matched intervals include zero.
  Thus the 3B model is technically viable and control-useful, yet supplies no
  accuracy or closed-loop advantage over the cheap numeric baselines here.
  The held-out stratum slice is sharper: Qwen and the MLP reach 100% exact gain
  accuracy and 75% success on multi-step fault rows, and 100%/100% on one-step
  rows, but only 37.5% accuracy and 12.5% success on boundary/clipping. Frozen
  linear has lower 25% boundary accuracy yet 25% success. On this small slice,
  exact class accuracy is therefore not monotonically aligned with control.
- A transparent post-Qwen development ablation combines the frozen classifier
  and its posterior mean: keep the discrete gain when that visible estimate is
  below 1.0, otherwise use the mean. Exact replay validates the switch source
  on all 150 episodes (84 continuous, 66 discrete). Fault success reaches
  70.8%, +16.7 points over direct (95% +10.8 to +23.3) and +5.8 over frozen
  discrete (95% +0.8 to +11.7), with seven fault recoveries and no fault
  regressions. It also adds +2.5 points over the unguarded posterior mean.
  This success is not free: fault saturation is 28.3% versus 7.5% discrete,
  reaching 43.3%/70.0% at true gains 1.25/1.5. The hybrid is therefore a
  promising base-seed success/risk result, not a replacement for the protected
  freeze.
  A cached matched frontier shows the base-seed hybrid can be made strictly
  safer without losing success: require the posterior mean to lie no more than
  0.125 below the discrete estimate. This keeps 70.8% fault success and the
  +5.8-point gain over discrete while lowering fault saturation from 28.3% to
  25.8%. A 0.10 margin yields exactly +5.0 points at 24.2% saturation. These
  are development Pareto points, not protected selections. Across eight
  planner seeds, the unbounded and 0.125-margin policies both remain above the
  registered +5-point threshold versus direct (mean +14.6, minimum +8.3), but
  the incremental result versus frozen discrete does not replicate strongly:
  margin 0.125 averages only +1.35 points with seed-bootstrap 95%
  `[-0.21,+2.92]` and regresses by 1.67/2.50 points on seeds 7/8. Mean fault
  saturation is 25.3% versus 7.8% discrete, amounting to 168 net new
  saturation events for 13 net recoveries across seed-by-episode units. All
  1,050 cross-seed probe-feature and gain-belief comparisons are bit-exact, so
  this reversal is planner variation rather than estimator drift. The hybrid
  hypothesis is therefore closed as a useful negative result: it beats direct
  because it inherits the frozen diagnosis, but does not robustly beat frozen
  discrete enough to justify the added actuator risk.
- Direct-CEM uncertainty weights 0, 0.1, and 0.25 have identical fault success.
  This equality is not merely caused by identical plans: weights 0 and 0.25
  alter 15 and 17 of 150 episode plans (12 and 14 fault episodes), with maximum
  command differences near 0.028 mm, yet all 120 matched fault outcomes remain
  identical. Weight zero recovers one nominal episode and weight 0.25 regresses
  one nominal episode. The tested risk penalty is behaviorally active but has
  no observed fault-outcome leverage.
  Doubling population from 24 to 48 produces a smaller +3.3-point change with
  a matched 95% interval spanning zero. Population 96 and five CEM iterations
  plateau at the same +3.3 points, while population 12 collapses by 20.8 points.
  Replication on planner seeds 2 and 3 changes that population-48 effect to
  -2.5 and +2.5 points; the three-seed mean is +1.1 with seed-bootstrap 95%
  `[-2.5,+3.3]`. Boundary success changes +5.0/-7.5/0.0 points across the
  three seeds. The sampler needs a minimum budget, but brute-force budget
  beyond the frozen setting is neither large nor directionally robust and is
  not supported as a >=5-point fix.
- A controlled CEM elite-diversity treatment is behaviorally effective but
  harmful. Enforcing minimum normalized elite distance 0.25 raises the observed
  minimum from mean 0.185 to 0.309 and satisfies the constraint in every one of
  1,407 treated planner iterations. The zero-distance arm exactly reproduces
  all frozen direct commands, outcomes, and final distances. Yet treated fault
  success falls 54.2%→49.2%, a matched -5.0 points (95% `[-10.0,0.0]`) with
  one recovery and seven regressions; boundary success falls 10 points while
  saturation falls only 3.3. Candidate collapse within the tested elite rule
  is therefore not the bottleneck, and forced diversity degrades ranking.
- A matched feasible-proposal treatment removes the remaining sampled
  out-of-feasible mass (10.5% to zero) and reduces fault saturation from 41.7%
  to 37.5%, but changes fault success only 54.2%→55.8%, +1.7 points with 95%
  `[-3.3,+6.7]`, from nine recoveries and seven regressions. It gains five
  points in the one-step stratum and exactly zero in both multi-step and
  boundary/clipping. Unique effective sequences were already 100%. Together
  with the risk, population, and diversity controls, this closes broad CEM
  knob tuning without a robust boundary fix.
- On the exact 20 boundary episodes failed by direct and oracle-known control,
  nested evaluator-only two-step best-of-k coverage at budgets 8/24/48/96/192
  is 0/1/4/7/8 successes for both uniform-feasible and boundary-conditioned
  proposals. Their success IDs are identical: boundary conditioning adds no
  recoveries. Against the eight-step temporal mechanism, the maximum candidate
  union has 4 shared recoveries, 4 candidate-only, 3 temporal-only, and 9
  neither. Thus broader sequence coverage has a complementary +3.3-point fault
  upper bound beyond temporal eight, but no visible chooser is established.
  Because the complete union count 8 exceeded the preregistered threshold 5, a
  distinct proposal seed capped at budget 96 was run. Its curve is 0/2/3/5 and
  again identical across proposal families, versus 0/1/4/7 for seed 1 through
  the common budget. Both preregistered uncertainty triggers launched a third
  capped proposal seed; its curve is 0/0/1/3. At budget 96 the per-seed unions
  are 7/5/3, only one ID is stable among ten recovered by any seed, stable/any
  is 0.10, and pairwise recovery Jaccard is 0.111–0.333. Six cases recover on
  one seed, three on two seeds, and one on all three. The single all-seed case
  is already recovered by temporal horizon eight, so no complementary proposal
  recovery is stable across all three seeds; the minimum per-seed candidate-
  union fault upper bound is +2.5 points. Uniform and conditioned success IDs
  remain identical; seed-3 conditioning reaches one episode and 0.52% of
  candidates. This measures sampling instability and cannot convert the
  evaluator bound into deployable-policy evidence. An independent command-only
  replay of eight stored sequences (16 simulator steps) reproduces every
  terminal distance to machine precision and every success flag exactly.
- Exact development failure mining identifies 20 nonnominal boundary episodes
  failed by both direct and oracle-known control. Eighteen improve monotonically
  through the fourth/final control step. Oracle gain reduces their normalized
  target distance by median 74.8%, yet they stop at median 8.99; normalized
  forward error has only 0.073 correlation with actual step improvement. A
  matched causal extension recovers 6/20 by six steps and 7/20 by eight, with
  zero saturation and bit-exact replay of all first-four traces. The conditional
  six-step effect is selected on baseline failure and is not an unconditional
  policy estimate.
  The full matched result confirms that temporal truncation is the largest
  tested post-freeze bottleneck. Six controls raise direct/oracle/frozen-probe
  fault success by +21.7/+15.8/+15.8 points, with 30-group bootstrap intervals
  `[+15.0,+29.2]`, `[+6.7,+26.7]`, and `[+6.7,+26.7]`, respectively, no
  regressions, and 150/150 exact four-step trace prefixes in every arm. The
  frozen probe moves 65.0→80.8% fault success and 45.0→60.0% boundary success;
  mean controls rise 2.98→3.58 while fault saturation remains exactly 7.5%.
  A group-OOF visible rule—continue after an unsuccessful fourth control only
  when observed normalized distance has not worsened—retains +12.5 points
  (95% `[+5.0,+21.7]`), 77.5% fault and 60.0% boundary success, and unchanged
  saturation. The modal rule is now held fixed for seed replication; none of
  this post-freeze evidence can alter the protected result.
  The held rule replicates without threshold changes on four additional planner
  seeds. Fixed-six frozen-probe gains are +15.8/+17.5/+15.0/+14.2/+15.8 points
  on seeds 1–5, while held-rule gains are +15.8/+16.7/+15.0/+13.3/+15.8;
  every seed is positive and there are zero regressions. A two-way bootstrap
  that independently resamples the five planner seeds and thirty setup groups
  estimates the held-rule gain at +15.3 points with 95% `[+7.5,+24.2]`
  (fixed six +15.7, `[+8.0,+24.5]`). The rule retains 92/94 fixed-six
  recoveries; its two misses are `v12_mpcdiag_primary_02_0004__g1.5` on seed 2
  and `v12_mpcdiag_primary_02_0004__g1.25` on seed 4. Of 600 serialized
  fault-seed records, 27 exact episodes fail at four steps on all five seeds,
  25 are recovered whenever they fail, and 18 are never recovered by step six
  on any observed seed. This makes premature stopping the strongest robust
  post-freeze mechanism tested so far.
  Exact base-seed horizons 4/5/6/7/8 show that the effect extends beyond six.
  Frozen-probe fault success is 65.0/75.0/80.8/84.2/87.5%, with boundary
  success 45.0/50.0/60.0/60.0/65.0%; steps five through eight recover
  12/7/4/4 fault episodes and never regress an earlier success. Fault
  saturation stays exactly 7.5%, while mean controls rise 2.95→3.82. A
  transparent sequential rule rechecks the same visible evidence after every
  unsuccessful step four through seven. Seed-1 group-OOF chooses minimum
  observed improvement 0.25 and no distance cap in 29/30 folds (0.5 in one),
  yielding 84.2% fault success. The frozen full seed-1 rule reaches 85.8%,
  +20.8 points over fixed-four, retains 25/27 fixed-eight recoveries, uses 3.66
  mean controls versus 3.82 fixed-eight, keeps 7.5% saturation, and causes zero
  regressions. Held unchanged on seeds 2–5, the rule gains
  +24.2/+20.0/+20.8/+19.2 points. Across all five seeds the two-way seed/group
  estimate is +21.0 points with 95% `[+10.2,+32.8]`; all 126 recoveries have
  zero regressions. Fixed horizon eight gains +22.5/+25.8/+22.5/+25.0/+21.7
  points over four and +6.7/+8.3/+7.5/+10.8/+5.8 over six. Thus every seed's
  best tested 4/6/8 success remains at horizon eight, while the visible rule
  retains most of that headroom with fewer controls. Thirteen of 42 episodes
  ever recovered by the rule are recovered on all five planner seeds
  (stable/any 0.31), versus 1/10 (0.10) for proposal coverage. Seeds 4/5 were the
  preregistered final extension and cannot reselect the rule. The stability
  ratios are descriptive rather than a shared-denominator test because the
  mechanisms, eligible failure sets, and numbers of seeds differ.
- A one-shot fresh-suite check uses 30 newly generated simulator setups, ten in
  each diagnostic stratum, with zero training/original-suite group-ID or setup-
  hash overlap. Without estimator or rule retuning, fixed-4/fixed-6/fixed-8/
  frozen-sequential fault success is 75.0/85.8/91.7/90.0%; boundary/clipping
  success is 55.0/65.0/75.0/70.0%. The frozen sequential rule gains +15.0
  points over fixed four with a 30-group bootstrap 95% `[+5.8,+25.8]`, 18
  recoveries, zero regressions, unchanged 7.5% fault saturation, and 3.23 mean
  controls versus 3.43 for fixed eight. It retains 18/20 fixed-eight
  recoveries. A second mutually group/hash-disjoint 30-setup suite independently
  gains the same +15.0 points (95% `[+6.7,+24.2]`) with 18 recoveries and zero
  regressions. Pooled over all 60 independent setup groups, the frozen-rule
  gain is +15.0 points with 95% `[+8.3,+22.1]`, 36 recoveries, and no
  regressions; suite-level gain-classification accuracy is 77.3% and 80.0%.
  A balanced second planner seed is complete on both suites. Suite A motivated
  that expansion, but it did not change the frozen model or rule, and the full
  seed-by-setup analysis was fixed before suite B completed. Across the 2×60
  seed/setup matrix, seed-level gains are +15.0/+16.25 points, all four suite/
  seed cells are positive, and the two-way estimate is +15.63 with 95%
  `[+8.96,+22.92]`; 75 recoveries have zero regressions. Exact frozen-rule
  failure outcomes agree on 98.33% of 240 fault episodes, with 16 failures
  stable across seven setup groups and only four planner-seed-discordant IDs.
  A posthoc cohort-shift audit prevents overinterpretation: fresh A/B fixed-four
  success is 10/15 points higher than the original cohort and median initial
  normalized distance is 6.94/9.41 lower. The fresh rule gain is consequently
  5.83 points smaller than the original seed-1 gain, although it remains +15.0
  in each fresh suite. Boundary/clipping gains are +15/+25 points in fresh A/B
  versus +15 in the original cohort, so the boundary mechanism replicates even
  though overall fresh difficulty is lower. This is same-simulator, new-setup
  replication, not equal-difficulty or hardware validation.
- To address that measured difficulty shift without outcome selection, a third
  cohort was assigned from 90 newly generated excluded setup candidates using
  only stratum and `log1p(initial normalized distance)`. It contains 30 unique
  setup hashes, zero original/fresh-suite ID or hash overlap, and ten groups per
  stratum. Overall median initial distance is 22.97 versus 23.23 in the original
  development cohort; the three stratum-median differences are -0.79, -0.07,
  and +0.84. Mean absolute log-distance mismatch is 0.0476 and maximum mismatch
  is 0.2928. Candidate outcomes, protected covariates/results, estimator labels,
  and sequential-rule outcomes are absent from matching. On planner seeds 1/2,
  fixed-four success is 67.5/70.0% and the frozen sequential rule reaches
  91.67/91.67%, gains of +24.17/+21.67 points with intervals
  `[+12.5,+36.67]` and `[+10.83,+33.33]`, 55 recoveries, and zero regressions.
  Boundary success rises 45.0→82.5/77.5%. Across all three fresh suites and
  both seeds, all six cells are positive; the balanced 2×90 two-way estimate is
  +18.06 points with 95% `[+12.22,+24.44]`, 130 recoveries, and zero
  regressions. Failure outcomes agree on 96.67% of 360 episodes per seed, with
  22 stable failures across ten groups and 12 planner-seed-discordant IDs.
  Within the matched suite specifically, both seeds leave ten failures but only
  six IDs are stable and eight are discordant (93.33% outcome agreement); stable
  aggregate value therefore coexists with meaningful episode-level variation.
- The first nominally labeled frozen-probe six-step attempt actually bound the
  earlier formal full-feature classifier SHA `2fd9f58f...`; strict replay found
  three changed gain beliefs. All 150 rows are preserved with a manifest under
  `invalid_evidence/wrong_probe_model/` and excluded. The corrected arm binds
  Branch-A SHA `f9e826e9...` and replays all 150 prefixes exactly.
- The post-freeze reduced decision dataset is development-only and leakage-free:
  120 training rows from 24 groups and 30 held-out rows from 6 groups, with no
  group overlap, balanced gain labels, randomized gain-independent candidate
  ordering, and no hidden gain in prompts. Its 32 numeric features are exactly
  the frozen estimator's retained feature indices. The retrained cheap linear
  baseline reproduces 80.0% group-OOF accuracy; a small MLP reaches 78.0%, so
  the extra nonlinearity does not improve the offline estimator.
- The one allowed protected evaluation completed 90 unique matched episodes per
  arm with no errors or reselection. On the 72 non-nominal episodes, direct
  succeeds at 59.7%, oracle-known at 72.2%, and the frozen probe policy at
  72.2%. The probe's +12.5-point control value has 12 recoveries, 3 regressions,
  and group-bootstrap 95% `[+1.4,+25.0]`; its exact gain accuracy is 77.8%.
  Protected fault impact is +18.1 points and oracle recovery is +12.5 points,
  both with positive matched intervals. Across all 90 episodes, direct/oracle/
  probe success is 63.3/73.3/73.3%; probe saturation is 4.4% overall. This is
  confirmatory evidence for the frozen development decision, not a new selector.
- Posthoc protected taxonomy (explicitly no reselection) shows that the frozen
  probe recovers 11 oracle-recoverable fault episodes, one additional episode
  beyond the oracle policy, regresses three, and leaves 17 failures that even
  oracle-known gain does not solve. Multi-step fault success rises from 45.8%
  direct to 70.8% probe; boundary/clipping changes only 41.7% to 45.8% and
  contains 11/17 oracle-impossible failures. Non-nominal exact gain accuracy is
  80.6%. Incorrect-estimate episodes are confounded toward easy high-gain cases
  and still succeed at 92.9%, but 28.6% saturate versus zero saturation when the
  estimate is correct; error-conditioned success is therefore not causal.

## Ranked bottlenecks and smallest next experiments

| Rank | Bottleneck | Best supported gain | Smallest decisive next experiment |
|---:|---|---:|---|
| 1 | Four-step stopping truncates still-improving control | +21.0 pp across five original-setup planner seeds; +18.06 pp across the balanced 2×90 fresh setup/seed matrix | Use 30 external groups (10 per stratum) and two preregistered planner seeds for a paired identical-rule versus fixed-four test, capped at eight with success, steps, and saturation fixed in advance. |
| 2 | Residual boundary failures need action-sequence coverage | +3.3 pp seed-1 evaluator upper bound beyond temporal eight; zero all-three-seed-stable complementary cases | Train one visible proposal model from simulator-best sequence labels and require held-group coverage gain before changing CEM or H1. |
| 3 | Visible residual calibration does not repair candidate ordering | 0 pp supported; both tested calibrators regress | Collect new boundary transitions and test forward-model retraining on held groups; do not deploy either reranker. |
| 4 | Continuous gain compensation adds actuator risk without robust incremental value | +1.35 pp mean versus frozen discrete, with 95% interval crossing zero | Require an explicit success/saturation utility and a new held-group safety result before reconsidering continuous compensation. |
| 5 | Larger Qwen capacity is not the limiting observable | -4.2 pp versus frozen linear on the matched held-out fault slice | Add genuinely observable boundary evidence before any further language-model scaling. |

Ranks summarize post-freeze development evidence and cannot alter the formal
Branch-A freeze or protected-once conclusion. Rank 1 is supported by five
planner seeds on the original groups, two planner seeds on 90 independent fresh
setups, and the outcome-free difficulty-matched cohort; all are same-simulator
evidence and do not substitute for external or hardware validation.

## Exact residual failure IDs

The base-seed frozen probe still fails these 15 nonnominal episodes at fixed
horizon eight:

```text
v12_mpcdiag_primary_01_0005__g0.5
v12_mpcdiag_primary_02_0000__g0.5
v12_mpcdiag_primary_02_0000__g0.75
v12_mpcdiag_primary_02_0002__g0.5
v12_mpcdiag_primary_02_0004__g0.5
v12_mpcdiag_primary_02_0004__g0.75
v12_mpcdiag_primary_02_0005__g0.5
v12_mpcdiag_primary_02_0006__g0.5
v12_mpcdiag_primary_02_0006__g0.75
v12_mpcdiag_primary_02_0006__g1.25
v12_mpcdiag_primary_02_0006__g1.5
v12_mpcdiag_primary_02_0008__g0.5
v12_mpcdiag_primary_02_0008__g0.75
v12_mpcdiag_primary_02_0008__g1.25
v12_mpcdiag_primary_02_0008__g1.5
```

Of the 20 episodes failed by both four-step direct and oracle-known control,
these nine are recovered by neither frozen-probe temporal horizon eight nor the
maximum seed-1 candidate-proposal union:

```text
v12_mpcdiag_primary_02_0002__g0.5
v12_mpcdiag_primary_02_0005__g0.5
v12_mpcdiag_primary_02_0006__g0.5
v12_mpcdiag_primary_02_0006__g0.75
v12_mpcdiag_primary_02_0006__g1.25
v12_mpcdiag_primary_02_0006__g1.5
v12_mpcdiag_primary_02_0008__g0.5
v12_mpcdiag_primary_02_0008__g0.75
v12_mpcdiag_primary_02_0008__g1.25
```

`final_synthesis/final_research_decision.json` is the authoritative exhaustive
machine-readable inventory: it also records direct horizon-eight failures,
candidate-only and neither-proposal partitions, every fresh-suite sequential
failure, stable cross-seed fresh failures, and planner-seed-discordant failures.
The final refreshed inventory contains 22 stable failures across ten groups and
12 planner-seed-discordant IDs over the complete three-suite matrix.

## Active jobs

- No simulator, model-training, or protected job remains active.
- Proposal seed 3 completed all 20 selected boundary oracle failures with the
  fixed budget-96 cap. Its schema-v2 audit binds root seed `2026080103`; the
  final three-seed artifact preserves seed 1's explicit schema-v1 provenance
  limitation rather than inferring a serialized root field.
- All probe matrices, Gate replications, temporal horizons, fresh suites,
  estimator/Qwen paths, guarded hybrids, and broad CEM controls are complete.
  The full current source suite passes 102/102, all 153 Python sources compile,
  and the clean-process command audit passes 166/166. JUnit records zero
  failures/errors. The strict progress audit covers the full 12-hour budget,
  and the terminal deliverables audit binds this final report.
- Split, protected-once, branch-resolution, and frozen-model validators enforce
  exact group bounds, unique records, non-reselection, and binary SHA binding.
- A pre-freeze source snapshot records 117 exact files across v13, the v12
  controller, simulator source, and optics adapters; branch `v13`, commit
  `aa7e1f66`, and aggregate source-tree SHA-256
  `3d276e073544ca290c5c04115647c716433dd718f9fa616f487ac772ae301656`.
  It records that formal/protected artifacts were absent and does not traverse
  the protected path. The separate post-freeze snapshot binds 165 completed
  source files to SHA-256
  `7abe066e282fec549d1b932eb985fecf795c03eb8b417c44934a5d54a0f0f295`
  without traversing protected outputs or overwriting pre-freeze evidence.

## Early-completion queue

All nine registered items are complete on development evidence: seven-seed
expansion; bootstrap intervals; top-k coverage; feature ablations; uncertainty
calibration; 69 serialized hard negatives; exhaustive categorization of all
150 matched episodes (including all 120 non-nominal episodes); 15/15 exact
behavioral replays; and 166/166 clean-process reproduction-command checks. The
machine-readable ledger records paths, hashes, and concise outcomes in
`development/early_completion_queue_status.json`.

An earlier integrity sweep parsed all 47 JSON files and 9,984 rows across 55
JSONL files (143.9 MB total), recursively rejected non-finite numbers, checked
within-file record-ID uniqueness, rejected protected suffixes, and checked for
temporary writes. It passed with zero errors and no live research process at
that checkpoint. The expanded post-candidate sweep also passes: 138 JSON files,
111 JSONL files, 18,471 rows, and 283.37 MB of machine-readable development
evidence all parse, remain finite and unique within file, contain no protected
suffixes, and leave no temporary writes.

All 25 final PNG artifacts are RGB/RGBA and at least 1620×810 pixels. A
nine-figure contact sheet plus the protected-confirmation, five-seed temporal,
2×90 fresh-setup, difficulty-matching, and three-seed candidate figures were
visually inspected for label legibility, clipping, and layout defects.

## Reproduction commands

The authoritative command ledger is `commands/exact_commands.md`. Its latest
clean-process audit passes all 166 documented Python-module commands by invoking
their help paths and checking every long option without rerunning experiments.
The key final-analysis commands, run from `/home/jiamo/VLM`, are:

```bash
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m active_diagnosis_v13.analyze_sequential_horizon_rule --seed seed_2026080101=runs/active_diagnosis_v13_20260801_010051/development/control/probe_nores_budget8.jsonl --seed seed_2026080102=runs/active_diagnosis_v13_20260801_010051/development/control/probe_nores_budget8_seed_2026080102.jsonl --seed seed_2026080103=runs/active_diagnosis_v13_20260801_010051/development/control/probe_nores_budget8_seed_2026080103.jsonl --seed seed_2026080104=runs/active_diagnosis_v13_20260801_010051/development/control/probe_nores_budget8_seed_2026080104.jsonl --seed seed_2026080105=runs/active_diagnosis_v13_20260801_010051/development/control/probe_nores_budget8_seed_2026080105.jsonl --selection-seed seed_2026080101 --output runs/active_diagnosis_v13_20260801_010051/development/sequential_horizon_rule_five_seed.json
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m active_diagnosis_v13.analyze_fresh_holdout_seed_setup_robustness --artifact suite_a:2026080101=runs/active_diagnosis_v13_20260801_010051/development/frozen_sequential_fresh_holdout.json --artifact suite_a:2026080202=runs/active_diagnosis_v13_20260801_010051/development/frozen_sequential_fresh_holdout_seed_2026080202.json --artifact suite_b:2026080101=runs/active_diagnosis_v13_20260801_010051/development/frozen_sequential_fresh_holdout_b.json --artifact suite_b:2026080202=runs/active_diagnosis_v13_20260801_010051/development/frozen_sequential_fresh_holdout_b_seed_2026080202.json --artifact suite_matched:2026080101=runs/active_diagnosis_v13_20260801_010051/development/frozen_sequential_fresh_holdout_matched.json --artifact suite_matched:2026080202=runs/active_diagnosis_v13_20260801_010051/development/frozen_sequential_fresh_holdout_matched_seed_2026080202.json --output runs/active_diagnosis_v13_20260801_010051/development/frozen_sequential_fresh_holdout_seed_setup_robustness.json
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m active_diagnosis_v13.analyze_boundary_candidate_seed_robustness --audit seed_2026080101=runs/active_diagnosis_v13_20260801_010051/development/boundary_candidate_coverage_audit.json --audit seed_2026080102=runs/active_diagnosis_v13_20260801_010051/development/boundary_candidate_coverage_seed_2026080102_audit.json --audit seed_2026080103=runs/active_diagnosis_v13_20260801_010051/development/boundary_candidate_coverage_seed_2026080103_audit.json --output runs/active_diagnosis_v13_20260801_010051/development/boundary_candidate_coverage_three_seed.json
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m active_diagnosis_v13.synthesize_final_research_decision --run-dir runs/active_diagnosis_v13_20260801_010051 --output-dir runs/active_diagnosis_v13_20260801_010051/final_synthesis
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m active_diagnosis_v13.audit_reproduction_commands --commands-markdown runs/active_diagnosis_v13_20260801_010051/commands/exact_commands.md --output runs/active_diagnosis_v13_20260801_010051/development/reproduction_command_audit.json
```

The ledger also preserves every acquisition command, root seed, frozen-model
path, source audit, progress audit, and final-deliverables audit; it is the
canonical source for exact end-to-end reproduction rather than this compact
analysis subset.

## 12-hour timeline

Budget window: `2026-08-01T01:00:51+08:00` through
`2026-08-01T13:00:51+08:00`.

- T+0–T+1: provenance inventory; leakage contracts; fault simulator; smoke;
  invalid seed-dependence isolated and quarantined; 2,400 probes and matched
  direct/oracle development controls completed; powered observability analysis
  and supporting calibration/learning-curve tooling completed.
- T+1–T+2: selected closed loop; estimator comparison; three-seed robustness;
  deterministic replay; confidence-gate confirmation; transient-safety audit;
  feature/learning curves; candidate, calibration, and failure forensics.
- T+2–T+3: complete — 16-cell control sensitivity, CEM 12/24/48/96 budget
  curve, ranked bottlenecks, five-seed refinement expansion, one-sided-probe
  treatment confound, freeze-chain and protected-driver hardening.
- T+3–T+4: complete — seven-seed impact/recovery/control decomposition,
  posterior-mean tradeoff and failure taxonomy, complete matrix consolidation,
  clean-process reproduction audit, final tests/validation, and pre-freeze
  hash/absence audit, plus an eighth supporting robustness seed.
- T+4–T+5: complete — formal Branch-A freeze, strict post-gate validation,
  one-shot protected evaluation, formal plots, reduced development dataset,
  cheap linear/MLP baselines, capped Qwen training/inference/closed loop, and
  the base-seed guarded continuous-control frontier.
- T+5–T+6: complete — eight-seed guarded robustness; population replication;
  uncertainty, elite-diversity, and feasible-proposal controls; post-freeze CEM
  synthesis; and exact boundary oracle failure/step-budget forensics.
- T+6–T+7: complete — matched direct/oracle/frozen-probe six-step arms; exact
  step-prefix replay; group-OOF visible continuation rule; first temporal seed
  replications; residual recalibration negative; boundary candidate launch.
- T+7–T+8: complete — base 4/5/6/7/8 direct/probe horizon curves; frozen
  sequential-rule selection; five-seed six-step/held-rule statistics and hard
  cases; fresh suite A generation/launch; exact command and test expansion.
- T+8–T+9: complete — fresh suites A/B and pooled 60-setup evidence;
  seed-2/3 horizon-eight replication; two-seed fresh-setup robustness;
  complete first candidate curve and triggered proposal-seed replication;
  outcome-free difficulty-matched fresh cohort preparation.
- T+9–T+10: complete — completed five-seed horizon/sequential robustness;
  outcome-free matched cohort on both planner seeds; balanced 2×90 setup/seed
  inference and exact failure stability; four-cohort shift; 100-test and
  166-command audits; candidate proposal seed-2 audit and conditional seed-3
  launch after both fixed instability triggers fired.
- T+10–T+11: complete — third proposal seed completed 20/20 with curve
  0/0/1/3 and only one stable recovery among ten seen on any seed; candidate
  aggregation gained per-case frequency and temporal-intersection evidence;
  integration/semantic-auditor tests, 102/102 full tests, formal Gate/SHA/JUnit
  bindings, interim progress/source/integrity audits, exact taxonomy
  reconciliations, and report consistency checks all passed.
- T+11–T+12: complete — all 12 hourly evidence bins are populated; exact
  protected-row, failure-ID, fresh-disjointness, candidate-semantic, plot,
  source, path, command, and report contracts pass. The T+11h15m cutoff was
  enforced, no long job remained, and all terminal strict progress, timeline,
  source, integrity, command, JUnit, and deliverables checks pass.

## Next research decision

Branch A is formally frozen rather than obtained by rounding the 78.7% formal
result. The no-residual discrete estimator is the protected policy and the one
allowed protected evaluation independently confirms +12.5 points of control
value without reselection. Capped Qwen is technically successful but trails
the cheap frozen linear controller by 4.2 points on its matched held-out subset.
The guarded posterior mean also closes as a negative result: across eight seeds
it adds only +1.35 points over frozen discrete with an interval spanning zero,
while tripling fault saturation. Population, risk, diversity, and feasibility
controls likewise provide no robust boundary fix. The strongest remaining
development-only mechanism is temporal truncation. The frozen visible rule is
positive on all five original-group planner seeds and all six cells of the
2-seed × 90-fresh-setup matrix, with two-way gains of +21.0 and +18.06 points
and positive intervals. The smallest decisive next experiment is therefore an
externally held generator or hardware comparison on 30 groups (ten per stratum)
and two preregistered planner seeds: identical rule versus fixed four, capped
at eight with success, steps, and saturation fixed before execution and no
threshold retuning. Require the two-way seed/group bootstrap lower bound on
strict-success gain to exceed zero; any step or saturation noninferiority
margin must also be fixed before data collection.
Remaining candidate-only sequence headroom is an evaluator upper bound until a
visible proposal/chooser replicates on held groups; no complementary recovery
survives all three proposal seeds. Stop broad CEM knob sweeps,
current-feature Qwen scaling, and continuous compensation without a safety
utility; none of these results can reopen the protected choice.
