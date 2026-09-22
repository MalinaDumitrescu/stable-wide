# TabPFN-Wide stability pilot v4

This is the corrected pilot for testing whether predictive reliability and biomarker-interpretation reliability diverge in high-dimensional, low-sample biomedical data.

## What v4 fixes

- repeated stratified outer splits;
- bootstrap confidence intervals across outer-split estimates, not overlapping subsamples;
- independent synthetic datasets per outer replicate by default;
- fixed model seed for patient-subsampling stability;
- seed experiment restricted to RF and TabPFN-Wide attention;
- no held-out-test leakage in imputation, variance filtering, or co-expression modules;
- exact ties stay ties;
- L1 zeros are unranked and L1 is evaluated primarily by its nonzero support;
- L2 logistic regression provides a continuous-ranking baseline;
- pairwise matched budgets prevent sparse L1 from forcing every method to top-3/top-5;
- fixed decision-k comparison is retained separately from matched-budget diagnostics;
- query experiment is only an attention-readout composition control;
- anchor predictions use `np.allclose` with explicit absolute/relative tolerances;
- prediction stability includes mean/max probability change, JSD, class agreement, and SD;
- optional nested/inner-CV tuned RF/L1/L2 performance track is separate from the fixed-hyperparameter stability track;
- pre-specified pilot go/no-go criteria are stored in `meta.json` before interpretation.

## Install

```bash
pip install numpy scipy pandas scikit-learn matplotlib
pip install tabpfnwide
```

GPU is strongly recommended for the attention runs.

## 1. Baseline smoke test

```bash
python pilot_tabpfnwide_stability_v4.py \
  --synthetic --quick \
  --methods anova,rf,l1,l2 \
  --out pilot_v4_baseline_smoke
```

`--quick` intentionally uses only up to 2 outer replicates, so the preregistered decision should remain **INCONCLUSIVE** when the default requires 3 outer replicates. A quick run is a plumbing check, not a research result.

## 2. First real TabPFN-Wide attention check

```bash
python pilot_tabpfnwide_stability_v4.py \
  --synthetic --quick \
  --methods anova,rf,l1,l2,attention \
  --out pilot_v4_attention_smoke
```

Inspect first:

- `outer_*/results.csv` query rows;
- `query_invariance_pass`;
- `pred_mad` / `pred_max_abs`;
- attention vs RF seed rows;
- attention returned feature-vector shape.

If anchor predictions fail the allclose control materially, stop and investigate before interpreting attention rankings.

## 3. Main fixed-hyperparameter stability pilot

Example:

```bash
python pilot_tabpfnwide_stability_v4.py \
  --synthetic \
  --methods anova,rf,l1,l2,attention \
  --outer_splits 5 \
  --n_reps 20 \
  --n_seeds 10 \
  --sizes 200,100,50 \
  --n_features 2000 \
  --topk 5,10,20,50 \
  --decision_k 10 \
  --decision_auc_floor 0.75 \
  --decision_gene_stability_ceiling 0.50 \
  --decision_stability_margin 0.10 \
  --min_outer_for_decision 3 \
  --out pilot_v4_main
```

Synthetic mode generates a new dataset from the same DGP for every outer replicate unless `--synthetic_fixed_dataset` is supplied.

## 4. Separate tuned performance track

This is intentionally separate from stability. Tuning is done only inside the current training subset by stratified inner CV.

```bash
python pilot_tabpfnwide_stability_v4.py \
  --synthetic \
  --methods rf,l1,l2 \
  --outer_splits 5 \
  --n_reps 20 \
  --sizes 200,100,50 \
  --n_features 2000 \
  --run_tuned_performance \
  --inner_folds 3 \
  --out pilot_v4_tuned_performance
```

Because synthetic generation and split seeds are deterministic, using the same `--syn_seed`, `--split_seed`, and outer-split settings reproduces the same outer datasets/splits as the corresponding stability run.

## Main outputs

- `results_outer.csv` — one stability summary per outer split / method / experiment / k;
- `aggregate_results.csv` — means and bootstrap CIs across outer splits;
- `support_results_outer.csv` — L1/RF support diagnostics;
- `pairwise_matched_budget_outer.csv` — split-local pairwise matched-k comparisons;
- `fixed_k_pairwise_outer.csv` — pre-specified decision-k differences vs selected baselines;
- `aggregate_fixed_k_pairwise.csv` — outer-split CIs for those differences;
- `tuned_performance_outer.csv` — tuned baseline AUROC track, if requested;
- `aggregate_tuned_performance.csv` — tuned AUROC CIs;
- `preregistered_decision.txt` — pilot go/no-go evaluation;
- `meta.json` — arguments, software versions, split metadata, and pre-specified decision thresholds.

## Interpretation

Do not claim TabPFN-Wide is unstable merely because exact genes vary. Check, separately:

1. predictive reliability;
2. gene-ranking reliability;
3. L1 sparse-selection reliability;
4. module-level reliability;
5. whether any TabPFN-specific difference survives outer-split uncertainty.

A null pilot is a valid outcome. If no pre-specified signal survives, the script explicitly recommends stopping/redirecting rather than manufacturing a story.
