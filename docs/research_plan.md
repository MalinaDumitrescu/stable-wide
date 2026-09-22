# Research plan

## Question

When sample size becomes small, does predictive performance remain useful after biomarker-level explanations have become unreliable?

## Primary comparisons

1. prediction reliability;
2. exact-gene ranking reliability;
3. sparse support reliability for L1;
4. co-expression-module reliability.

## Methods

- TabPFN-Wide attention
- ANOVA F
- Random Forest
- L1 logistic regression
- L2 logistic regression

## Controlled perturbations

- random/model seed;
- patient subsampling;
- query-batch composition as an attention-readout control.

## Pre-specified pilot signal

At the chosen decision `k`, continue only if repeated outer splits support a practically meaningful prediction/interpretation gap or a TabPFN-specific stability difference relative to the declared baselines. The script stores the thresholds and returns `INCONCLUSIVE` when too few outer splits exist.

## What is not a result

- a quick run;
- one train/test split;
- a gene list padded with tied zero coefficients;
- a stable ranking without synthetic truth or biological validation;
- query-batch changes in the aggregate attention summary interpreted as prediction instability.
