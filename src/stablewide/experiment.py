"""
Pilot v4: repeated-split prediction reliability vs biomarker-interpretation reliability for TabPFN-Wide.

Research questions
------------------
1) Seed stability
   With identical training/query patients, how much do predictions and feature explanations change
   when only method randomness changes?

2) Attention-readout composition control (TabPFN-Wide only)
   Query/test rows in TabPFN do not attend to one another. Therefore, with a fixed fitted context,
   fixed seed and fixed anchor patients, anchor predictions should be invariant (up to numerical
   noise) when unrelated query rows are added/removed. This experiment is NOT a transductive-
   prediction experiment. It asks a narrower reporting question: does the aggregate attention-based
   feature-importance readout change when the composition of the prediction batch changes?

3) Patient-subsampling stability
   With a fixed held-out test set and fixed model seed, how quickly do predictive performance,
   prediction stability, exact-gene explanation stability and module-level explanation stability
   deteriorate as the training cohort becomes smaller?

Methodological rules in v4
--------------------------
* No held-out-test leakage in imputation, variance filtering or co-expression modules.
* Patient subsampling does not change model seed.
* Exact ties are treated as ties. v4 never fabricates a primary top-k ranking by index/random order
  when a tie crosses the requested cutoff.
* L1 logistic regression is evaluated primarily as a sparse selector: its nonzero support and support
  stability are first-class outputs. Zero coefficients are NOT ranked.
* Random/index tie-break top-k results are retained only as diagnostics showing how tie-dependent a
  requested top-k would be; they are not primary stability estimates.
* A non-sparse L2 logistic baseline is included for an apples-to-apples continuous-ranking baseline.
* Top-k comparisons are reported only when enough runs have an identifiable top-k. Pairwise matched
  budgets prevent a sparse baseline from collapsing every comparison to a tiny global k.
* Repeated stratified outer splits quantify uncertainty; confidence intervals are computed across outer
  split-level estimates, never by pretending overlapping within-split subsamples are independent.
* Stability uses fixed pre-specified hyperparameters. Optional tuned RF/L1/L2 performance is evaluated
  separately with inner CV, so hyperparameter selection does not contaminate the stability experiment.
* Synthetic mode generates an independent dataset from the same DGP for every outer replicate by default.
* Prediction stability includes mean/max absolute probability change, probability SD, Jensen-Shannon
  distance and class agreement. Correlation is secondary.
* The query readout experiment treats anchor prediction invariance as a numerical/unit control using explicit relative and absolute tolerances.
* Empty result tables and small/imbalanced edge cases fail gracefully.

Important interpretation limits
-------------------------------
This is a pilot / falsification study. Overlapping subsamples are not independent cohorts. A stable
ranking can be biologically wrong, and an unstable exact-gene ranking can still represent a stable
co-expression programme. Synthetic truth and external/known biology are therefore separate evidence.
The fixed pool-fitted feature universe intentionally isolates model/cohort stability from upstream
feature-selection instability; a later pipeline-robustness study would need to vary preprocessing too.

Install
-------
  pip install numpy scipy pandas scikit-learn matplotlib
  pip install tabpfnwide

Examples
--------
  # Baseline smoke test, no TabPFN required
  python pilot_tabpfnwide_stability_v4.py --synthetic --quick --methods anova,rf,l1,l2

  # Include TabPFN-Wide attention (GPU recommended)
  python pilot_tabpfnwide_stability_v4.py --synthetic --quick

  # Real data: samples as rows, genes as columns; labels indexed by same sample IDs
  python pilot_tabpfnwide_stability_v4.py \
      --x brca_mrna.csv --y brca_labels.csv --label_col subtype \
      --n_features 2000 --known_genes ESR1,FOXA1,GATA3,FOXC1,CCDC170 \
      --save_scores --save_feature_tables
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import jensenshannon, squareform
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class MethodOutput:
    scores: np.ndarray
    proba: np.ndarray | None
    support: np.ndarray | None = None
    support_semantics: str = "none"


# synthetic data

def make_synthetic(n, p, n_modules, n_classes, delta, seed, label_noise=0.0):
    """Correlated gene modules; first n_classes latent modules carry class signal.

    `truth` marks every gene in a signal-bearing module as informative. This intentionally makes
    module-level recovery meaningful: swapping correlated genes inside one signal module should be
    penalized less than switching to a noise module.
    """
    if n_modules < n_classes:
        raise ValueError("syn_modules must be >= syn_classes")
    if p < n_modules:
        raise ValueError("n_features must be >= syn_modules")
    rng = np.random.default_rng(seed)
    per = int(np.ceil(p / n_modules))
    mod_of_gene = np.repeat(np.arange(n_modules), per)[:p]
    y = rng.integers(0, n_classes, n)
    Z = rng.standard_normal((n, n_modules))
    for c in range(n_classes):
        Z[y == c, c] += delta
    load = rng.uniform(0.6, 0.95, p) * rng.choice([-1.0, 1.0], p)
    eps = rng.standard_normal((n, p))
    X = Z[:, mod_of_gene] * load + np.sqrt(1.0 - load**2) * eps

    if label_noise > 0:
        flip = rng.random(n) < label_noise
        for i in np.where(flip)[0]:
            choices = np.delete(np.arange(n_classes), y[i])
            y[i] = rng.choice(choices)

    truth = mod_of_gene < n_classes
    genes = np.array([f"g{i:05d}" for i in range(p)])
    return X.astype(np.float32), y.astype(int), genes, truth


# input / preprocessing

def load_raw_data(args):
    if args.synthetic:
        return make_synthetic(
            args.syn_n,
            args.n_features,
            args.syn_modules,
            args.syn_classes,
            args.syn_delta,
            123,
            args.syn_label_noise,
        )

    if not args.x or not args.y:
        sys.exit("Provide --x and --y, or use --synthetic.")

    Xdf = pd.read_csv(args.x, index_col=0)
    ydf = pd.read_csv(args.y, index_col=0)
    col = args.label_col or ydf.columns[0]
    if col not in ydf.columns:
        sys.exit(f"Label column {col!r} not found in {args.y}.")

    if Xdf.index.has_duplicates or ydf.index.has_duplicates:
        sys.exit("Duplicate sample IDs detected in X or y; resolve them before running the pilot.")

    if set(Xdf.index) == set(ydf.index):
        ydf = ydf.loc[Xdf.index]
    elif args.allow_positional_labels and len(Xdf) == len(ydf):
        log("WARNING: aligning X/y by row position because --allow_positional_labels was set.")
    else:
        sys.exit(
            "X and y sample IDs do not match. Fix the indices, or use --allow_positional_labels only "
            "after independently verifying that row order is identical."
        )

    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    y = LabelEncoder().fit_transform(ydf[col].astype(str).values)
    genes = np.array(Xdf.columns.astype(str))
    return Xdf.values.astype(np.float64), y.astype(int), genes, None


def fit_pool_preprocessor(X, pool, genes, n_features):
    """Fit label-free imputation/filtering on development pool only and freeze feature universe."""
    Xpool = X[pool]
    med = np.nanmedian(Xpool, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Ximp = np.where(np.isfinite(X), X, med[None, :])

    var_pool = np.var(Ximp[pool], axis=0)
    keep = np.isfinite(var_pool) & (var_pool > 0)
    if not np.any(keep):
        raise ValueError("No non-constant features remain after pool-fitted preprocessing.")

    kept_idx = np.where(keep)[0]
    order = np.argsort(-var_pool[kept_idx], kind="mergesort")
    if n_features and len(order) > n_features:
        order = order[:n_features]
    sel = kept_idx[order]
    return Ximp[:, sel].astype(np.float32), genes[sel], sel, {
        "n_input_features": int(X.shape[1]),
        "n_nonconstant_pool": int(keep.sum()),
        "n_selected": int(len(sel)),
    }


def modules_from_corr(X_pool, n_modules):
    """Label-free co-expression modules fitted on development pool only."""
    p = X_pool.shape[1]
    if p == 1 or n_modules <= 1:
        return np.zeros(p, dtype=int)
    n_modules = min(n_modules, p)
    C = np.nan_to_num(np.corrcoef(X_pool.T), nan=0.0, posinf=0.0, neginf=0.0)
    D = 1.0 - np.abs(C)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=n_modules, criterion="maxclust") - 1


def stratified_subsample(y, idx, n, rng):
    idx = np.asarray(idx, dtype=int)
    classes, counts = np.unique(y[idx], return_counts=True)
    if n > len(idx):
        raise ValueError("subsample size exceeds available rows")
    if n < len(classes):
        raise ValueError("subsample size smaller than number of represented classes")

    frac = counts / counts.sum()
    alloc = np.minimum(np.maximum(1, np.floor(frac * n).astype(int)), counts)
    while alloc.sum() > n:
        candidates = np.where(alloc > 1, alloc, -1)
        j = int(np.argmax(candidates))
        alloc[j] -= 1
    while alloc.sum() < n:
        deficit = frac * n - alloc
        deficit[alloc >= counts] = -np.inf
        j = int(np.argmax(deficit))
        if not np.isfinite(deficit[j]):
            break
        alloc[j] += 1

    parts = [rng.choice(idx[y[idx] == c], a, replace=False) for c, a in zip(classes, alloc)]
    return np.sort(np.concatenate(parts))


# --------------------------------------------------------------------------- importance methods

def f_stat(X, y):
    F, _ = f_classif(X, y)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def imp_anova(Xtr, ytr, Xq, seed, args):
    return MethodOutput(f_stat(Xtr, ytr), None)


def imp_rf(Xtr, ytr, Xq, seed, args):
    m = RandomForestClassifier(
        n_estimators=args.rf_trees,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced" if args.rf_balanced else None,
    )
    m.fit(Xtr, ytr)
    scores = np.asarray(m.feature_importances_, dtype=float)
    support = scores > args.zero_tol
    return MethodOutput(scores, m.predict_proba(Xq), support, "positive RF importance (diagnostic support)")


def _ovr_logistic(Xtr, ytr, Xq, penalty, C, seed, class_weight):
    sc = StandardScaler().fit(Xtr)
    base = LogisticRegression(
        penalty=penalty,
        solver="liblinear",
        C=C,
        max_iter=3000,
        random_state=seed,
        class_weight=class_weight,
    )
    m = OneVsRestClassifier(base).fit(sc.transform(Xtr), ytr)
    scores = np.mean([np.abs(e.coef_[0]) for e in m.estimators_], axis=0)
    return scores, m.predict_proba(sc.transform(Xq))


def imp_l1(Xtr, ytr, Xq, seed, args):
    scores, proba = _ovr_logistic(
        Xtr, ytr, Xq, "l1", args.l1_C, seed, "balanced" if args.l1_balanced else None
    )
    support = scores > args.zero_tol
    return MethodOutput(scores, proba, support, "nonzero absolute L1 coefficient")


def imp_l2(Xtr, ytr, Xq, seed, args):
    scores, proba = _ovr_logistic(
        Xtr, ytr, Xq, "l2", args.l2_C, seed, "balanced" if args.l2_balanced else None
    )
    return MethodOutput(scores, proba)


def imp_attention(Xtr, ytr, Xq, seed, args):
    from tabpfnwide.classifier import TabPFNWideClassifier

    clf = TabPFNWideClassifier(
        model_name=args.model_name,
        device=args.device,
        save_attention_maps=True,
        n_estimators=1,
        features_per_group=1,
        random_state=seed,
    )
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xq)
    scores = np.asarray(clf.get_attention_to_label(), dtype=np.float64).reshape(-1)
    if scores.shape[0] != Xtr.shape[1]:
        raise RuntimeError(
            f"TabPFN-Wide returned {scores.shape[0]} attention scores for {Xtr.shape[1]} features. "
            "Check package/model compatibility before interpreting results."
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("TabPFN-Wide returned non-finite attention scores.")
    return MethodOutput(scores, np.asarray(proba, dtype=float))


METHODS = {
    "anova": imp_anova,
    "rf": imp_rf,
    "l1": imp_l1,
    "l2": imp_l2,
    "attention": imp_attention,
}
SEED_SENSITIVE = {"rf", "attention"}
QUERY_READOUT_METHODS = {"attention"}
PRIMARY_SUPPORT_METHODS = {"l1"}


# tuned performance track

def _inner_cv_splits(ytr, requested_folds, seed):
    """Return stratified inner-CV splits, or [] when tuning is not defensible."""
    ytr = np.asarray(ytr, dtype=int)
    _classes, counts = np.unique(ytr, return_counts=True)
    if len(counts) < 2:
        return []
    folds = min(int(requested_folds), int(counts.min()))
    if folds < 2:
        return []
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(ytr)), ytr))


def _cv_auc_for_logistic(Xtr, ytr, penalty, C, args, seed):
    splits = _inner_cv_splits(ytr, args.inner_folds, seed)
    if not splits:
        return np.nan
    vals = []
    n_classes = len(np.unique(ytr))
    for fold, (a, b) in enumerate(splits):
        try:
            _scores, p = _ovr_logistic(
                Xtr[a], ytr[a], Xtr[b], penalty, C, seed + fold,
                "balanced" if args.tuned_logistic_balanced else None,
            )
            vals.append(safe_auc(ytr[b], p, n_classes))
        except Exception:
            vals.append(np.nan)
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan


def tune_logistic_predict(Xtr, ytr, Xq, penalty, args, seed):
    grid = args.l1_C_grid if penalty == "l1" else args.l2_C_grid
    scored = [(float(C), _cv_auc_for_logistic(Xtr, ytr, penalty, float(C), args, seed)) for C in grid]
    finite = [(C, sc) for C, sc in scored if np.isfinite(sc)]
    if finite:
        # deterministic tie rule: highest CV AUROC, then smaller C (stronger regularization)
        best_C, best_score = sorted(finite, key=lambda z: (-z[1], z[0]))[0]
    else:
        best_C = args.l1_C if penalty == "l1" else args.l2_C
        best_score = np.nan
    scores, proba = _ovr_logistic(
        Xtr, ytr, Xq, penalty, best_C, seed,
        "balanced" if args.tuned_logistic_balanced else None,
    )
    return np.asarray(proba, dtype=float), {"C": float(best_C), "inner_auc": float(best_score) if np.isfinite(best_score) else None}


def _cv_auc_for_rf(Xtr, ytr, max_features, min_samples_leaf, args, seed):
    splits = _inner_cv_splits(ytr, args.inner_folds, seed)
    if not splits:
        return np.nan
    vals = []
    n_classes = len(np.unique(ytr))
    for fold, (a, b) in enumerate(splits):
        try:
            m = RandomForestClassifier(
                n_estimators=args.rf_tune_trees,
                n_jobs=-1,
                random_state=seed + fold,
                class_weight="balanced" if args.tuned_rf_balanced else None,
                max_features=max_features,
                min_samples_leaf=int(min_samples_leaf),
            )
            m.fit(Xtr[a], ytr[a])
            vals.append(safe_auc(ytr[b], m.predict_proba(Xtr[b]), n_classes))
        except Exception:
            vals.append(np.nan)
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan


def tune_rf_predict(Xtr, ytr, Xq, args, seed):
    candidates = []
    for mf in args.rf_max_features_grid:
        mf_value = mf
        try:
            if isinstance(mf, str) and mf not in {"sqrt", "log2"}:
                mf_value = float(mf)
        except Exception:
            pass
        for leaf in args.rf_min_leaf_grid:
            sc = _cv_auc_for_rf(Xtr, ytr, mf_value, leaf, args, seed)
            candidates.append((mf_value, int(leaf), sc))
    finite = [x for x in candidates if np.isfinite(x[2])]
    if finite:
        # deterministic tie rule: best AUROC, then simpler/larger leaf, then stringified max_features
        best_mf, best_leaf, best_score = sorted(
            finite, key=lambda z: (-z[2], -z[1], str(z[0]))
        )[0]
    else:
        best_mf, best_leaf, best_score = "sqrt", 1, np.nan
    m = RandomForestClassifier(
        n_estimators=args.rf_tune_trees,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced" if args.tuned_rf_balanced else None,
        max_features=best_mf,
        min_samples_leaf=best_leaf,
    )
    m.fit(Xtr, ytr)
    return np.asarray(m.predict_proba(Xq), dtype=float), {
        "max_features": best_mf,
        "min_samples_leaf": int(best_leaf),
        "inner_auc": float(best_score) if np.isfinite(best_score) else None,
    }


def run_tuned_performance(method, jobs, X, y, n_classes, args):
    """Nested/inner-CV performance track. Not used for primary stability estimates."""
    aucs, chosen = [], []
    for j, job in enumerate(jobs):
        tr = np.asarray(job["tr"], dtype=int)
        q = np.asarray(job["q"], dtype=int)
        ev = np.asarray(job.get("eval", q), dtype=int)
        seed = int(job.get("seed", 0))
        if method == "rf":
            p, meta = tune_rf_predict(X[tr], y[tr], X[q], args, seed)
        elif method in {"l1", "l2"}:
            p, meta = tune_logistic_predict(X[tr], y[tr], X[q], method, args, seed)
        else:
            raise ValueError(f"unsupported tuned baseline: {method}")
        pos = build_eval_positions(q, ev)
        aucs.append(safe_auc(y[ev], p[pos], n_classes))
        chosen.append(meta)
    arr = np.asarray(aucs, dtype=float)
    return {
        "method": method,
        "auroc_mean": float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan,
        "auroc_sd": float(np.nanstd(arr)) if np.isfinite(arr).any() else np.nan,
        "n_runs": int(len(arr)),
        "chosen_params": json.dumps(chosen, sort_keys=True),
    }


# general metrics

def safe_auc(yq, proba, n_classes):
    if proba is None or len(yq) == 0:
        return np.nan
    try:
        present = np.unique(yq)
        if n_classes == 2:
            if len(present) < 2:
                return np.nan
            return float(roc_auc_score(yq, proba[:, 1]))
        if len(present) < n_classes:
            return np.nan
        return float(roc_auc_score(yq, proba, multi_class="ovr", labels=np.arange(n_classes)))
    except Exception:
        return np.nan


def pair_mean(M):
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] < 2:
        return np.nan
    iu = np.triu_indices(M.shape[0], 1)
    vals = M[iu]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else np.nan


def pairwise_jaccard_from_masks(M):
    M = np.asarray(M, dtype=bool)
    if M.ndim != 2 or M.shape[0] < 2:
        return np.nan
    vals = []
    for i in range(M.shape[0]):
        for j in range(i + 1, M.shape[0]):
            union = np.logical_or(M[i], M[j]).sum()
            inter = np.logical_and(M[i], M[j]).sum()
            vals.append(1.0 if union == 0 else inter / union)
    return float(np.mean(vals)) if vals else np.nan


def masks_to_modules(masks, mod):
    masks = np.asarray(masks, dtype=bool)
    n_mod = int(mod.max()) + 1 if len(mod) else 0
    out = np.zeros((masks.shape[0], n_mod), dtype=bool)
    for i in range(masks.shape[0]):
        idx = np.flatnonzero(masks[i])
        if len(idx):
            out[i, np.unique(mod[idx])] = True
    return out


def tie_aware_ranks(scores):
    """Average ranks for exact ties; 1 = highest score. No arbitrary feature ordering."""
    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)
    return rankdata(-scores, method="average").astype(float)


def identifiable_topk_mask(scores, k, require_support=None):
    """Return an exact-k mask only when the cutoff is uniquely identifiable.

    If a tie crosses the kth boundary, returns (None, False). For a declared sparse support, k must
    also be <= support size; zero-valued/non-selected features are never used to pad a primary top-k.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    p = len(scores)
    if k <= 0 or k > p:
        return None, False
    if require_support is not None and int(np.sum(require_support)) < k:
        return None, False

    order = np.argsort(-scores, kind="mergesort")
    kth = scores[order[k - 1]]
    above = int(np.sum(scores > kth))
    equal = int(np.sum(scores == kth))
    slots = k - above
    if equal != slots:  # tie group straddles cutoff
        return None, False
    mask = scores >= kth
    if int(mask.sum()) != k:
        return None, False
    if require_support is not None and not np.all(require_support[mask]):
        return None, False
    return mask, True


def index_topk_mask(scores, k):
    """Diagnostic only: deterministic index ordering inside ties."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    k = min(max(int(k), 0), len(scores))
    if k == 0:
        return np.zeros(len(scores), dtype=bool)
    idx = np.arange(len(scores))
    order = np.lexsort((idx, -scores))
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:k]] = True
    return mask


def random_topk_mask(scores, k, rng):
    """Diagnostic only: random ordering inside exact ties."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    k = min(max(int(k), 0), len(scores))
    if k == 0:
        return np.zeros(len(scores), dtype=bool)
    # lexsort uses last key as primary: -scores primary, random values secondary.
    order = np.lexsort((rng.random(len(scores)), -scores))
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:k]] = True
    return mask


def topk_summary(score_runs, support_runs, k, mod, truth, known_idx, min_valid_fraction, random_seed=777):
    m, p = score_runs.shape
    k_eff = min(int(k), p)
    primary_masks = []
    valid = []
    index_masks = []
    random_masks = []
    shortfall = []
    boundary_tied = []
    rng = np.random.default_rng(random_seed + k_eff)

    for i in range(m):
        support = None if support_runs is None else support_runs[i]
        mask, ok = identifiable_topk_mask(score_runs[i], k_eff, support)
        valid.append(ok)
        primary_masks.append(mask if ok else np.zeros(p, dtype=bool))
        index_masks.append(index_topk_mask(score_runs[i], k_eff))
        random_masks.append(random_topk_mask(score_runs[i], k_eff, rng))
        if support is None:
            shortfall.append(0.0)
        else:
            shortfall.append(max(0, k_eff - int(support.sum())) / max(k_eff, 1))
        boundary_tied.append(0 if ok else 1)

    valid = np.asarray(valid, dtype=bool)
    valid_fraction = float(valid.mean()) if len(valid) else 0.0
    primary_ok = valid.sum() >= 2 and valid_fraction >= min_valid_fraction
    if primary_ok:
        P = np.asarray(primary_masks, dtype=bool)[valid]
        gene_j = pairwise_jaccard_from_masks(P)
        module_j = pairwise_jaccard_from_masks(masks_to_modules(P, mod))
        freq = P.mean(axis=0)
        core80 = int(np.sum(freq >= 0.8))
        precision_truth = float(np.mean([truth[row].mean() for row in P])) if truth is not None else np.nan
    else:
        gene_j = module_j = precision_truth = np.nan
        core80 = 0

    I = np.asarray(index_masks, dtype=bool)
    R = np.asarray(random_masks, dtype=bool)

    # Known-gene median ranks are tie-aware and do not need top-k to be identifiable.
    ranks = np.vstack([tie_aware_ranks(s) for s in score_runs])
    known = (
        "; ".join(f"{g}:{np.median(ranks[:, j]):.1f}" for g, j in known_idx.items())
        if known_idx else ""
    )

    return {
        "k_effective": k_eff,
        "topk_valid_fraction": valid_fraction,
        "topk_primary_defined": bool(primary_ok),
        "jaccard_gene": gene_j,
        "jaccard_module": module_j,
        "core80": core80,
        "precision_truth": precision_truth,
        "known_median_rank": known,
        "tie_boundary_fraction": float(np.mean(boundary_tied)),
        "support_shortfall_fraction": float(np.mean(shortfall)),
        "jaccard_index_tiebreak_diag": pairwise_jaccard_from_masks(I),
        "jaccard_random_tiebreak_diag": pairwise_jaccard_from_masks(R),
    }


def support_summary(support_runs, mod, truth):
    if support_runs is None:
        return {
            "support_size_mean": np.nan,
            "support_size_sd": np.nan,
            "support_size_median": np.nan,
            "support_size_min": np.nan,
            "support_size_max": np.nan,
            "support_jaccard_gene": np.nan,
            "support_jaccard_module": np.nan,
            "support_truth_precision": np.nan,
            "support_truth_recall": np.nan,
        }
    S = np.asarray(support_runs, dtype=bool)
    sizes = S.sum(axis=1)
    precision = recall = np.nan
    if truth is not None:
        precs = []
        recs = []
        truth_n = int(np.sum(truth))
        for row in S:
            if row.sum() > 0:
                precs.append(float(np.mean(truth[row])))
            if truth_n > 0:
                recs.append(float(np.logical_and(row, truth).sum() / truth_n))
        precision = float(np.mean(precs)) if precs else np.nan
        recall = float(np.mean(recs)) if recs else np.nan
    return {
        "support_size_mean": float(np.mean(sizes)),
        "support_size_sd": float(np.std(sizes)),
        "support_size_median": float(np.median(sizes)),
        "support_size_min": int(np.min(sizes)),
        "support_size_max": int(np.max(sizes)),
        "support_jaccard_gene": pairwise_jaccard_from_masks(S),
        "support_jaccard_module": pairwise_jaccard_from_masks(masks_to_modules(S, mod)),
        "support_truth_precision": precision,
        "support_truth_recall": recall,
    }


def rank_stability(score_runs):
    if score_runs.shape[0] < 2:
        return np.nan
    ranks = np.vstack([tie_aware_ranks(s) for s in score_runs])
    C = np.corrcoef(ranks)
    return pair_mean(C)


def prediction_stability(preds):
    if preds is None or preds.ndim != 3 or preds.shape[0] < 2:
        return {
            "pred_prob_corr": np.nan,
            "pred_prob_sd": np.nan,
            "pred_class_agreement": np.nan,
            "pred_jsd": np.nan,
            "pred_mad": np.nan,
            "pred_max_abs": np.nan,
        }

    Q = preds[:, :, 1] if preds.shape[2] == 2 else preds.reshape(preds.shape[0], -1)
    if Q.ndim == 1:
        Q = Q[:, None]
    prob_corr = pair_mean(np.corrcoef(Q))
    prob_sd = float(np.mean(np.std(preds, axis=0)))

    cls = np.argmax(preds, axis=2)
    agreements = []
    mad = []
    max_abs = []
    js_vals = []
    for i in range(preds.shape[0]):
        for j in range(i + 1, preds.shape[0]):
            agreements.append(float(np.mean(cls[i] == cls[j])))
            d = np.abs(preds[i] - preds[j])
            mad.append(float(np.mean(d)))
            max_abs.append(float(np.max(d)))
            for s in range(preds.shape[1]):
                p = np.clip(preds[i, s], 1e-12, 1.0)
                q = np.clip(preds[j, s], 1e-12, 1.0)
                p = p / p.sum()
                q = q / q.sum()
                js_vals.append(float(jensenshannon(p, q, base=2.0)))

    return {
        "pred_prob_corr": prob_corr,
        "pred_prob_sd": prob_sd,
        "pred_class_agreement": float(np.mean(agreements)) if agreements else np.nan,
        "pred_jsd": float(np.mean(js_vals)) if js_vals else np.nan,
        "pred_mad": float(np.mean(mad)) if mad else np.nan,
        "pred_max_abs": float(np.max(max_abs)) if max_abs else np.nan,
    }


def build_eval_positions(q_abs, eval_abs):
    pos = {int(a): i for i, a in enumerate(np.asarray(q_abs, dtype=int))}
    missing = [int(a) for a in np.asarray(eval_abs, dtype=int) if int(a) not in pos]
    if missing:
        raise ValueError(f"Evaluation rows are not contained in query rows: {missing[:5]}")
    return np.array([pos[int(a)] for a in np.asarray(eval_abs, dtype=int)], dtype=int)


# execution / summaries

def run_jobs(method, jobs, X, y, n_classes, args):
    scores_runs, support_runs, aucs, pred_runs = [], [], [], []
    t0 = time.time()
    eval_reference = None
    support_semantics = "none"
    all_have_support = True

    for j, job in enumerate(jobs):
        tr = np.asarray(job["tr"], dtype=int)
        q = np.asarray(job["q"], dtype=int)
        seed = int(job["seed"])
        ev = np.asarray(job.get("eval", q), dtype=int)

        if eval_reference is None:
            eval_reference = ev.copy()
        elif not np.array_equal(eval_reference, ev):
            raise ValueError("All jobs in one stability experiment must use the same evaluation patients.")

        out = METHODS[method](X[tr], y[tr], X[q], seed, args)
        scores = np.asarray(out.scores, dtype=float).reshape(-1)
        if scores.shape[0] != X.shape[1]:
            raise RuntimeError(f"{method} returned {len(scores)} scores for {X.shape[1]} features")
        scores_runs.append(scores)
        support_semantics = out.support_semantics
        if out.support is None:
            all_have_support = False
        else:
            support_runs.append(np.asarray(out.support, dtype=bool).reshape(-1))

        if out.proba is None:
            aucs.append(np.nan)
        else:
            proba_all = np.asarray(out.proba, dtype=float)
            pos = build_eval_positions(q, ev)
            proba_eval = proba_all[pos]
            pred_runs.append(proba_eval)
            aucs.append(safe_auc(y[ev], proba_eval, n_classes))

        if (j + 1) % 5 == 0 or j + 1 == len(jobs):
            log(f"    {method}: {j + 1}/{len(jobs)} runs ({time.time() - t0:.0f}s)")

    S = np.vstack(scores_runs)
    supports = np.vstack(support_runs) if all_have_support and len(support_runs) == len(jobs) else None
    P = np.stack(pred_runs) if pred_runs and len(pred_runs) == len(jobs) else None
    return {
        "scores": S,
        "supports": supports,
        "support_semantics": support_semantics,
        "aucs": np.asarray(aucs, dtype=float),
        "preds": P,
    }


def rows_for(method, exp, n_train, detail, args, mod, truth, known_idx):
    scores = detail["scores"]
    supports = detail["supports"] if method in PRIMARY_SUPPORT_METHODS or method == "rf" else None
    pred_stab = prediction_stability(detail["preds"])
    sup = support_summary(supports, mod, truth)
    rows = []
    for k in args.topk:
        top = topk_summary(
            scores,
            supports if method in PRIMARY_SUPPORT_METHODS else None,
            k,
            mod,
            truth,
            known_idx,
            args.min_topk_valid_fraction,
        )
        rows.append({
            "method": method,
            "experiment": exp,
            "n_train": int(n_train),
            "k": int(k),
            "n_runs": int(scores.shape[0]),
            "rank_spearman_tie_aware": rank_stability(scores),
            "auroc_mean": np.nanmean(detail["aucs"]) if np.isfinite(detail["aucs"]).any() else np.nan,
            "auroc_sd": np.nanstd(detail["aucs"]) if np.isfinite(detail["aucs"]).any() else np.nan,
            "support_semantics": detail["support_semantics"],
            **pred_stab,
            **sup,
            **top,
        })
    return rows


def feature_stability_table(detail, genes, topk, zero_tol):
    scores = detail["scores"]
    supports = detail["supports"]
    ranks = np.vstack([tie_aware_ranks(s) for s in scores])
    df = pd.DataFrame({
        "gene": genes,
        "median_score": np.median(scores, axis=0),
        "median_rank_tie_aware": np.median(ranks, axis=0),
        "rank_q25": np.quantile(ranks, 0.25, axis=0),
        "rank_q75": np.quantile(ranks, 0.75, axis=0),
    })
    if supports is not None:
        df["support_freq"] = supports.mean(axis=0)
    for k in topk:
        masks = []
        valid = []
        for i, s in enumerate(scores):
            req = supports[i] if supports is not None and detail["support_semantics"].startswith("nonzero") else None
            mask, ok = identifiable_topk_mask(s, min(k, scores.shape[1]), req)
            masks.append(mask if ok else np.zeros(scores.shape[1], dtype=bool))
            valid.append(ok)
        valid = np.asarray(valid, dtype=bool)
        df[f"top{k}_valid_run_fraction"] = float(valid.mean())
        df[f"top{k}_freq_among_valid_runs"] = (
            np.asarray(masks, dtype=bool)[valid].mean(axis=0) if valid.any() else np.nan
        )
    sort_cols = ["support_freq", "median_rank_tie_aware"] if "support_freq" in df else ["median_rank_tie_aware"]
    ascending = [False, True] if "support_freq" in df else [True]
    return df.sort_values(sort_cols, ascending=ascending)


def max_identifiable_k(detail, method, max_k, min_fraction):
    supports = detail["supports"] if method in PRIMARY_SUPPORT_METHODS else None
    best = 0
    for k in range(1, min(max_k, detail["scores"].shape[1]) + 1):
        valid = []
        for i, s in enumerate(detail["scores"]):
            req = None if supports is None else supports[i]
            _, ok = identifiable_topk_mask(s, k, req)
            valid.append(ok)
        if np.mean(valid) >= min_fraction:
            best = k
        else:
            break
    return best


def matched_budget_table(details, args, mod, truth, known_idx):
    """Largest common identifiable k per experiment/n across all available methods."""
    rows = []
    groups = sorted(set((exp, n) for (exp, n, _m) in details))
    for exp, n in groups:
        methods = sorted(m for (e, nn, m) in details if e == exp and nn == n)
        if len(methods) < 2:
            continue
        maxima = {
            m: max_identifiable_k(
                details[(exp, n, m)], m, args.matched_k_max, args.min_topk_valid_fraction
            ) for m in methods
        }
        k = min(maxima.values()) if maxima else 0
        if k < 1:
            rows.append({
                "experiment": exp,
                "n_train": n,
                "method": "__group__",
                "matched_k": 0,
                "note": "No common identifiable top-k at requested coverage",
                "method_k_maxima": json.dumps(maxima, sort_keys=True),
            })
            continue
        for m in methods:
            d = details[(exp, n, m)]
            supports = d["supports"] if m in PRIMARY_SUPPORT_METHODS else None
            top = topk_summary(
                d["scores"], supports, k, mod, truth, known_idx,
                args.min_topk_valid_fraction, random_seed=991,
            )
            rows.append({
                "experiment": exp,
                "n_train": n,
                "method": m,
                "matched_k": k,
                "auroc_mean": np.nanmean(d["aucs"]) if np.isfinite(d["aucs"]).any() else np.nan,
                "jaccard_gene": top["jaccard_gene"],
                "jaccard_module": top["jaccard_module"],
                "topk_valid_fraction": top["topk_valid_fraction"],
                "method_k_maximum": maxima[m],
                "method_k_maxima": json.dumps(maxima, sort_keys=True),
                "note": "",
            })
    return pd.DataFrame(rows)



def pairwise_matched_budget_table(details, args, mod, truth, known_idx, reference="attention"):
    """Pairwise matched budgets so one sparse method cannot collapse every other comparison."""
    rows = []
    groups = sorted(set((exp, n) for (exp, n, _m) in details))
    for exp, n in groups:
        if (exp, n, reference) not in details:
            continue
        ref = details[(exp, n, reference)]
        ref_max = max_identifiable_k(ref, reference, args.matched_k_max, args.min_topk_valid_fraction)
        for baseline in sorted(m for (e, nn, m) in details if e == exp and nn == n and m != reference):
            base = details[(exp, n, baseline)]
            base_max = max_identifiable_k(base, baseline, args.matched_k_max, args.min_topk_valid_fraction)
            k = min(ref_max, base_max)
            if k < 1:
                rows.append({
                    "experiment": exp, "n_train": int(n), "reference": reference,
                    "baseline": baseline, "matched_k": 0,
                    "reference_k_max": ref_max, "baseline_k_max": base_max,
                    "note": "No pairwise identifiable top-k at requested coverage",
                })
                continue
            rsup = ref["supports"] if reference in PRIMARY_SUPPORT_METHODS else None
            bsup = base["supports"] if baseline in PRIMARY_SUPPORT_METHODS else None
            rtop = topk_summary(ref["scores"], rsup, k, mod, truth, known_idx,
                                args.min_topk_valid_fraction, random_seed=2301)
            btop = topk_summary(base["scores"], bsup, k, mod, truth, known_idx,
                                args.min_topk_valid_fraction, random_seed=2302)
            rauc = np.nanmean(ref["aucs"]) if np.isfinite(ref["aucs"]).any() else np.nan
            bauc = np.nanmean(base["aucs"]) if np.isfinite(base["aucs"]).any() else np.nan
            rows.append({
                "experiment": exp, "n_train": int(n), "reference": reference,
                "baseline": baseline, "matched_k": int(k),
                "reference_k_max": int(ref_max), "baseline_k_max": int(base_max),
                "reference_gene_jaccard": rtop["jaccard_gene"],
                "baseline_gene_jaccard": btop["jaccard_gene"],
                "delta_gene_jaccard": rtop["jaccard_gene"] - btop["jaccard_gene"]
                    if np.isfinite(rtop["jaccard_gene"]) and np.isfinite(btop["jaccard_gene"]) else np.nan,
                "reference_module_jaccard": rtop["jaccard_module"],
                "baseline_module_jaccard": btop["jaccard_module"],
                "delta_module_jaccard": rtop["jaccard_module"] - btop["jaccard_module"]
                    if np.isfinite(rtop["jaccard_module"]) and np.isfinite(btop["jaccard_module"]) else np.nan,
                "reference_auroc": rauc,
                "baseline_auroc": bauc,
                "delta_auroc": rauc - bauc if np.isfinite(rauc) and np.isfinite(bauc) else np.nan,
                "note": "",
            })
    return pd.DataFrame(rows)


def query_invariance_pass(preds, atol, rtol):
    if preds is None or len(preds) < 2:
        return np.nan
    ref = np.asarray(preds[0], dtype=float)
    return bool(all(np.allclose(np.asarray(p, dtype=float), ref, atol=atol, rtol=rtol) for p in preds[1:]))


def bootstrap_mean_ci(values, level, n_boot, seed):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(vals.mean())
    if len(vals) == 1:
        return mean, np.nan, np.nan, 1
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = rng.choice(vals, size=len(vals), replace=True).mean()
    alpha = 1.0 - level
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return mean, float(lo), float(hi), int(len(vals))


def aggregate_results(df, args, group_cols, metrics, seed_base=7000):
    """Aggregate scalar outer-split estimates; CIs are across outer splits, not within-run pairs."""
    if df is None or df.empty:
        return pd.DataFrame()
    rows = []
    for gi, (keys, g) in enumerate(df.groupby(group_cols, dropna=False)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_outer_rows"] = int(g["outer_id"].nunique()) if "outer_id" in g else int(len(g))
        for mi, metric in enumerate(metrics):
            if metric not in g:
                continue
            mean, lo, hi, n_eff = bootstrap_mean_ci(
                g[metric].values, args.ci_level, args.ci_bootstrap,
                seed_base + gi * 100 + mi,
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
            row[f"{metric}_n"] = n_eff
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_k_pairwise(df, args, reference="attention"):
    """Within-outer-split deltas at the pre-specified decision k."""
    if df is None or df.empty:
        return pd.DataFrame()
    sub = df[(df.experiment == "subsample") & (df.k == args.decision_k)].copy()
    rows = []
    for (outer_id, n_train), g in sub.groupby(["outer_id", "n_train"]):
        rg = g[g.method == reference]
        if rg.empty:
            continue
        r = rg.iloc[0]
        for baseline in args.decision_baselines:
            bg = g[g.method == baseline]
            if bg.empty:
                continue
            b = bg.iloc[0]
            rows.append({
                "outer_id": int(outer_id), "n_train": int(n_train),
                "reference": reference, "baseline": baseline, "k": int(args.decision_k),
                "reference_gene_jaccard": r.jaccard_gene,
                "baseline_gene_jaccard": b.jaccard_gene,
                "delta_gene_jaccard": r.jaccard_gene - b.jaccard_gene
                    if np.isfinite(r.jaccard_gene) and np.isfinite(b.jaccard_gene) else np.nan,
                "reference_module_jaccard": r.jaccard_module,
                "baseline_module_jaccard": b.jaccard_module,
                "delta_module_jaccard": r.jaccard_module - b.jaccard_module
                    if np.isfinite(r.jaccard_module) and np.isfinite(b.jaccard_module) else np.nan,
                "reference_auroc": r.auroc_mean,
                "baseline_auroc": b.auroc_mean,
                "delta_auroc": r.auroc_mean - b.auroc_mean
                    if np.isfinite(r.auroc_mean) and np.isfinite(b.auroc_mean) else np.nan,
                "reference_topk_defined": bool(r.topk_primary_defined),
                "baseline_topk_defined": bool(b.topk_primary_defined),
            })
    return pd.DataFrame(rows)


def preregistered_decision(agg, pair_agg, args):
    """Pilot go/no-go criterion fixed by CLI arguments before attention results are inspected."""
    lines = [
        "=== Pre-registered pilot decision (v4) ===",
        f"decision k={args.decision_k}; AUROC floor={args.decision_auc_floor:.2f}; ",
        f"gene-stability ceiling={args.decision_gene_stability_ceiling:.2f}; pairwise margin={args.decision_stability_margin:.2f}",
        f"minimum outer splits with finite estimates={args.min_outer_for_decision}",
    ]
    signals = []
    if agg is not None and not agg.empty:
        a = agg[(agg.method == "attention") & (agg.experiment == "subsample") & (agg.k == args.decision_k)]
        for _, r in a.iterrows():
            n = int(r.n_train)
            n_auc = int(r.get("auroc_mean_n", 0))
            n_j = int(r.get("jaccard_gene_n", 0))
            if min(n_auc, n_j) < args.min_outer_for_decision:
                continue
            auc_lo = r.get("auroc_mean_ci_low", np.nan)
            j_hi = r.get("jaccard_gene_ci_high", np.nan)
            if np.isfinite(auc_lo) and np.isfinite(j_hi):
                hit = auc_lo >= args.decision_auc_floor and j_hi <= args.decision_gene_stability_ceiling
                signals.append((f"high-prediction/low-interpretation at n={n}", bool(hit)))
                lines.append(
                    f"n={n}: attention AUROC lower CI={auc_lo:.3f}; gene-Jaccard upper CI={j_hi:.3f} -> "
                    + ("SIGNAL" if hit else "no signal")
                )
    if pair_agg is not None and not pair_agg.empty:
        for _, r in pair_agg.iterrows():
            n_eff = int(r.get("delta_gene_jaccard_n", 0))
            if n_eff < args.min_outer_for_decision:
                continue
            lo = r.get("delta_gene_jaccard_ci_low", np.nan)
            hi = r.get("delta_gene_jaccard_ci_high", np.nan)
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            ref_auc_lo = r.get("reference_auroc_ci_low", np.nan)
            specific_gap = hi < -args.decision_stability_margin or lo > args.decision_stability_margin
            predictive_ok = np.isfinite(ref_auc_lo) and ref_auc_lo >= args.decision_auc_floor
            specific = bool(specific_gap and predictive_ok)
            signals.append((f"TabPFN-specific stability difference vs {r.baseline} at n={int(r.n_train)}", specific))
            lines.append(
                f"n={int(r.n_train)} vs {r.baseline}: Δgene-Jaccard CI=[{lo:.3f},{hi:.3f}], "
                f"attention AUROC lower CI={ref_auc_lo:.3f} -> "
                + ("SIGNAL" if specific else "no decision signal")
            )
    go = any(v for _, v in signals)
    if not signals:
        lines.append("Insufficient outer-split evidence for a decision; do not interpret as a null result.")
        verdict = "INCONCLUSIVE"
    elif go:
        verdict = "GO: at least one pre-specified signal survived the outer-split uncertainty check."
    else:
        verdict = "STOP/REDIRECT: no pre-specified signal survived; do not manufacture a stability story."
    lines += ["", verdict]
    return "\n".join(lines), verdict


# plotting / reporting

def plot_results(df, out):
    if df is None or df.empty:
        log("no result rows; skipping plots")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        log("matplotlib unavailable, skipping plots")
        return

    sub = df[df.experiment == "subsample"]
    if not sub.empty:
        # Prefer the smallest requested k for primary plots because it is most likely to be identifiable.
        k = int(sub.k.min())
        g0 = sub[sub.k == k]
        for metric, title, fname in [
            ("auroc_mean", "Held-out AUROC", "auroc_vs_n.png"),
            ("jaccard_gene", f"Identifiable top-{k} gene stability", "gene_stability_vs_n.png"),
            ("jaccard_module", f"Identifiable top-{k} module stability", "module_stability_vs_n.png"),
        ]:
            fig = plt.figure(figsize=(7, 4.5))
            ax = fig.add_subplot(111)
            for m, g in g0.groupby("method"):
                g = g.sort_values("n_train")
                ax.plot(g.n_train, g[metric], marker="o", label=m)
            ax.set_xlabel("Training samples")
            ax.set_title(title)
            if metric != "auroc_mean":
                ax.set_ylim(0, 1)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / fname, dpi=150)
            plt.close(fig)

        l1 = g0[g0.method == "l1"]
        if not l1.empty:
            fig = plt.figure(figsize=(7, 4.5))
            ax = fig.add_subplot(111)
            l1 = l1.sort_values("n_train")
            ax.plot(l1.n_train, l1.support_size_mean, marker="o")
            ax.set_xlabel("Training samples")
            ax.set_ylabel("Mean nonzero features")
            ax.set_title("L1 support size vs cohort size")
            fig.tight_layout()
            fig.savefig(out / "l1_support_size_vs_n.png", dpi=150)
            plt.close(fig)

    q = df[(df.experiment == "query") & (df.method == "attention")]
    if not q.empty:
        k = int(q.k.min())
        r = q[q.k == k].iloc[0]
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.bar(["Gene Jaccard", "Module Jaccard", "Class agreement"],
               [r.jaccard_gene, r.jaccard_module, r.pred_class_agreement])
        ax.set_ylim(0, 1)
        ax.set_title("Attention readout-composition control")
        fig.tight_layout()
        fig.savefig(out / "query_readout_control.png", dpi=150)
        plt.close(fig)


def package_versions():
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    try:
        import sklearn
        versions["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import importlib.metadata as md
        versions["tabpfnwide"] = md.version("tabpfnwide")
        versions["tabpfn"] = md.version("tabpfn")
    except Exception:
        pass
    return versions


# CLI / experiment construction

def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x"); ap.add_argument("--y"); ap.add_argument("--label_col")
    ap.add_argument("--allow_positional_labels", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--syn_n", type=int, default=440)
    ap.add_argument("--syn_modules", type=int, default=100)
    ap.add_argument("--syn_classes", type=int, default=4)
    ap.add_argument("--syn_delta", type=float, default=1.5)
    ap.add_argument("--syn_label_noise", type=float, default=0.0)
    ap.add_argument("--methods", default="anova,rf,l1,l2,attention")
    ap.add_argument("--n_features", type=int, default=2000, help="pool-fitted label-free variance filter")
    ap.add_argument("--sizes", default="200,100,50")
    ap.add_argument("--n_reps", type=int, default=20)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--query_size", type=int, default=40, help="total query batch size in readout-composition control")
    ap.add_argument("--query_anchor_size", type=int, default=0, help="0 = choose automatically")
    ap.add_argument("--query_atol", type=float, default=1e-5, help="absolute tolerance for anchor-prediction invariance")
    ap.add_argument("--query_rtol", type=float, default=1e-4, help="relative tolerance for anchor-prediction invariance")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--outer_splits", type=int, default=5, help="repeated stratified outer splits; synthetic mode regenerates an independent dataset per outer replicate by default")
    ap.add_argument("--synthetic_fixed_dataset", action="store_true", help="reuse one synthetic matrix across outer splits instead of regenerating from the DGP")
    ap.add_argument("--syn_seed", type=int, default=123)
    ap.add_argument("--ci_level", type=float, default=0.95)
    ap.add_argument("--ci_bootstrap", type=int, default=2000, help="bootstrap resamples across outer-split estimates")
    ap.add_argument("--subsample_model_seed", type=int, default=0)
    ap.add_argument("--topk", default="5,10,20,50")
    ap.add_argument("--min_topk_valid_fraction", type=float, default=0.90)
    ap.add_argument("--matched_k_max", type=int, default=50)
    ap.add_argument("--decision_k", type=int, default=10, help="pre-specified top-k used for go/no-go and fixed-k pairwise comparisons")
    ap.add_argument("--decision_auc_floor", type=float, default=0.75)
    ap.add_argument("--decision_gene_stability_ceiling", type=float, default=0.50)
    ap.add_argument("--decision_stability_margin", type=float, default=0.10, help="practical pairwise difference in gene-stability Jaccard")
    ap.add_argument("--decision_baselines", default="rf,l2", help="continuous/stochastic baselines for TabPFN-specific decision check")
    ap.add_argument("--min_outer_for_decision", type=int, default=3)
    ap.add_argument("--n_modules", type=int, default=100)
    ap.add_argument("--known_genes", default="")
    ap.add_argument("--model_name", default="wide-v2-5k")
    ap.add_argument("--device", default=None)
    ap.add_argument("--rf_trees", type=int, default=500)
    ap.add_argument("--rf_balanced", action="store_true")
    ap.add_argument("--l1_C", type=float, default=0.1, help="pre-specify; do not tune post hoc to manufacture support size")
    ap.add_argument("--l1_balanced", action="store_true")
    ap.add_argument("--l2_C", type=float, default=1.0)
    ap.add_argument("--l2_balanced", action="store_true")
    ap.add_argument("--run_tuned_performance", action="store_true", help="run separate nested/inner-CV tuned AUROC track for RF/L1/L2")
    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--l1_C_grid", default="0.01,0.03,0.1,0.3,1,3")
    ap.add_argument("--l2_C_grid", default="0.01,0.1,1,10")
    ap.add_argument("--rf_max_features_grid", default="sqrt,0.2,0.5")
    ap.add_argument("--rf_min_leaf_grid", default="1,2,5")
    ap.add_argument("--rf_tune_trees", type=int, default=300)
    ap.add_argument("--tuned_logistic_balanced", action="store_true")
    ap.add_argument("--tuned_rf_balanced", action="store_true")
    ap.add_argument("--zero_tol", type=float, default=1e-12)
    ap.add_argument("--out", default="pilot_out_v4")
    ap.add_argument("--force", action="store_true", help="skip conservative attention-memory guard")
    ap.add_argument("--save_scores", action="store_true")
    ap.add_argument("--save_feature_tables", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    if a.quick:
        a.n_features = min(a.n_features, 500)
        a.n_reps = min(a.n_reps, 5)
        a.outer_splits = min(a.outer_splits, 2)
        a.ci_bootstrap = min(a.ci_bootstrap, 200)
        a.n_seeds = min(a.n_seeds, 3)
        a.sizes = "100,50"
        a.rf_trees = min(a.rf_trees, 200)
        a.query_size = min(a.query_size, 20)
        a.matched_k_max = min(a.matched_k_max, 20)

    a.methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    a.decision_baselines = [m.strip() for m in a.decision_baselines.split(",") if m.strip()]
    a.l1_C_grid = [float(x) for x in a.l1_C_grid.split(",") if x.strip()]
    a.l2_C_grid = [float(x) for x in a.l2_C_grid.split(",") if x.strip()]
    a.rf_max_features_grid = [x.strip() for x in a.rf_max_features_grid.split(",") if x.strip()]
    a.rf_min_leaf_grid = [int(x) for x in a.rf_min_leaf_grid.split(",") if x.strip()]
    a.sizes = [int(s) for s in a.sizes.split(",") if s]
    a.topk = sorted(set(int(s) for s in a.topk.split(",") if s))
    unknown = [m for m in a.methods if m not in METHODS]
    if unknown:
        sys.exit(f"unknown methods: {unknown}; choose from {list(METHODS)}")
    if not a.topk or any(k <= 0 for k in a.topk):
        sys.exit("--topk must contain positive integers")
    if not (0.0 <= a.syn_label_noise < 1.0):
        sys.exit("--syn_label_noise must be in [0,1)")
    if not (0.0 < a.min_topk_valid_fraction <= 1.0):
        sys.exit("--min_topk_valid_fraction must be in (0,1]")
    if a.matched_k_max < 1:
        sys.exit("--matched_k_max must be >=1")
    if a.l1_C <= 0 or a.l2_C <= 0 or any(x <= 0 for x in a.l1_C_grid + a.l2_C_grid):
        sys.exit("L1/L2 C values must be >0")
    if a.outer_splits < 1:
        sys.exit("--outer_splits must be >=1")
    if not (0.5 < a.ci_level < 1.0):
        sys.exit("--ci_level must be in (0.5,1)")
    if a.ci_bootstrap < 100:
        sys.exit("--ci_bootstrap must be >=100")
    if a.decision_k < 1:
        sys.exit("--decision_k must be >=1")
    if a.min_outer_for_decision < 1:
        sys.exit("--min_outer_for_decision must be >=1")
    bad_dec = [m for m in a.decision_baselines if m not in {"rf", "l2", "anova"}]
    if bad_dec:
        sys.exit(f"unsupported --decision_baselines: {bad_dec}; choose from rf,l2,anova")

    if a.device is None:
        try:
            import torch
            a.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            a.device = "cpu"
    return a


def build_query_jobs(pool, test, y, args):
    """Fixed anchor rows + varying unrelated query rows. Predictions on anchors are a control."""
    test = np.asarray(test, dtype=int)
    n_classes = len(np.unique(y[test]))
    if len(test) < max(2 * n_classes, 6):
        log("query control skipped: held-out test set too small for anchors + varying context")
        return [], None

    rng_anchor = np.random.default_rng(2026)
    anchor_n = args.query_anchor_size if args.query_anchor_size > 0 else max(n_classes, min(10, len(test) // 3))
    anchor_n = min(anchor_n, len(test) - n_classes)
    if anchor_n < n_classes:
        log("query control skipped: not enough rows to include every class in anchors")
        return [], None

    anchors = stratified_subsample(y, test, anchor_n, rng_anchor)
    remaining = np.setdiff1d(test, anchors, assume_unique=False)
    if len(remaining) < 2:
        log("query control skipped: fewer than two context candidates remain")
        return [], None

    desired_context = max(1, args.query_size - len(anchors))
    context_n = min(desired_context, max(1, len(remaining) // 2))
    represented = len(np.unique(y[remaining]))
    context_n = max(context_n, represented)
    if context_n >= len(remaining):
        context_n = len(remaining) - 1
    if context_n <= 0:
        log("query control skipped: no room for varying context")
        return [], None

    rng = np.random.default_rng(1)
    contexts, seen = [], set()
    attempts = 0
    while len(contexts) < args.n_reps and attempts < args.n_reps * 100:
        attempts += 1
        try:
            c = stratified_subsample(y, remaining, context_n, rng)
        except ValueError:
            c = np.sort(rng.choice(remaining, context_n, replace=False))
        key = tuple(c.tolist())
        if key not in seen:
            seen.add(key)
            contexts.append(c)

    if len(contexts) < 2:
        log("query control skipped: could not generate at least two distinct context sets")
        return [], None
    if len(contexts) < args.n_reps:
        log(f"query control: only {len(contexts)} distinct context sets available (requested {args.n_reps})")

    jobs = [
        {"tr": pool, "q": np.concatenate([anchors, c]), "eval": anchors, "seed": 0}
        for c in contexts
    ]
    return jobs, anchors


# v4 outer-split orchestration

def _tag_rows(rows, outer_id, split_seed):
    for r in rows:
        r["outer_id"] = int(outer_id)
        r["outer_split_seed"] = int(split_seed)
    return rows


def run_one_outer(args, outer_id, raw_tuple, root_out):
    Xraw, y, genes_raw, truth_raw = raw_tuple
    n = len(y)
    n_classes = len(np.unique(y))
    class_counts = np.bincount(y)
    if n_classes < 2 or np.min(class_counts) < 2:
        raise ValueError("Every outer replicate needs at least two classes and at least two samples per class.")

    split_seed = int(args.split_seed + outer_id * 1009)
    idx = np.arange(n)
    pool, test = train_test_split(idx, test_size=args.test_frac, stratify=y, random_state=split_seed)
    split_out = root_out / f"outer_{outer_id:02d}"
    split_out.mkdir(parents=True, exist_ok=True)

    log(f"outer {outer_id}: raw n={n}, p={Xraw.shape[1]}, pool={len(pool)}, test={len(test)}, split_seed={split_seed}")
    X, genes, sel, prep_meta = fit_pool_preprocessor(Xraw, pool, genes_raw, args.n_features)
    truth = truth_raw[sel] if truth_raw is not None else None
    p = X.shape[1]
    if "attention" in args.methods and not args.force:
        est = 12 * p**2 * 4 / 1e9
        if est > 16:
            raise RuntimeError(
                f"Conservative attention-buffer estimate ~{est:.0f} GB at p={p}. "
                "Lower --n_features or use --force only after checking actual memory behavior."
            )

    mod = modules_from_corr(X[pool], args.n_modules)
    requested_known = [g.strip() for g in args.known_genes.split(",") if g.strip()]
    known_idx = {g: int(np.where(genes == g)[0][0]) for g in requested_known if (genes == g).any()}

    rows, details, support_rows, tuned_rows = [], {}, [], []

    def save_detail(exp, method, n_train, detail):
        tag = f"{exp}_{method}_n{n_train}"
        if args.save_scores:
            np.save(split_out / f"scores_{tag}.npy", detail["scores"])
            if detail["supports"] is not None:
                np.save(split_out / f"supports_{tag}.npy", detail["supports"])
        if args.save_feature_tables:
            feature_stability_table(detail, genes, args.topk, args.zero_tol).to_csv(
                split_out / f"features_{tag}.csv", index=False
            )

    # Seed stability: only methods for which randomness is scientifically meaningful here.
    for m in [m for m in args.methods if m in SEED_SENSITIVE]:
        log(f"outer {outer_id}: seed stability -> {m}")
        jobs = [{"tr": pool, "q": test, "eval": test, "seed": s} for s in range(args.n_seeds)]
        d = run_jobs(m, jobs, X, y, n_classes, args)
        details[("seed", len(pool), m)] = d
        new_rows = rows_for(m, "seed", len(pool), d, args, mod, truth, known_idx)
        rows.extend(_tag_rows(new_rows, outer_id, split_seed))
        save_detail("seed", m, len(pool), d)

    # Query-batch composition control. Predictions on fixed anchors are a numerical/unit control.
    qjobs, anchors = build_query_jobs(pool, test, y, args)
    if qjobs and "attention" in args.methods:
        log(f"outer {outer_id}: attention readout-composition control ({len(qjobs)} context sets)")
        d = run_jobs("attention", qjobs, X, y, n_classes, args)
        details[("query", len(pool), "attention")] = d
        inv = query_invariance_pass(d["preds"], args.query_atol, args.query_rtol)
        new_rows = rows_for("attention", "query", len(pool), d, args, mod, truth, known_idx)
        for r in new_rows:
            r["query_invariance_pass"] = inv
            r["query_atol"] = args.query_atol
            r["query_rtol"] = args.query_rtol
        rows.extend(_tag_rows(new_rows, outer_id, split_seed))
        save_detail("query", "attention", len(pool), d)

    # Full-pool tuned performance: one outer held-out evaluation per split.
    if args.run_tuned_performance:
        full_job = [{"tr": pool, "q": test, "eval": test, "seed": args.subsample_model_seed}]
        for m in [x for x in ("rf", "l1", "l2") if x in args.methods]:
            tr = run_tuned_performance(m, full_job, X, y, n_classes, args)
            tuned_rows.append({
                "outer_id": outer_id, "outer_split_seed": split_seed,
                "experiment": "full_pool", "n_train": len(pool), **tr,
            })

    # Patient-subsampling stability with fixed model seed.
    for size in args.sizes:
        if size >= len(pool) or size < n_classes:
            continue
        log(f"outer {outer_id}: subsample n={size}")
        rng = np.random.default_rng(1000 + size + outer_id * 100003)
        subs = [stratified_subsample(y, pool, size, rng) for _ in range(args.n_reps)]
        jobs = [{"tr": ss, "q": test, "eval": test, "seed": args.subsample_model_seed} for ss in subs]
        for m in args.methods:
            d = run_jobs(m, jobs, X, y, n_classes, args)
            details[("subsample", size, m)] = d
            new_rows = rows_for(m, "subsample", size, d, args, mod, truth, known_idx)
            rows.extend(_tag_rows(new_rows, outer_id, split_seed))
            save_detail("subsample", m, size, d)
            if d["supports"] is not None:
                support_rows.append({
                    "outer_id": outer_id, "outer_split_seed": split_seed,
                    "experiment": "subsample", "n_train": size, "method": m,
                    "support_semantics": d["support_semantics"],
                    **support_summary(d["supports"], mod, truth),
                    "auroc_mean": np.nanmean(d["aucs"]) if np.isfinite(d["aucs"]).any() else np.nan,
                    "auroc_sd": np.nanstd(d["aucs"]) if np.isfinite(d["aucs"]).any() else np.nan,
                })
        if args.run_tuned_performance:
            for m in [x for x in ("rf", "l1", "l2") if x in args.methods]:
                tr = run_tuned_performance(m, jobs, X, y, n_classes, args)
                tuned_rows.append({
                    "outer_id": outer_id, "outer_split_seed": split_seed,
                    "experiment": "subsample", "n_train": size, **tr,
                })

    # Pairwise matched budgets are split-local diagnostics, so a sparse baseline cannot set k for all methods.
    pair = pairwise_matched_budget_table(details, args, mod, truth, known_idx)
    if not pair.empty:
        pair["outer_id"] = outer_id
        pair["outer_split_seed"] = split_seed

    common = matched_budget_table(details, args, mod, truth, known_idx)
    if not common.empty:
        common["outer_id"] = outer_id
        common["outer_split_seed"] = split_seed

    split_meta = {
        "outer_id": outer_id,
        "split_seed": split_seed,
        "n": int(n), "p": int(p), "pool_n": int(len(pool)), "test_n": int(len(test)),
        "class_counts": class_counts.tolist(), "preprocessing": prep_meta,
        "selected_feature_indices": sel.tolist(),
    }
    (split_out / "meta.json").write_text(
        json.dumps(split_meta, indent=2, default=str),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(split_out / "results.csv", index=False)
    pair.to_csv(split_out / "pairwise_matched_budget.csv", index=False)
    common.to_csv(split_out / "common_matched_budget_diagnostic.csv", index=False)
    pd.DataFrame(support_rows).to_csv(split_out / "support_results.csv", index=False)
    pd.DataFrame(tuned_rows).to_csv(split_out / "tuned_performance.csv", index=False)
    return {
        "rows": rows, "pairwise": pair, "common": common,
        "support": support_rows, "tuned": tuned_rows, "meta": split_meta,
    }


def plot_aggregate(agg, out):
    if agg is None or agg.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    sub = agg[agg.experiment == "subsample"]
    if sub.empty:
        return
    k = int(sub.k.min())
    ss = sub[sub.k == k]
    for metric, title, fname in [
        ("auroc_mean", "Held-out AUROC across outer splits", "aggregate_auroc.png"),
        ("jaccard_gene", f"Top-{k} gene stability across outer splits", "aggregate_gene_stability.png"),
        ("jaccard_module", f"Top-{k} module stability across outer splits", "aggregate_module_stability.png"),
    ]:
        mean_col, lo_col, hi_col = f"{metric}_mean", f"{metric}_ci_low", f"{metric}_ci_high"
        if mean_col not in ss:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for m, g in ss.groupby("method"):
            g = g.sort_values("n_train")
            y = g[mean_col].to_numpy(dtype=float)
            lo = g[lo_col].to_numpy(dtype=float)
            hi = g[hi_col].to_numpy(dtype=float)
            err_low = np.where(np.isfinite(lo), y - lo, 0.0)
            err_hi = np.where(np.isfinite(hi), hi - y, 0.0)
            ax.errorbar(g.n_train, y, yerr=np.vstack([err_low, err_hi]), marker="o", capsize=3, label=m)
        ax.set_xlabel("Training samples")
        ax.set_title(title)
        if metric != "auroc_mean":
            ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)


def main():
    args = parse()
    # Decision k must exist in the reported top-k grid; add it before any result is seen.
    args.topk = sorted(set(args.topk + [args.decision_k]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # For real data, load once. For synthetic data, v4 defaults to independent DGP replicates.
    base_raw = None
    if not args.synthetic or args.synthetic_fixed_dataset:
        if args.synthetic:
            base_raw = make_synthetic(
                args.syn_n, args.n_features, args.syn_modules, args.syn_classes,
                args.syn_delta, args.syn_seed, args.syn_label_noise,
            )
        else:
            base_raw = load_raw_data(args)

    all_rows, all_pair, all_common, all_support, all_tuned, outer_meta = [], [], [], [], [], []
    for outer_id in range(args.outer_splits):
        if args.synthetic and not args.synthetic_fixed_dataset:
            raw = make_synthetic(
                args.syn_n, args.n_features, args.syn_modules, args.syn_classes,
                args.syn_delta, args.syn_seed + outer_id, args.syn_label_noise,
            )
        else:
            raw = base_raw
        try:
            res = run_one_outer(args, outer_id, raw, out)
        except Exception as e:
            log(f"outer {outer_id} FAILED: {type(e).__name__}: {e}")
            raise
        all_rows.extend(res["rows"])
        if not res["pairwise"].empty:
            all_pair.append(res["pairwise"])
        if not res["common"].empty:
            all_common.append(res["common"])
        all_support.extend(res["support"])
        all_tuned.extend(res["tuned"])
        outer_meta.append(res["meta"])

    df = pd.DataFrame(all_rows)
    pair = pd.concat(all_pair, ignore_index=True) if all_pair else pd.DataFrame()
    common = pd.concat(all_common, ignore_index=True) if all_common else pd.DataFrame()
    support = pd.DataFrame(all_support)
    tuned = pd.DataFrame(all_tuned)

    df.to_csv(out / "results_outer.csv", index=False)
    pair.to_csv(out / "pairwise_matched_budget_outer.csv", index=False)
    common.to_csv(out / "common_matched_budget_diagnostic_outer.csv", index=False)
    support.to_csv(out / "support_results_outer.csv", index=False)
    tuned.to_csv(out / "tuned_performance_outer.csv", index=False)

    result_metrics = [
        "auroc_mean", "jaccard_gene", "jaccard_module", "rank_spearman_tie_aware",
        "topk_valid_fraction", "support_size_mean", "support_jaccard_gene", "support_jaccard_module",
        "pred_mad", "pred_max_abs", "pred_jsd", "pred_class_agreement",
    ]
    agg = aggregate_results(
        df, args, ["method", "experiment", "n_train", "k"], result_metrics, seed_base=8100
    )
    agg.to_csv(out / "aggregate_results.csv", index=False)

    fixed_pair = fixed_k_pairwise(df, args)
    fixed_pair.to_csv(out / "fixed_k_pairwise_outer.csv", index=False)
    pair_agg = aggregate_results(
        fixed_pair, args, ["reference", "baseline", "n_train", "k"],
        ["delta_gene_jaccard", "delta_module_jaccard", "delta_auroc",
         "reference_gene_jaccard", "baseline_gene_jaccard", "reference_auroc", "baseline_auroc"],
        seed_base=9100,
    )
    pair_agg.to_csv(out / "aggregate_fixed_k_pairwise.csv", index=False)

    if not tuned.empty:
        tuned_agg = aggregate_results(
            tuned, args, ["method", "experiment", "n_train"], ["auroc_mean"], seed_base=10100
        )
    else:
        tuned_agg = pd.DataFrame()
    tuned_agg.to_csv(out / "aggregate_tuned_performance.csv", index=False)

    decision_text, verdict = preregistered_decision(agg, pair_agg, args)
    (out / "preregistered_decision.txt").write_text(
        decision_text,
        encoding="utf-8",
    )
    plot_aggregate(agg, out)

    meta = {
        "args": vars(args),
        "versions": package_versions(),
        "outer_meta": outer_meta,
        "pre_registered_decision": {
            "decision_k": args.decision_k,
            "auc_floor": args.decision_auc_floor,
            "gene_stability_ceiling": args.decision_gene_stability_ceiling,
            "stability_margin": args.decision_stability_margin,
            "decision_baselines": args.decision_baselines,
            "min_outer_for_decision": args.min_outer_for_decision,
        },
        "methodology": {
            "outer_uncertainty": "CIs bootstrap outer-split scalar estimates; overlapping within-split subsamples are not treated as independent.",
            "synthetic": "By default each outer replicate is a newly generated dataset from the same DGP.",
            "stability_track": "Uses fixed pre-specified hyperparameters so cohort sampling is the main perturbation.",
            "performance_track": "Optional tuned RF/L1/L2 uses inner stratified CV and is kept separate from stability metrics.",
            "l1": "Primary object is the nonzero support; zero coefficients are unranked.",
            "pairwise_budget": "Matched top-k is pairwise; sparse L1 cannot collapse RF/L2/attention comparisons globally.",
            "query": "Fixed-anchor prediction invariance uses np.allclose(atol=query_atol, rtol=query_rtol); attention variation is interpreted only as readout-composition variation.",
        },
    }
    (out / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n" + decision_text)
    log(f"wrote v4 outputs to {out}; verdict={verdict}")


if __name__ == "__main__":
    main()
