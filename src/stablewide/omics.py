from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def validate_matrix_and_labels(x_csv, y_csv, label_col=None):
    """Validate patient-by-feature matrix and label file before a real-data run."""
    x_csv, y_csv = Path(x_csv), Path(y_csv)
    X = pd.read_csv(x_csv, index_col=0)
    y = pd.read_csv(y_csv, index_col=0)
    if X.index.has_duplicates or y.index.has_duplicates:
        raise ValueError("Duplicate sample IDs detected.")
    if set(X.index) != set(y.index):
        missing_y = sorted(set(X.index) - set(y.index))[:10]
        missing_x = sorted(set(y.index) - set(X.index))[:10]
        raise ValueError(f"Sample IDs differ. Missing in y: {missing_y}; missing in X: {missing_x}")
    y = y.loc[X.index]
    col = label_col or y.columns[0]
    if col not in y.columns:
        raise ValueError(f"Label column {col!r} not found.")
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, y, col


def pseudobulk_from_h5ad(
    h5ad_path,
    out_x,
    out_y,
    patient_col,
    label_col,
    layer="counts",
    min_cells=20,
    log_cpm=True,
):
    """Create patient-level pseudobulk from an AnnData file.

    Counts are summed by patient. If log_cpm=True, the output is log1p(CPM).
    Each patient must have exactly one non-null label.
    """
    try:
        import anndata as ad
        from scipy import sparse
    except ImportError as e:
        raise ImportError("Install optional omics dependencies: pip install anndata scipy") from e

    adata = ad.read_h5ad(h5ad_path)
    for col in (patient_col, label_col):
        if col not in adata.obs.columns:
            raise KeyError(f"{col!r} not found in adata.obs")

    if layer is None:
        M = adata.X
    elif layer in adata.layers:
        M = adata.layers[layer]
    else:
        raise KeyError(f"Layer {layer!r} not found. Available layers: {list(adata.layers.keys())}")

    patients = adata.obs[patient_col].astype(str)
    labels = adata.obs[label_col]
    rows, ids, out_labels = [], [], []

    for patient in patients.unique():
        idx = np.flatnonzero((patients == patient).to_numpy())
        if len(idx) < min_cells:
            continue
        vals = labels.iloc[idx].dropna().astype(str).unique()
        if len(vals) != 1:
            raise ValueError(f"Patient {patient!r} has {len(vals)} unique labels; expected exactly one.")
        block = M[idx]
        summed = np.asarray(block.sum(axis=0)).ravel() if sparse.issparse(block) else np.asarray(block).sum(axis=0)
        summed = np.asarray(summed, dtype=float)
        if log_cpm:
            lib = summed.sum()
            if lib <= 0:
                continue
            summed = np.log1p(summed / lib * 1_000_000.0)
        rows.append(summed)
        ids.append(patient)
        out_labels.append(vals[0])

    if not rows:
        raise ValueError("No patients passed min_cells / label checks.")

    X = pd.DataFrame(np.vstack(rows), index=ids, columns=adata.var_names.astype(str))
    y = pd.DataFrame({label_col: out_labels}, index=ids)
    Path(out_x).parent.mkdir(parents=True, exist_ok=True)
    Path(out_y).parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(out_x)
    y.to_csv(out_y)
    return X, y
