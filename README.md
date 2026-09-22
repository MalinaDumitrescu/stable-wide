# STABLE-WIDE

**Prediction reliability vs biomarker-interpretation reliability in high-dimensional, low-sample biomedical data.**

The project tests TabPFN-Wide attention against classical baselines while keeping prediction, feature-ranking, sparse-selection and module-level stability separate.

## Open in PyCharm

1. Open this folder as a project.
2. Create a Python 3.11 environment.
3. Install `requirements.txt`.
4. For TabPFN-Wide attention runs, install the optional package and use a CUDA GPU if possible.
5. Open the notebooks in order.

```bash
pip install -r requirements.txt
pip install -e .
pip install tabpfnwide
```

For `.h5ad` pseudobulk preparation:

```bash
pip install -r requirements-omics.txt
```

## Notebook order

- `00_setup.ipynb` — environment and project check
- `01_synthetic_baselines.ipynb` — fast baseline sanity check
- `02_attention_smoke.ipynb` — first real TabPFN-Wide attention run
- `03_full_synthetic_study.ipynb` — main controlled experiment
- `04_prepare_real_data.ipynb` — validate CSV data or create patient-level pseudobulk
- `05_brca_real_study.ipynb` — real breast-cancer experiment
- `06_final_analysis.ipynb` — final tables, plots and go/no-go decision

## Core research rules

- held-out test data never fit imputation, variance filtering or modules;
- patient subsampling uses a fixed model seed;
- exact ties are not given fake ranks;
- L1 is evaluated primarily by its nonzero support;
- uncertainty is estimated across repeated outer splits;
- tuned predictive performance is separate from the fixed-hyperparameter stability study;
- the query experiment is only a readout-composition control, not a transductive-prediction claim;
- a null pilot is a valid stopping result.

The exact experiment engine is in `src/stablewide/experiment.py`. The notebooks are deliberately thin so the research logic stays readable.
