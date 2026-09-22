# Data layout

The experiment expects a matrix with **samples/patients as rows** and **genes/features as columns**, plus a separate label CSV indexed by the same sample IDs.

Recommended real-data files:

```text
data/processed/brca_mrna.csv
data/processed/brca_labels.csv
```

Example label file:

```csv
sample_id,subtype
P001,ER+
P002,TNBC
```

Do not treat single cells from one patient as independent samples. For single-cell data, create patient-level pseudobulk first (`04_prepare_real_data.ipynb`).
