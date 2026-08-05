# Supplier ranker training report

- Records: **41**
- Target: **Suitability Score (0-100)**
- Model: **Ridge (alpha=0.1)**
- Features: **83**
- Model SHA-256: `02494a01470a7f46311714dc56bb01d3f20b09cd6d157d088ad1c7939ef5134a`

## Validation

| Evaluation | MAE | RMSE | R² / rank correlation |
|---|---:|---:|---:|
| Leave-one-out | 3.717 | 5.094 | R² 0.055; rank 0.452 |
| Repeated 5-fold (5 repeats) | 4.065 ± 1.203 | 5.511 ± 1.575 | — |
| Median baseline, leave-one-out | 4.049 | 5.259 | — |

## Intended use

The model predicts a supplier suitability prior from structured supplier attributes. Runtime preferences remain explicit and inspectable.

## Limitations

- The dataset contains only 41 suppliers.
- The target is an expert-curated score rather than realized purchasing outcomes.
- The model supports shortlisting and manual review, not autonomous procurement.
