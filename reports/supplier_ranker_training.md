# Supplier ranker training report

- Records: **41**
- Target: **Suitability Score (0-100)**
- Model: **ExtraTreesRegressor**
- Model SHA-256: `45ee9642c7ce8d59a86fb9ddcf184c3905f8c2ad23119701209431009ef949b5`

## Validation

| Evaluation | MAE | RMSE | R² / rank correlation |
|---|---:|---:|---:|
| Leave-one-out | 3.955 | 4.799 | R² 0.162; rank 0.443 |
| Repeated 5-fold (5 repeats) | 3.585 ± 0.676 | 4.436 ± 0.777 | — |
| Median baseline, leave-one-out | 4.049 | 5.259 | — |

## Intended use

The model predicts a supplier suitability prior from structured supplier attributes. Runtime preferences such as category, dropshipping, 3D availability, lead time, MOQ, and price are applied transparently after prediction.

## Limitations

- The dataset contains only 41 suppliers.
- The target is an expert-curated score rather than realized purchasing outcomes.
- The model should support shortlisting and manual review, not autonomous procurement.
