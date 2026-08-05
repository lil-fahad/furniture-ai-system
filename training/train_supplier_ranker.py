from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, RepeatedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

from furniture_ai.supplier_ranker import StructuredSupplierTransformer, load_supplier_rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual_order = actual.argsort().argsort().astype(float)
    predicted_order = predicted.argsort().argsort().astype(float)
    return float(np.corrcoef(actual_order, predicted_order)[0, 1])


def build_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("features", StructuredSupplierTransformer()),
            ("vectorizer", DictVectorizer(sparse=False)),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    max_features=0.7,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the supplier suitability ranker")
    parser.add_argument("--data", type=Path, default=Path("data/suppliers_master.csv.gz.b64"))
    parser.add_argument(
        "--model", type=Path, default=Path("models/supplier_ranker/model.joblib")
    )
    parser.add_argument(
        "--parts-manifest",
        type=Path,
        default=Path("models/supplier_ranker/model.parts.json"),
    )
    parser.add_argument("--part-chars", type=int, default=16000)
    parser.add_argument(
        "--metrics", type=Path, default=Path("models/supplier_ranker/metrics.json")
    )
    parser.add_argument(
        "--predictions", type=Path, default=Path("reports/supplier_ranker_predictions.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/supplier_ranker_training.md")
    )
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    rows = load_supplier_rows(args.data)
    targets = np.asarray([float(row["Suitability Score"]) for row in rows], dtype=float)
    model = build_model(args.seed)

    leave_one_out = LeaveOneOut()
    loo_predictions = cross_val_predict(model, rows, targets, cv=leave_one_out, n_jobs=1)
    baseline_predictions = cross_val_predict(
        DummyRegressor(strategy="median"),
        np.zeros((len(rows), 1)),
        targets,
        cv=leave_one_out,
    )
    repeated_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=args.seed)
    repeated_mae = -cross_val_score(
        model,
        rows,
        targets,
        cv=repeated_cv,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    repeated_rmse = -cross_val_score(
        model,
        rows,
        targets,
        cv=repeated_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )

    metrics = {
        "schema_version": 1,
        "trained_at": "2026-08-05",
        "seed": args.seed,
        "records": len(rows),
        "target": "Suitability Score",
        "model": "ExtraTreesRegressor",
        "leave_one_out": {
            "mae": float(mean_absolute_error(targets, loo_predictions)),
            "rmse": float(mean_squared_error(targets, loo_predictions) ** 0.5),
            "r2": float(r2_score(targets, loo_predictions)),
            "rank_correlation": rank_correlation(targets, loo_predictions),
        },
        "repeated_5_fold_5_repeats": {
            "mae_mean": float(repeated_mae.mean()),
            "mae_std": float(repeated_mae.std()),
            "rmse_mean": float(repeated_rmse.mean()),
            "rmse_std": float(repeated_rmse.std()),
        },
        "median_baseline_leave_one_out": {
            "mae": float(mean_absolute_error(targets, baseline_predictions)),
            "rmse": float(mean_squared_error(targets, baseline_predictions) ** 0.5),
        },
        "limitations": [
            "Only 41 supplier records are available.",
            (
                "The target score is an expert-curated suitability score, "
                "not observed procurement outcomes."
            ),
            "The model is a ranking assistant and should not make autonomous purchasing decisions.",
        ],
    }

    model.fit(rows, targets)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model, compress=3)
    args.parts_manifest.parent.mkdir(parents=True, exist_ok=True)
    encoded = base64.b64encode(args.model.read_bytes()).decode("ascii")
    part_names: list[str] = []
    for index, start in enumerate(range(0, len(encoded), args.part_chars)):
        part_name = f"model.part{index:03d}.b64"
        (args.parts_manifest.parent / part_name).write_text(
            encoded[start : start + args.part_chars] + "\n", encoding="ascii"
        )
        part_names.append(part_name)
    parts_payload = {
        "schema_version": 1,
        "format": "joblib-base64-parts",
        "parts": part_names,
        "binary_size_bytes": args.model.stat().st_size,
        "binary_sha256": sha256_file(args.model),
    }
    args.parts_manifest.write_text(
        json.dumps(parts_payload, indent=2) + "\n", encoding="utf-8"
    )
    metrics["artifact"] = {
        "binary_path": str(args.model),
        "binary_size_bytes": args.model.stat().st_size,
        "binary_sha256": sha256_file(args.model),
        "parts_manifest": str(args.parts_manifest),
        "parts": len(part_names),
        "parts_manifest_sha256": sha256_file(args.parts_manifest),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Supplier Name",
                "Reference Score",
                "LOO Predicted Score",
                "Absolute Error",
            ]
        )
        for row, actual, predicted in sorted(
            zip(rows, targets, loo_predictions, strict=True),
            key=lambda item: item[2],
            reverse=True,
        ):
            writer.writerow(
                [
                    row["Supplier Name"],
                    round(float(actual), 2),
                    round(float(predicted), 2),
                    round(abs(float(actual - predicted)), 2),
                ]
            )

    loo = metrics["leave_one_out"]
    repeated = metrics["repeated_5_fold_5_repeats"]
    baseline = metrics["median_baseline_leave_one_out"]
    report = (
        "# Supplier ranker training report\n\n"
        f"- Records: **{len(rows)}**\n"
        "- Target: **Suitability Score (0-100)**\n"
        "- Model: **ExtraTreesRegressor**\n"
        f"- Model SHA-256: `{metrics['artifact']['binary_sha256']}`\n\n"
        "## Validation\n\n"
        "| Evaluation | MAE | RMSE | R² / rank correlation |\n"
        "|---|---:|---:|---:|\n"
        f"| Leave-one-out | {loo['mae']:.3f} | {loo['rmse']:.3f} | "
        f"R² {loo['r2']:.3f}; rank {loo['rank_correlation']:.3f} |\n"
        "| Repeated 5-fold (5 repeats) | "
        f"{repeated['mae_mean']:.3f} ± {repeated['mae_std']:.3f} | "
        f"{repeated['rmse_mean']:.3f} ± {repeated['rmse_std']:.3f} | — |\n"
        f"| Median baseline, leave-one-out | {baseline['mae']:.3f} | "
        f"{baseline['rmse']:.3f} | — |\n\n"
        "## Intended use\n\n"
        "The model predicts a supplier suitability prior from structured supplier "
        "attributes. Runtime preferences such as category, dropshipping, 3D availability, "
        "lead time, MOQ, and price are applied transparently after prediction.\n\n"
        "## Limitations\n\n"
        "- The dataset contains only 41 suppliers.\n"
        "- The target is an expert-curated score rather than realized purchasing outcomes.\n"
        "- The model should support shortlisting and manual review, not autonomous procurement.\n"
    )
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
