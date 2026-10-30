#!/usr/bin/env python3
"""
Classical machine learning workflow:
dataset loading, exploratory analysis, feature engineering, preprocessing,
model comparison, cross-validation, final evaluation, and artifact export.

Dependencies:
    pip install numpy pandas scikit-learn joblib

Run:
    python classical_ml_project.py
    python classical_ml_project.py --output-dir results --test-size 0.2 --folds 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
POSITIVE_CLASS = "malignant"

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

LOG = logging.getLogger("classical_ml")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible classical machine learning experiment."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("classical_ml_results"),
        help="Directory for datasets, reports, predictions, and model artifacts.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction reserved for the final holdout test set.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=SEED,
        help="Seed for reproducible splitting and model initialization.",
    )
    return parser.parse_args()


def validate_configuration(args: argparse.Namespace) -> None:
    if not 0.05 <= args.test_size <= 0.40:
        raise ValueError("--test-size must be between 0.05 and 0.40.")
    if args.folds < 2 or args.folds > 10:
        raise ValueError("--folds must be between 2 and 10.")
    if args.random_state < 0:
        raise ValueError("--random-state must be non-negative.")


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the Wisconsin Diagnostic Breast Cancer dataset bundled with
    scikit-learn. The original target is 0 for malignant and 1 for benign.
    This project explicitly uses malignant as the positive class (label 1).
    """
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()

    if features.empty or features.isnull().all().any():
        raise ValueError("The dataset contains unusable feature columns.")

    # These transformations use only values from the same observation.
    # They do not estimate parameters from the complete dataset.
    mean_radius = features["mean radius"].replace(0, np.nan)
    mean_perimeter = features["mean perimeter"].replace(0, np.nan)

    features["worst_to_mean_radius_ratio"] = (
        features["worst radius"] / mean_radius
    )
    features["mean_area_perimeter_ratio"] = (
        features["mean area"] / mean_perimeter.pow(2)
    )
    features["texture_radius_interaction"] = (
        features["mean texture"] * features["mean radius"]
    )

    target = (dataset.target == 0).astype("int8")
    target.name = "target"

    if set(target.unique()) != {0, 1}:
        raise ValueError("The classification target must contain both classes.")

    if not np.isfinite(
        features.to_numpy(dtype=float)[
            np.isfinite(features.to_numpy(dtype=float))
        ]
    ).all():
        raise ValueError("Unexpected non-finite dataset values were found.")

    return features, target


def explore_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    output_dir: Path,
) -> dict[str, Any]:
    """Create reproducible descriptive statistics and correlation reports."""
    distribution = target.value_counts().sort_index()

    class_distribution = {
        "benign": int(distribution.get(0, 0)),
        "malignant": int(distribution.get(1, 0)),
    }

    missing = features.isna().sum().astype(int).to_dict()
    descriptive_statistics = features.describe().round(5).to_dict()

    correlation = features.corr(numeric_only=True).abs()
    upper_triangle = np.triu(
        np.ones(correlation.shape, dtype=bool),
        k=1,
    )
    pairs = (
        correlation.where(upper_triangle)
        .stack()
        .sort_values(ascending=False)
        .head(15)
    )

    correlation_records = [
        {
            "feature_a": str(pair[0]),
            "feature_b": str(pair[1]),
            "absolute_correlation": float(value),
        }
        for pair, value in pairs.items()
    ]

    profile: dict[str, Any] = {
        "dataset_name": "Wisconsin Diagnostic Breast Cancer",
        "source": "scikit-learn built-in breast cancer dataset",
        "rows": int(len(features)),
        "feature_count": int(features.shape[1]),
        "target_name": "target",
        "positive_class": POSITIVE_CLASS,
        "class_distribution": class_distribution,
        "missing_values_by_feature": missing,
        "descriptive_statistics": descriptive_statistics,
        "top_absolute_feature_correlations": correlation_records,
    }

    (output_dir / "dataset_profile.json").write_text(
        json.dumps(profile, indent=2, default=json_value),
        encoding="utf-8",
    )

    pd.DataFrame(correlation_records).to_csv(
        output_dir / "feature_correlations.csv",
        index=False,
    )

    LOG.info("Dataset dimensions: %s rows, %s features", *features.shape)
    LOG.info("Class distribution: %s", class_distribution)
    LOG.info(
        "Features with missing values: %s",
        sum(count > 0 for count in missing.values()),
    )

    return profile


def json_value(value: Any) -> Any:
    """Convert common NumPy values into JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def build_pipeline(estimator: Any, scale: bool) -> Pipeline:
    """
    Keep learned preprocessing inside the estimator pipeline.
    Cross-validation therefore fits imputation and scaling independently
    within each training fold.
    """
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", estimator))
    return Pipeline(steps)


def create_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "LogisticRegression": build_pipeline(
            LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=random_state,
            ),
            scale=True,
        ),
        "RandomForest": build_pipeline(
            RandomForestClassifier(
                n_estimators=350,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
            scale=False,
        ),
        "HistGradientBoosting": build_pipeline(
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.08,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=random_state,
            ),
            scale=False,
        ),
    }


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict[str, float], dict[str, int]]:
    """Calculate binary classification metrics for malignant label 1."""
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())

    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, zero_division=0)
        ),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probabilities)),
    }

    confusion = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    return metrics, confusion


def evaluate_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Evaluate a majority-class baseline to establish a reference point."""
    baseline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )
    baseline.fit(X_train, y_train)

    predictions = baseline.predict(X_test)
    probabilities = baseline.predict_proba(X_test)[:, 1]

    metrics, confusion = calculate_metrics(
        y_test.to_numpy(),
        predictions,
        probabilities,
    )

    return {
        "model": "DummyClassifier",
        "positive_class": POSITIVE_CLASS,
        "metrics": metrics,
        "confusion_matrix": confusion,
    }


def evaluate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    folds: int,
    random_state: int,
) -> tuple[list[dict[str, Any]], dict[str, Pipeline]]:
    """
    Compare candidates using cross-validation on training data.
    The final holdout set is not used for selecting the champion.
    """
    models = create_models(random_state)
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )

    runs: list[dict[str, Any]] = []

    for model_name, pipeline in models.items():
        LOG.info("Cross-validating %s", model_name)

        cv = cross_validate(
            pipeline,
            X_train,
            y_train,
            scoring=SCORING,
            cv=splitter,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )

        cv_summary = {
            "accuracy_mean": float(np.mean(cv["test_accuracy"])),
            "precision_mean": float(np.mean(cv["test_precision"])),
            "recall_mean": float(np.mean(cv["test_recall"])),
            "f1_mean": float(np.mean(cv["test_f1"])),
            "f1_std": float(np.std(cv["test_f1"])),
            "roc_auc_mean": float(np.mean(cv["test_roc_auc"])),
            "roc_auc_std": float(np.std(cv["test_roc_auc"])),
            "folds": int(folds),
        }

        # Fit on the complete training partition only.
        pipeline.fit(X_train, y_train)

        classes = pipeline.classes_
        positive_index = int(np.flatnonzero(classes == 1)[0])
        probabilities = pipeline.predict_proba(X_test)[:, positive_index]
        predictions = (probabilities >= 0.5).astype("int8")

        metrics, confusion = calculate_metrics(
            y_test.to_numpy(),
            predictions,
            probabilities,
        )

        runs.append(
            {
                "model": model_name,
                "positive_class": POSITIVE_CLASS,
                "decision_threshold": 0.5,
                "training_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "metrics": metrics,
                "cross_validation": cv_summary,
                "confusion_matrix": confusion,
            }
        )

        LOG.info(
            "%s | validation F1 %.4f | test recall %.4f | test AUC %.4f",
            model_name,
            cv_summary["f1_mean"],
            metrics["recall"],
            metrics["roc_auc"],
        )

    return runs, models


def export_predictions(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """Export the selected model's predictions for independent inspection."""
    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
    probabilities = model.predict_proba(X_test)[:, positive_index]
    predictions = (probabilities >= 0.5).astype("int8")

    result = pd.DataFrame(
        {
            "actual_malignant": y_test.to_numpy(),
            "predicted_malignant": predictions,
            "malignant_probability": probabilities,
            "correct": y_test.to_numpy() == predictions,
        },
        index=X_test.index,
    )

    result.index.name = "source_row"
    result.to_csv(output_path)


def write_experiment_report(
    profile: dict[str, Any],
    runs: list[dict[str, Any]],
    baseline: dict[str, Any],
    champion: str,
    output_dir: Path,
    random_state: int,
    test_size: float,
) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "dataset": {
            "name": profile["dataset_name"],
            "source": profile["source"],
            "rows": profile["rows"],
            "feature_count": profile["feature_count"],
            "target": "target",
            "positive_class": POSITIVE_CLASS,
            "positive_label": 1,
            "random_state": random_state,
            "test_size": test_size,
        },
        "baseline": baseline,
        "champion_model": champion,
        "selection_rule": (
            "Highest mean cross-validation F1 on the training partition; "
            "ROC AUC is the tie-breaker."
        ),
        "runs": runs,
        "notes": [
            "The final test partition is reserved for final reporting.",
            "Imputation and scaling are fitted inside each training fold.",
            "The positive class is malignant.",
            "This educational benchmark is not a diagnostic system.",
        ],
    }

    output_path = output_dir / "metrics.json"
    output_path.write_text(
        json.dumps(report, indent=2, default=json_value),
        encoding="utf-8",
    )

    rows = []
    for run in runs:
        rows.append(
            {
                "model": run["model"],
                **run["metrics"],
                "cv_f1_mean": run["cross_validation"]["f1_mean"],
                "cv_f1_std": run["cross_validation"]["f1_std"],
                "cv_roc_auc_mean": run["cross_validation"]["roc_auc_mean"],
            }
        )

    pd.DataFrame(rows).sort_values(
        by=["cv_f1_mean", "cv_roc_auc_mean"],
        ascending=False,
    ).to_csv(output_dir / "model_comparison.csv", index=False)

    return report


def main() -> None:
    configure_logging()
    args = parse_arguments()
    validate_configuration(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading dataset")
    features, target = load_dataset()

    if target.value_counts().min() < args.folds:
        raise ValueError(
            "The smallest class must have at least as many records "
            "as the number of cross-validation folds."
        )

    # Save the reproducible input table used by the other implementations.
    complete_dataset = features.copy()
    complete_dataset["target"] = target
    complete_dataset.to_csv(
        args.output_dir / "dataset.csv",
        index=False,
    )

    profile = explore_dataset(features, target, args.output_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
    )

    LOG.info(
        "Training rows: %d | Holdout rows: %d",
        len(X_train),
        len(X_test),
    )

    baseline = evaluate_baseline(X_train, y_train, X_test, y_test)

    runs, pipelines = evaluate_candidates(
        X_train,
        y_train,
        X_test,
        y_test,
        args.folds,
        args.random_state,
    )

    # Model selection is based on training-only cross-validation statistics.
    champion_run = max(
        runs,
        key=lambda run: (
            run["cross_validation"]["f1_mean"],
            run["cross_validation"]["roc_auc_mean"],
        ),
    )
    champion_name = champion_run["model"]
    champion_pipeline = pipelines[champion_name]

    champion_run["selected_as_champion"] = True
    for run in runs:
        if run["model"] != champion_name:
            run["selected_as_champion"] = False

    export_predictions(
        champion_pipeline,
        X_test,
        y_test,
        args.output_dir / "champion_predictions.csv",
    )

    # Joblib artifacts use Python object serialization. Only load artifacts
    # produced by a trusted source; loading untrusted files can execute code.
    model_path = args.output_dir / "champion_model.joblib"
    joblib.dump(champion_pipeline, model_path, compress=3)

    report = write_experiment_report(
        profile,
        runs,
        baseline,
        champion_name,
        args.output_dir,
        args.random_state,
        args.test_size,
    )

    LOG.info("scikit-learn version: %s", sklearn.__version__)
    LOG.info("Champion selected by training CV: %s", champion_name)
    LOG.info("Baseline test F1: %.4f", baseline["metrics"]["f1"])
    LOG.info(
        "Champion test F1: %.4f",
        champion_run["metrics"]["f1"],
    )
    LOG.info("Artifacts written to: %s", args.output_dir.resolve())

    print("\nFinal model comparison")
    print(pd.read_csv(args.output_dir / "model_comparison.csv").to_string(index=False))
    print("\nChampion:", report["champion_model"])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, OSError) as exc:
        logging.basicConfig(level=logging.ERROR)
        LOG.exception("Experiment failed: %s", exc)
        raise SystemExit(1) from exc
