#!/usr/bin/env python3
"""
Production-Style Classical Machine Learning Pipeline

Problem domain:
Customer churn prediction using classical supervised learning.

Pipeline:
Data ingestion -> Validation -> Preprocessing -> Training ->
Cross-validation -> Hyperparameter selection -> Threshold selection ->
Testing -> Evaluation -> Model persistence -> Prediction -> Monitoring.

Dependencies:
    pip install numpy pandas scikit-learn joblib

Usage:
    python ml_pipeline.py train
    python ml_pipeline.py train --data customers.csv
    python ml_pipeline.py predict --data new_customers.csv
    python ml_pipeline.py monitor --data customers.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@dataclass(frozen=True)
class Config:
    target: str = "churn"
    id_column: str = "customer_id"
    test_size: float = 0.20
    validation_size: float = 0.20
    random_state: int = 42
    cv_folds: int = 3
    threshold: float = 0.50
    minimum_rows: int = 50
    minimum_roc_auc: float = 0.55
    artifact_directory: str = "ml_artifacts"


CONFIG = Config()

ARTIFACT_DIR = Path(CONFIG.artifact_directory)
MODEL_PATH = ARTIFACT_DIR / "churn_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
PREDICTION_PATH = ARTIFACT_DIR / "predictions.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ml_pipeline.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger("ml_pipeline")


class PipelineError(Exception):
    """Base exception for pipeline failures."""


class DataValidationError(PipelineError):
    """Raised when the dataset violates its contract."""


class ModelQualityError(PipelineError):
    """Raised when the trained model fails quality requirements."""


# ---------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------

def generate_dataset(rows: int = 1500) -> pd.DataFrame:
    """
    Generate reproducible customer data.

    This dataset is for demonstrating the complete pipeline.
    Production training should use an approved, versioned dataset.
    """

    if rows < CONFIG.minimum_rows:
        raise ValueError(
            f"At least {CONFIG.minimum_rows} rows are required."
        )

    rng = np.random.default_rng(CONFIG.random_state)

    age = rng.integers(18, 75, rows)
    tenure = rng.integers(1, 73, rows)
    monthly_charges = rng.uniform(20, 150, rows).round(2)
    support_calls = rng.poisson(2, rows)
    contract = rng.choice(
        ["monthly", "annual", "two_year"],
        rows,
        p=[0.55, 0.30, 0.15],
    )
    payment_method = rng.choice(
        ["card", "bank_transfer", "electronic"],
        rows,
    )
    internet_service = rng.choice(
        ["fiber", "dsl", "none"],
        rows,
    )

    # Synthetic probability model.
    logits = (
        -2.0
        + 0.025 * support_calls
        + 0.018 * monthly_charges
        - 0.025 * tenure
        + 0.85 * (contract == "monthly")
        - 0.60 * (contract == "two_year")
        + 0.35 * (internet_service == "fiber")
    )

    probability = 1 / (1 + np.exp(-logits))
    churn = rng.binomial(1, probability)

    return pd.DataFrame({
        "customer_id": np.arange(100001, 100001 + rows),
        "age": age,
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "support_calls": support_calls,
        "contract": contract,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "churn": churn,
    })


# ---------------------------------------------------------
# Data ingestion and validation
# ---------------------------------------------------------

def load_data(path: str | None = None) -> pd.DataFrame:
    if path is None:
        logger.info("Generating demonstration dataset.")
        return generate_dataset()

    source = Path(path)

    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")

    if source.suffix.lower() != ".csv":
        raise DataValidationError("Only CSV datasets are supported.")

    if source.stat().st_size > 250 * 1024 * 1024:
        raise DataValidationError("Dataset exceeds the 250 MB limit.")

    try:
        return pd.read_csv(source)
    except Exception as exc:
        raise DataValidationError(
            f"Unable to read CSV dataset: {exc}"
        ) from exc


def validate_dataset(
    df: pd.DataFrame,
    training: bool = True,
) -> pd.DataFrame:

    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("Input must be a pandas DataFrame.")

    if df.empty:
        raise DataValidationError("Dataset is empty.")

    if len(df) < CONFIG.minimum_rows:
        raise DataValidationError(
            f"Dataset requires at least {CONFIG.minimum_rows} rows."
        )

    if df.columns.duplicated().any():
        raise DataValidationError("Duplicate column names detected.")

    if CONFIG.id_column not in df.columns:
        raise DataValidationError(
            f"Missing identifier column: {CONFIG.id_column}"
        )

    if training and CONFIG.target not in df.columns:
        raise DataValidationError(
            f"Missing target column: {CONFIG.target}"
        )

    if not training and CONFIG.target in df.columns:
        df = df.drop(columns=[CONFIG.target])

    if df[CONFIG.id_column].isna().any():
        raise DataValidationError("Customer identifiers cannot be null.")

    if df[CONFIG.id_column].duplicated().any():
        raise DataValidationError("Duplicate customer identifiers detected.")

    if training:
        if df[CONFIG.target].isna().any():
            raise DataValidationError("Target contains missing values.")

        labels = set(df[CONFIG.target].unique())

        if not labels.issubset({0, 1}):
            raise DataValidationError(
                "Target must contain binary values 0 and 1."
            )

        if len(labels) != 2:
            raise DataValidationError(
                "Both target classes must be represented."
            )

    features = df.drop(
        columns=[CONFIG.target, CONFIG.id_column],
        errors="ignore",
    )

    if features.empty:
        raise DataValidationError("No predictive features available.")

    for column in features.select_dtypes(
        include=["object", "category", "string"]
    ).columns:
        if features[column].nunique(dropna=True) > 100:
            raise DataValidationError(
                f"High-cardinality categorical feature: {column}"
            )

    for column in features.select_dtypes(include=np.number).columns:
        values = features[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise DataValidationError(
                f"Non-finite numeric values detected in {column}."
            )

    return df.copy()


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """
    Generate a deterministic fingerprint of the dataset.

    Useful for identifying training inputs and auditing runs.
    """

    hashed = pd.util.hash_pandas_object(
        df,
        index=True,
    ).values.tobytes()

    return hashlib.sha256(hashed).hexdigest()


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

def build_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:

    numerical_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
        ),
    ])

    transformers = []

    if numerical_columns:
        transformers.append(
            ("numeric", numerical_pipeline, numerical_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise DataValidationError("No usable feature columns.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ---------------------------------------------------------
# Model construction and selection
# ---------------------------------------------------------

def build_search(
    X: pd.DataFrame,
) -> GridSearchCV:

    preprocessor = build_preprocessor(X)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=CONFIG.random_state,
            ),
        ),
    ])

    parameter_grid = [
        {
            "classifier": [
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=CONFIG.random_state,
                )
            ],
            "classifier__C": [0.1, 1.0, 10.0],
        },
        {
            "classifier": [
                RandomForestClassifier(
                    random_state=CONFIG.random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                )
            ],
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 10],
            "classifier__min_samples_leaf": [2, 5],
        },
        {
            "classifier": [
                HistGradientBoostingClassifier(
                    random_state=CONFIG.random_state,
                )
            ],
            "classifier__max_iter": [100],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_leaf_nodes": [15, 31],
        },
    ]

    cv = StratifiedKFold(
        n_splits=CONFIG.cv_folds,
        shuffle=True,
        random_state=CONFIG.random_state,
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
        return_train_score=True,
    )


# ---------------------------------------------------------
# Threshold optimization
# ---------------------------------------------------------

def select_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> float:
    """
    Select a classification threshold using validation data.

    The chosen threshold maximizes F1. It is not selected
    using the held-out test set.
    """

    candidates = np.linspace(0.10, 0.90, 81)

    best_threshold = CONFIG.threshold
    best_score = -1.0

    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
) -> dict[str, Any]:

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(
            average_precision_score(y, probabilities)
        ),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(
            y,
            predictions,
        ).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


# ---------------------------------------------------------
# Model persistence
# ---------------------------------------------------------

def save_artifacts(
    model: Pipeline,
    metadata: dict[str, Any],
) -> None:

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    temporary_model = MODEL_PATH.with_suffix(".tmp")

    try:
        joblib.dump(model, temporary_model)
        os.replace(temporary_model, MODEL_PATH)

        temporary_metadata = METADATA_PATH.with_suffix(".tmp")

        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )

        os.replace(temporary_metadata, METADATA_PATH)

    except Exception:
        temporary_model.unlink(missing_ok=True)
        raise

    logger.info("Model saved to %s", MODEL_PATH)
    logger.info("Metadata saved to %s", METADATA_PATH)


def load_artifacts() -> tuple[Pipeline, dict[str, Any]]:

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts are missing. Train the model first."
        )

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    if metadata.get("target") != CONFIG.target:
        raise PipelineError("Model target does not match configuration.")

    # Only load joblib artifacts from trusted, controlled storage.
    model = joblib.load(MODEL_PATH)

    return model, metadata


# ---------------------------------------------------------
# Training workflow
# ---------------------------------------------------------

def train(
    data_path: str | None = None,
) -> dict[str, Any]:

    started = time.perf_counter()

    df = validate_dataset(
        load_data(data_path),
        training=True,
    )

    fingerprint = dataset_fingerprint(df)

    logger.info("Dataset fingerprint: %s", fingerprint)
    logger.info("Rows: %d | Columns: %d", *df.shape)

    X = df.drop(
        columns=[CONFIG.target, CONFIG.id_column]
    )

    y = df[CONFIG.target].astype(int)

    # Hold out test data before model selection.
    X_development, X_test, y_development, y_test = (
        train_test_split(
            X,
            y,
            test_size=CONFIG.test_size,
            stratify=y,
            random_state=CONFIG.random_state,
        )
    )

    # A separate validation partition determines the threshold.
    X_fit, X_validation, y_fit, y_validation = (
        train_test_split(
            X_development,
            y_development,
            test_size=CONFIG.validation_size,
            stratify=y_development,
            random_state=CONFIG.random_state,
        )
    )

    logger.info(
        "Fit: %d | Validation: %d | Test: %d",
        len(X_fit),
        len(X_validation),
        len(X_test),
    )

    search = build_search(X_fit)

    logger.info("Starting stratified cross-validation.")

    search.fit(X_fit, y_fit)

    best_model = search.best_estimator_

    validation_probabilities = best_model.predict_proba(
        X_validation
    )[:, 1]

    threshold = select_threshold(
        y_validation,
        validation_probabilities,
    )

    logger.info("Best parameters: %s", search.best_params_)
    logger.info("Best CV ROC-AUC: %.4f", search.best_score_)
    logger.info("Selected threshold: %.3f", threshold)

    # Refit the selected configuration on all development data.
    # Test observations remain completely isolated.
    final_model = search.best_estimator__

    final_model.fit(X_development, y_development)

    metrics = evaluate_model(
        final_model,
        X_test,
        y_test,
        threshold,
    )

    logger.info("Held-out test metrics: %s", metrics)

    if metrics["roc_auc"] < CONFIG.minimum_roc_auc:
        raise ModelQualityError(
            "Test ROC-AUC is below the configured release threshold."
        )

    metadata = {
        "model_name": "customer_churn_classifier",
        "model_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": CONFIG.target,
        "id_column": CONFIG.id_column,
        "feature_columns": X.columns.tolist(),
        "numeric_features": X.select_dtypes(
            include=np.number
        ).columns.tolist(),
        "categorical_features": X.select_dtypes(
            exclude=np.number
        ).columns.tolist(),
        "dataset_fingerprint": fingerprint,
        "training_rows": len(X_development),
        "test_rows": len(X_test),
        "best_parameters": search.best_params_,
        "cross_validation_roc_auc": float(search.best_score_),
        "threshold": threshold,
        "metrics": metrics,
        "training_duration_seconds": round(
            time.perf_counter() - started,
            3,
        ),
        "sklearn_version": __import__(
            "sklearn"
        ).__version__,
    }

    save_artifacts(final_model, metadata)

    return metadata


# ---------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------

def predict(data_path: str) -> pd.DataFrame:

    model, metadata = load_artifacts()

    df = validate_dataset(
        load_data(data_path),
        training=False,
    )

    expected_features = metadata["feature_columns"]

    missing = set(expected_features) - set(df.columns)
    unexpected = set(df.columns) - set(
        expected_features + [CONFIG.id_column]
    )

    if missing:
        raise DataValidationError(
            f"Missing prediction features: {sorted(missing)}"
        )

    if unexpected:
        raise DataValidationError(
            f"Unexpected prediction features: {sorted(unexpected)}"
        )

    X = df[expected_features]

    probabilities = model.predict_proba(X)[:, 1]

    threshold = float(metadata["threshold"])

    result = pd.DataFrame({
        CONFIG.id_column: df[CONFIG.id_column].values,
        "churn_probability": probabilities,
        "predicted_churn": (
            probabilities >= threshold
        ).astype(int),
        "model_version": metadata["model_version"],
        "prediction_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    })

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(PREDICTION_PATH, index=False)

    logger.info(
        "Generated %d predictions.",
        len(result),
    )

    logger.info(
        "Positive prediction rate: %.3f",
        result["predicted_churn"].mean(),
    )

    return result


# ---------------------------------------------------------
# Basic production monitoring
# ---------------------------------------------------------

def monitor(data_path: str) -> dict[str, Any]:

    _, metadata = load_artifacts()

    df = validate_dataset(
        load_data(data_path),
        training=False,
    )

    reference = set(metadata["feature_columns"])
    current = set(df.columns) - {CONFIG.id_column}

    missing = sorted(reference - current)
    unexpected = sorted(current - reference)

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "model_version": metadata["model_version"],
        "rows": len(df),
        "missing_features": missing,
        "unexpected_features": unexpected,
        "schema_healthy": not missing and not unexpected,
    }

    logger.info("Monitoring report: %s", report)

    return report


# ---------------------------------------------------------
# Unit-style checks
# ---------------------------------------------------------

def run_assertions() -> None:

    data = generate_dataset(200)

    assert len(data) == 200
    assert set(data["churn"].unique()) == {0, 1}
    assert data["customer_id"].is_unique

    validated = validate_dataset(data)

    assert validated.shape == data.shape

    threshold = select_threshold(
        pd.Series([0, 0, 1, 1]),
        np.array([0.1, 0.3, 0.7, 0.9]),
    )

    assert 0.10 <= threshold <= 0.90

    try:
        validate_dataset(
            data.drop(columns=["churn"]),
            training=True,
        )
    except DataValidationError:
        pass
    else:
        raise AssertionError("Missing target was not detected.")

    logger.info("All assertions passed.")


# ---------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Production-style classical ML pipeline."
    )

    parser.add_argument(
        "mode",
        choices=["train", "predict", "monitor", "test"],
    )

    parser.add_argument(
        "--data",
        default=None,
        help="Path to input CSV file.",
    )

    args = parser.parse_args()

    try:
        if args.mode == "train":
            metadata = train(args.data)

            print(
                json.dumps(
                    metadata["metrics"],
                    indent=2,
                )
            )

        elif args.mode == "predict":
            if not args.data:
                parser.error("--data is required for prediction.")

            print(predict(args.data).to_string(index=False))

        elif args.mode == "monitor":
            if not args.data:
                parser.error("--data is required for monitoring.")

            print(json.dumps(monitor(args.data), indent=2))

        elif args.mode == "test":
            run_assertions()

    except (
        PipelineError,
        FileNotFoundError,
        ValueError,
        OSError,
    ) as exc:
        logger.exception("Pipeline execution failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
