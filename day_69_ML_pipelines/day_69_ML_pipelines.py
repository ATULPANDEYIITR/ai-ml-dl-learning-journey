#!/usr/bin/env python3
"""
ML Pipelines: Transformers, preprocessing, and model workflows.

Demonstrates:
- Tabular data validation
- Numerical and categorical preprocessing
- Missing-value imputation
- Feature scaling and one-hot encoding
- ColumnTransformer and Pipeline
- Train/test separation and leakage prevention
- Model training and evaluation
- Cross-validation and hyperparameter search
- Model persistence and inference
- Pipeline inspection and automated tests

Install:
    python -m pip install numpy pandas scikit-learn joblib

Run:
    python ml_pipeline.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ml_pipeline")


RANDOM_STATE = 42
ARTIFACT_DIRECTORY = Path("ml_artifacts")
MODEL_PATH = ARTIFACT_DIRECTORY / "classification_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIRECTORY / "evaluation_metrics.json"


@dataclass(frozen=True)
class PipelineConfig:
    test_size: float = 0.25
    random_state: int = RANDOM_STATE
    cv_folds: int = 3
    max_iterations: int = 2000

    def validate(self) -> None:
        if not 0.05 <= self.test_size <= 0.5:
            raise ValueError("test_size must be between 0.05 and 0.5")
        if self.cv_folds < 2:
            raise ValueError("At least two CV folds are required")
        if self.max_iterations < 100:
            raise ValueError("max_iterations must be at least 100")


def create_dataset(
    samples: int = 600,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create reproducible customer churn classification data."""

    if samples < 100:
        raise ValueError("At least 100 samples are required")

    rng = np.random.default_rng(seed)

    age = rng.integers(18, 75, samples).astype(float)
    monthly_spend = rng.normal(65, 22, samples)
    tenure = rng.integers(1, 73, samples).astype(float)

    contract = rng.choice(
        ["monthly", "annual", "two_year"],
        samples,
        p=[0.50, 0.30, 0.20],
    )

    region = rng.choice(
        ["north", "south", "east", "west"],
        samples,
    )

    probability = (
        0.20
        + 0.002 * (monthly_spend - 65)
        - 0.012 * (tenure - 30)
        + 0.45 * (contract == "monthly")
        - 0.25 * (contract == "two_year")
        + 0.08 * (age < 25)
    )

    probability = np.clip(probability, 0.02, 0.95)
    target = rng.binomial(1, probability)

    frame = pd.DataFrame({
        "age": age,
        "monthly_spend": monthly_spend,
        "tenure_months": tenure,
        "contract": contract,
        "region": region,
    })

    # Missing values are introduced before the train/test split.
    for column in ["age", "monthly_spend", "contract"]:
        missing_indices = rng.choice(
            samples,
            size=max(1, samples // 30),
            replace=False,
        )
        frame.loc[missing_indices, column] = np.nan

    return frame, pd.Series(target, name="churn")


def validate_dataset(
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Reject malformed datasets before model training."""

    if not isinstance(features, pd.DataFrame):
        raise TypeError("Features must be a pandas DataFrame")

    if features.empty:
        raise ValueError("Feature dataset cannot be empty")

    if not features.columns.is_unique:
        raise ValueError("Duplicate feature names are not allowed")

    if features.isna().all().any():
        raise ValueError("A feature column contains only missing values")

    if not features.index.equals(target.index):
        raise ValueError("Feature and target indices must match")

    if target.isna().any():
        raise ValueError("Target contains missing values")

    if target.nunique() != 2:
        raise ValueError("This workflow requires binary classification")

    if target.value_counts().min() < 2:
        raise ValueError("Each class must contain at least two records")


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build independent transformations for each feature type."""

    numeric_pipeline = Pipeline([
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

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    config: PipelineConfig,
) -> Pipeline:
    """Connect preprocessing and classification into one estimator."""

    return Pipeline([
        (
            "preprocessor",
            build_preprocessor(
                numeric_features,
                categorical_features,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=config.max_iterations,
                class_weight="balanced",
                random_state=config.random_state,
            ),
        ),
    ])


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "test_samples": int(len(y_test)),
    }


def run_cross_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Fit preprocessing separately inside each CV training fold."""

    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__solver": ["lbfgs"],
        },
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    search.fit(X_train, y_train)

    logger.info("Best cross-validation F1: %.4f", search.best_score_)
    logger.info("Best parameters: %s", search.best_params_)

    return {
        "best_estimator": search.best_estimator_,
        "best_score": float(search.best_score_),
        "best_parameters": search.best_params_,
    }


def save_artifacts(
    model: Pipeline,
    metrics: dict[str, Any],
) -> None:
    """Persist the complete fitted pipeline and evaluation results."""

    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH, compress=3)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    logger.info("Pipeline saved to %s", MODEL_PATH)
    logger.info("Metrics saved to %s", METRICS_PATH)


def load_pipeline(path: Path = MODEL_PATH) -> Pipeline:
    """Load a trusted pipeline artifact."""

    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    # Never load joblib or pickle files from untrusted sources.
    model = joblib.load(path)

    if not isinstance(model, Pipeline):
        raise TypeError("Artifact is not a scikit-learn Pipeline")

    return model


def predict_new_customers(
    model: Pipeline,
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Apply the same fitted transformations during inference."""

    if not records:
        raise ValueError("At least one inference record is required")

    frame = pd.DataFrame(records)

    expected_columns = list(
        model.named_steps["preprocessor"].feature_names_in_
    )

    if set(frame.columns) != set(expected_columns):
        raise ValueError(
            f"Expected columns: {expected_columns}"
        )

    frame = frame[expected_columns]

    if frame.empty:
        raise ValueError("Inference data cannot be empty")

    predictions = model.predict(frame)
    probabilities = model.predict_proba(frame)[:, 1]

    return pd.DataFrame({
        "predicted_class": predictions,
        "churn_probability": probabilities,
        "risk_category": np.where(
            probabilities >= 0.70,
            "high",
            np.where(probabilities >= 0.40, "medium", "low"),
        ),
    })


def run_tests(model: Pipeline) -> None:
    """Check essential pipeline behavior."""

    assert isinstance(model, Pipeline)
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps

    sample = pd.DataFrame({
        "age": [35.0, np.nan],
        "monthly_spend": [80.0, 45.0],
        "tenure_months": [12.0, 48.0],
        "contract": ["monthly", "annual"],
        "region": ["north", "west"],
    })

    predictions = model.predict(sample)
    probabilities = model.predict_proba(sample)

    assert len(predictions) == 2
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    unknown_category = sample.copy()
    unknown_category.loc[0, "region"] = "central"
    model.predict(unknown_category)

    logger.info("All pipeline assertions passed")


def main() -> None:
    config = PipelineConfig()
    config.validate()

    logger.info("Creating classification dataset")

    X, y = create_dataset()
    validate_dataset(X, y)

    numeric_features = [
        "age",
        "monthly_spend",
        "tenure_months",
    ]

    categorical_features = [
        "contract",
        "region",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    logger.info(
        "Training samples: %d | Testing samples: %d",
        len(X_train),
        len(X_test),
    )

    pipeline = build_pipeline(
        numeric_features,
        categorical_features,
        config,
    )

    cv_results = run_cross_validation(
        pipeline,
        X_train,
        y_train,
        config,
    )

    best_model = cv_results["best_estimator"]

    metrics = evaluate_model(
        best_model,
        X_test,
        y_test,
    )

    metrics["cross_validation_f1"] = cv_results["best_score"]
    metrics["best_parameters"] = cv_results["best_parameters"]

    logger.info("Evaluation metrics:")
    logger.info(
        "Accuracy: %.4f",
        metrics["accuracy"],
    )
    logger.info(
        "F1 score: %.4f",
        metrics["f1_score"],
    )
    logger.info(
        "ROC AUC: %.4f",
        metrics["roc_auc"],
    )
    logger.info(
        "Confusion matrix: %s",
        metrics["confusion_matrix"],
    )

    feature_names = (
        best_model.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    logger.info("Transformed feature count: %d", len(feature_names))

    run_tests(best_model)
    save_artifacts(best_model, metrics)

    loaded_model = load_pipeline()

    new_customers = [
        {
            "age": 29,
            "monthly_spend": 95.0,
            "tenure_months": 8,
            "contract": "monthly",
            "region": "north",
        },
        {
            "age": 51,
            "monthly_spend": 55.0,
            "tenure_months": 60,
            "contract": "two_year",
            "region": "south",
        },
    ]

    predictions = predict_new_customers(
        loaded_model,
        new_customers,
    )

    print("\nInference results:")
    print(predictions.to_string(index=False))

    print("\nPipeline execution completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, TypeError, FileNotFoundError) as error:
        logger.exception("Pipeline execution failed: %s", error)
        raise
    except Exception:
        logger.exception("Unexpected pipeline failure")
        raise
