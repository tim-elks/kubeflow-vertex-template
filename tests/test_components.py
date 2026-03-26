"""Unit tests for pipeline component scripts.

These tests exercise the core logic of each component (preprocessing,
training, evaluation) using small in-memory datasets so no GCP
credentials or real GCS buckets are required.
"""

import json
import os
import pickle
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Make component source modules importable
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESS_SRC = os.path.join(REPO_ROOT, "components", "data_preprocessing", "src")
TRAINING_SRC = os.path.join(REPO_ROOT, "components", "training", "src")
EVALUATION_SRC = os.path.join(REPO_ROOT, "components", "evaluation", "src")

for path in (PREPROCESS_SRC, TRAINING_SRC, EVALUATION_SRC):
    if path not in sys.path:
        sys.path.insert(0, path)

import evaluate as evaluate_module
import preprocess as preprocess_module
import train as train_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_csv(tmp_path):
    """Create a small CSV dataset for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    df["label"] = y
    csv_path = tmp_path / "raw_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture()
def trained_model(tmp_path, sample_csv):
    """Train and pickle a model so evaluation tests can use it."""
    df = pd.read_csv(sample_csv)
    X = df.drop(columns=["label"])
    y = df["label"]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return str(model_path)


# ---------------------------------------------------------------------------
# GCS helper stubs
# ---------------------------------------------------------------------------


def _mock_download(gcs_uri: str, local_path: str, file_map: dict) -> None:
    """Copy from file_map instead of real GCS."""
    import shutil
    shutil.copy(file_map[gcs_uri], local_path)


def _mock_upload(local_path: str, gcs_uri: str, captured: dict) -> None:
    """Capture uploads instead of sending to GCS."""
    import shutil
    captured[gcs_uri] = local_path


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_preprocess_creates_train_and_test_files(self, sample_csv, tmp_path):
        """Preprocessing should split the dataset and write two CSV files."""
        train_out = str(tmp_path / "train.csv")
        test_out = str(tmp_path / "test.csv")
        uploads: dict = {}

        file_map = {
            "gs://bucket/raw.csv": sample_csv,
        }

        with (
            patch.object(
                preprocess_module,
                "download_from_gcs",
                side_effect=lambda u, p: _mock_download(u, p, file_map),
            ),
            patch.object(
                preprocess_module,
                "upload_to_gcs",
                side_effect=lambda lp, u: _mock_upload(lp, u, uploads),
            ),
        ):
            preprocess_module.preprocess(
                input_data_uri="gs://bucket/raw.csv",
                train_output_uri="gs://bucket/train.csv",
                test_output_uri="gs://bucket/test.csv",
                target_column="label",
                test_size=0.2,
                random_state=42,
            )

        assert "gs://bucket/train.csv" in uploads
        assert "gs://bucket/test.csv" in uploads

        train_df = pd.read_csv(uploads["gs://bucket/train.csv"])
        test_df = pd.read_csv(uploads["gs://bucket/test.csv"])

        assert "label" in train_df.columns
        assert "label" in test_df.columns
        assert len(train_df) > len(test_df)

    def test_preprocess_drops_nan_rows(self, tmp_path):
        """Preprocessing should remove rows with missing values."""
        df = pd.DataFrame({
            "f1": [1.0, None, 3.0, 4.0, 5.0],
            "f2": [2.0, 2.0, 3.0, 4.0, 5.0],
            "label": [0, 1, 0, 1, 0],
        })
        raw_path = str(tmp_path / "raw.csv")
        df.to_csv(raw_path, index=False)
        uploads: dict = {}

        file_map = {"gs://bucket/raw.csv": raw_path}

        with (
            patch.object(
                preprocess_module,
                "download_from_gcs",
                side_effect=lambda u, p: _mock_download(u, p, file_map),
            ),
            patch.object(
                preprocess_module,
                "upload_to_gcs",
                side_effect=lambda lp, u: _mock_upload(lp, u, uploads),
            ),
        ):
            preprocess_module.preprocess(
                input_data_uri="gs://bucket/raw.csv",
                train_output_uri="gs://bucket/train.csv",
                test_output_uri="gs://bucket/test.csv",
                target_column="label",
                test_size=0.4,
                random_state=0,
            )

        train_df = pd.read_csv(uploads["gs://bucket/train.csv"])
        test_df = pd.read_csv(uploads["gs://bucket/test.csv"])
        total_rows = len(train_df) + len(test_df)
        # One row had NaN, so we expect 4 rows total
        assert total_rows == 4


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------


class TestTrain:
    def test_train_produces_model_and_metrics(self, sample_csv, tmp_path):
        """Training should output a pickled model and a JSON metrics file."""
        model_out = str(tmp_path / "model.pkl")
        metrics_out_local = str(tmp_path / "train_metrics.json")
        uploads: dict = {}

        file_map = {"gs://bucket/train.csv": sample_csv}

        with (
            patch.object(
                train_module,
                "download_from_gcs",
                side_effect=lambda u, p: _mock_download(u, p, file_map),
            ),
            patch.object(
                train_module,
                "upload_to_gcs",
                side_effect=lambda lp, u: _mock_upload(lp, u, uploads),
            ),
        ):
            train_module.train(
                train_data_uri="gs://bucket/train.csv",
                model_output_uri="gs://bucket/model.pkl",
                metrics_output_uri="gs://bucket/metrics.json",
                target_column="label",
                n_estimators=10,
                max_depth=5,
                random_state=42,
            )

        assert "gs://bucket/model.pkl" in uploads
        assert "gs://bucket/metrics.json" in uploads

        # Verify model can be loaded
        with open(uploads["gs://bucket/model.pkl"], "rb") as f:
            model = pickle.load(f)
        assert hasattr(model, "predict")

        # Verify metrics JSON structure
        with open(uploads["gs://bucket/metrics.json"]) as f:
            metrics = json.load(f)
        assert "train_accuracy" in metrics
        assert 0.0 <= metrics["train_accuracy"] <= 1.0
        assert "hyperparameters" in metrics

    def test_train_accuracy_is_valid(self, sample_csv, tmp_path):
        """Training accuracy should be a float between 0 and 1."""
        uploads: dict = {}
        file_map = {"gs://bucket/train.csv": sample_csv}

        with (
            patch.object(train_module, "download_from_gcs",
                         side_effect=lambda u, p: _mock_download(u, p, file_map)),
            patch.object(train_module, "upload_to_gcs",
                         side_effect=lambda lp, u: _mock_upload(lp, u, uploads)),
        ):
            train_module.train(
                train_data_uri="gs://bucket/train.csv",
                model_output_uri="gs://bucket/model.pkl",
                metrics_output_uri="gs://bucket/metrics.json",
                target_column="label",
            )

        with open(uploads["gs://bucket/metrics.json"]) as f:
            metrics = json.load(f)
        assert 0.0 <= metrics["train_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_evaluate_writes_accuracy_file(self, sample_csv, trained_model, tmp_path):
        """Evaluation should write a scalar accuracy to the output path."""
        accuracy_path = str(tmp_path / "accuracy.txt")
        uploads: dict = {}

        file_map = {
            "gs://bucket/model.pkl": trained_model,
            "gs://bucket/test.csv": sample_csv,
        }

        with (
            patch.object(
                evaluate_module,
                "download_from_gcs",
                side_effect=lambda u, p: _mock_download(u, p, file_map),
            ),
            patch.object(
                evaluate_module,
                "upload_to_gcs",
                side_effect=lambda lp, u: _mock_upload(lp, u, uploads),
            ),
        ):
            evaluate_module.evaluate(
                model_uri="gs://bucket/model.pkl",
                test_data_uri="gs://bucket/test.csv",
                metrics_output_uri="gs://bucket/eval_metrics.json",
                accuracy_output_path=accuracy_path,
                target_column="label",
            )

        assert os.path.exists(accuracy_path)
        with open(accuracy_path) as f:
            accuracy = float(f.read().strip())
        assert 0.0 <= accuracy <= 1.0

    def test_evaluate_metrics_json_structure(self, sample_csv, trained_model, tmp_path):
        """Evaluation metrics JSON should contain expected keys."""
        accuracy_path = str(tmp_path / "accuracy.txt")
        uploads: dict = {}

        file_map = {
            "gs://bucket/model.pkl": trained_model,
            "gs://bucket/test.csv": sample_csv,
        }

        with (
            patch.object(evaluate_module, "download_from_gcs",
                         side_effect=lambda u, p: _mock_download(u, p, file_map)),
            patch.object(evaluate_module, "upload_to_gcs",
                         side_effect=lambda lp, u: _mock_upload(lp, u, uploads)),
        ):
            evaluate_module.evaluate(
                model_uri="gs://bucket/model.pkl",
                test_data_uri="gs://bucket/test.csv",
                metrics_output_uri="gs://bucket/eval_metrics.json",
                accuracy_output_path=accuracy_path,
                target_column="label",
            )

        assert "gs://bucket/eval_metrics.json" in uploads
        with open(uploads["gs://bucket/eval_metrics.json"]) as f:
            metrics = json.load(f)

        assert "test_accuracy" in metrics
        assert "classification_report" in metrics
        assert "confusion_matrix" in metrics
