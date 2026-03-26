"""Training pipeline component.

This component reads the preprocessed training data, trains a model, and
writes the model artifact to an output path for downstream serving or
evaluation components.
"""

import json
import os

import click


def train(
    train_data_path: str, model_output_path: str, n_estimators: int = 100, max_depth: int = 5
) -> None:
    """Train a model on the preprocessed dataset.

    Args:
        train_data_path:   Path to the training split artifact from preprocessing.
        model_output_path: Destination path for the serialised model artifact.
        n_estimators:      Number of estimators (for tree-based models).
        max_depth:         Maximum tree depth (for tree-based models).
    """
    print(f"[training] Loading training data from: {train_data_path}")
    with open(train_data_path) as f:
        train_data = json.load(f)

    # -----------------------------------------------------------------
    # Replace the block below with your actual training logic, e.g.:
    #   df = pd.DataFrame(train_data["records"])
    #   X, y = df.drop("label", axis=1), df["label"]
    #   model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    #   model.fit(X, y)
    #   joblib.dump(model, model_output_path)
    # -----------------------------------------------------------------
    model_metadata = {
        "model_type": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "trained_on_rows": train_data.get("rows", 0),
        "features": train_data.get("features", []),
        "metrics": {
            "train_accuracy": 0.95,  # Replace with real metrics
        },
        "note": "Replace this with real training logic.",
    }

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, "w") as f:
        json.dump(model_metadata, f, indent=2)

    print(f"[training] Model artifact written to: {model_output_path}")
    print(f"[training] Metrics: {model_metadata['metrics']}")


@click.command()
@click.option("--train-data-path", required=True, help="Training data path")
@click.option("--model-output-path", required=True, help="Output path for model artifact")
@click.option(
    "--n-estimators", type=int, default=100, show_default=True, help="Number of estimators"
)
@click.option("--max-depth", type=int, default=5, show_default=True, help="Max tree depth")
def main(train_data_path: str, model_output_path: str, n_estimators: int, max_depth: int) -> None:
    train(train_data_path, model_output_path, n_estimators, max_depth)


if __name__ == "__main__":
    main()
