"""Data ingestion pipeline component.

This component downloads a dataset from the Hugging Face Hub (or a local
path during testing) and writes it to disk as a Hugging Face Arrow dataset
that downstream components can consume.
"""

import os

import click
from datasets import load_dataset


def ingest_data(dataset_name: str, output_path: str, config_name: str = "default") -> None:
    """Load a dataset from the Hugging Face Hub and save it to output_path.

    Args:
        dataset_name: Hugging Face Hub dataset identifier, e.g. "imdb" or
                      "squad".  Set to a local directory path to load from disk.
        output_path:  Destination directory for the saved Arrow dataset.
        config_name:  Dataset configuration / subset name (default: "default").
    """
    print(f"[data_ingestion] Loading dataset: {dataset_name!r} (config={config_name!r})")

    # -----------------------------------------------------------------
    # Replace or extend this with your own loading logic, e.g.:
    #   dataset = load_dataset("csv", data_files={"train": "gs://..."})
    #   dataset = load_dataset("json", data_files={"train": gcs_path})
    # -----------------------------------------------------------------
    dataset = load_dataset(dataset_name, config_name if config_name != "default" else None)

    os.makedirs(output_path, exist_ok=True)
    dataset.save_to_disk(output_path)

    splits = {split: len(ds) for split, ds in dataset.items()}
    print(f"[data_ingestion] Dataset saved to: {output_path}")
    print(f"[data_ingestion] Splits: {splits}")


@click.command()
@click.option("--dataset-name", required=True, help="Hugging Face Hub dataset name or local path")
@click.option("--output-path", required=True, help="Directory to save the Arrow dataset")
@click.option(
    "--config-name",
    default="default",
    show_default=True,
    help="Dataset configuration / subset name",
)
def main(dataset_name: str, output_path: str, config_name: str) -> None:
    ingest_data(dataset_name, output_path, config_name)


if __name__ == "__main__":
    main()
