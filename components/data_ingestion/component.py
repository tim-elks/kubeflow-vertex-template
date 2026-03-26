"""Data ingestion pipeline component.

This component downloads or reads raw data and writes it to a GCS bucket
(or local path during testing) as a dataset artifact.
"""

import json
import os

import click


def ingest_data(data_source: str, output_path: str) -> None:
    """Download or read raw data and write it to output_path.

    Args:
        data_source: URI or identifier for the source data
                     (e.g. a GCS path, BigQuery table, or local file path).
        output_path: Destination path for the raw dataset artifact.
    """
    print(f"[data_ingestion] Reading data from: {data_source}")

    # -----------------------------------------------------------------
    # Replace the block below with your actual data-loading logic, e.g.:
    #   df = pd.read_csv(data_source)          # CSV on GCS / local
    #   df = bigquery.Client().query(sql)...   # BigQuery
    # -----------------------------------------------------------------
    sample_data = {
        "source": data_source,
        "rows": 1000,
        "features": ["feature_1", "feature_2", "label"],
        "note": "Replace this with real data loading logic.",
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sample_data, f, indent=2)

    print(f"[data_ingestion] Dataset written to: {output_path}")


@click.command()
@click.option("--data-source", required=True, help="Source data URI or path")
@click.option("--output-path", required=True, help="Output dataset path")
def main(data_source: str, output_path: str) -> None:
    ingest_data(data_source, output_path)


if __name__ == "__main__":
    main()
