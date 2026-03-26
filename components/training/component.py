"""Training pipeline component.

This component fine-tunes a Hugging Face model on the tokenized dataset
produced by preprocessing and saves the model artifact for downstream
serving or evaluation.
"""

import os

import click
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def train(
    train_data_path: str,
    model_output_path: str,
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
    num_epochs: int = 3,
    per_device_batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> None:
    """Fine-tune a Hugging Face model on the preprocessed dataset.

    Args:
        train_data_path:      Path to the tokenized training Arrow dataset.
        model_output_path:    Destination directory for the saved model and tokenizer.
        model_name:           Hugging Face Hub model identifier to fine-tune from.
        num_labels:           Number of classification labels.
        num_epochs:           Number of training epochs.
        per_device_batch_size: Batch size per device.
        learning_rate:        Learning rate for AdamW.
    """
    print(f"[training] Loading training data from: {train_data_path}")
    dataset = load_from_disk(train_data_path)
    dataset = dataset.with_format("torch")

    print(f"[training] Loading model: {model_name!r} (num_labels={num_labels})")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # -----------------------------------------------------------------
    # Adjust TrainingArguments for your environment, e.g.:
    #   fp16=True for GPU mixed-precision training
    #   report_to="wandb" / "tensorboard" for experiment tracking
    # -----------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=model_output_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        save_strategy="epoch",
        logging_steps=50,
        disable_tqdm=True,  # clean output in pipeline logs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("[training] Starting fine-tuning ...")
    trainer.train()

    os.makedirs(model_output_path, exist_ok=True)
    trainer.save_model(model_output_path)
    tokenizer.save_pretrained(model_output_path)

    print(f"[training] Model and tokenizer saved to: {model_output_path}")


@click.command()
@click.option("--train-data-path", required=True, help="Tokenized training Arrow dataset directory")
@click.option("--model-output-path", required=True, help="Directory to save the fine-tuned model")
@click.option(
    "--model-name",
    default="distilbert-base-uncased",
    show_default=True,
    help="Hugging Face Hub model identifier",
)
@click.option(
    "--num-labels", type=int, default=2, show_default=True, help="Number of classification labels"
)
@click.option(
    "--num-epochs", type=int, default=3, show_default=True, help="Number of training epochs"
)
@click.option(
    "--per-device-batch-size",
    type=int,
    default=16,
    show_default=True,
    help="Batch size per device",
)
@click.option(
    "--learning-rate",
    type=float,
    default=2e-5,
    show_default=True,
    help="Learning rate",
)
def main(
    train_data_path: str,
    model_output_path: str,
    model_name: str,
    num_labels: int,
    num_epochs: int,
    per_device_batch_size: int,
    learning_rate: float,
) -> None:
    train(
        train_data_path,
        model_output_path,
        model_name,
        num_labels,
        num_epochs,
        per_device_batch_size,
        learning_rate,
    )


if __name__ == "__main__":
    main()
