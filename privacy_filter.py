import argparse
import os
from typing import Any

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

"""Local privacy-filter service runner.

Provides a small CLI to load a token-classification model from a local
snapshot and run privacy-sensitive token detection on input text.
"""

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns an ArgumentParser configured with the options accepted by the
    script: model path, input text, offline flags, aggregation strategy,
    tokenizer max length.
    """
    parser = argparse.ArgumentParser(
        description="Run local privacy-filter token classification on CPU."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local model directory or local snapshot path.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Input text to classify.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Enable strict offline mode (HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE).",
    )
    parser.add_argument(
        "--allow-online",
        action="store_true",
        help="Allow online fallback by disabling local_files_only.",
    )
    parser.add_argument(
        "--aggregation-strategy",
        default="simple",
        choices=["none", "simple", "first", "average", "max"],
        help="Aggregation strategy for token-classification pipeline.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Tokenizer max length for truncation.",
    )
    return parser


def _configure_offline_mode(enable: bool) -> None:
    """Enable strict offline mode by setting hub-related environment vars.

    When `enable` is True this sets `HF_HUB_OFFLINE` and
    `TRANSFORMERS_OFFLINE` to '1' to prevent any network access to the
    Hugging Face Hub during model/tokenizer loading.
    """
    if not enable:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _load_model_and_tokenizer(
    model_path: str, local_files_only: bool, max_length: int
) -> tuple[Any, Any]:
    """Load tokenizer and model from `model_path`, 

    The tokenizer is loaded with `local_files_only` behavior and its
    `model_max_length` is set to `max_length` so the token-classification
    pipeline will perform truncation consistently. On failure it falls back 
    to a normal CPU model load. Returns (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    tokenizer.model_max_length = max_length

    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    model.to("cpu")
    model.eval()
    return model, tokenizer


def main() -> None:
    """CLI entrypoint: parse args, configure offline mode, load model, run.

    This function wires together argument parsing, optional offline mode,
    model/tokenizer loading, pipeline
    construction for `token-classification`, and runs the classifier on
    the provided input text, printing metadata and predictions.
    """
    parser = build_parser()
    args = parser.parse_args()

    _configure_offline_mode(args.offline)
    local_files_only = not args.allow_online

    model_path = os.path.abspath(args.model_path)
    model, tokenizer = _load_model_and_tokenizer(
        model_path=model_path,
        local_files_only=local_files_only,
        max_length=args.max_length,
    )

    classifier = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy=args.aggregation_strategy,
        device=-1,
    )

    results = classifier(args.text)

    print(f"model_path={model_path}")
    print(f"offline={args.offline}")
    print(f"local_files_only={local_files_only}")
    print("predictions=")
    for item in results:
        print(item)


if __name__ == "__main__":
    main()
