import argparse
import os
from typing import Any

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--try-4bit",
        action="store_true",
        help="Try loading in 4-bit quantization first. Falls back to normal CPU load on failure.",
    )
    return parser


def _configure_offline_mode(enable: bool) -> None:
    if not enable:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _load_model_and_tokenizer(
    model_path: str, local_files_only: bool, try_4bit: bool, max_length: int
) -> tuple[Any, Any, bool]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    tokenizer.model_max_length = max_length

    four_bit_loaded = False
    if try_4bit:
        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float32,
            )
            model = AutoModelForTokenClassification.from_pretrained(
                model_path,
                local_files_only=local_files_only,
                quantization_config=quantization_config,
                device_map={"": "cpu"},
            )
            four_bit_loaded = True
            return model, tokenizer, four_bit_loaded
        except Exception as exc:
            print("4bit load failed on current CPU/runtime. Falling back to regular CPU load.")
            print(f"4bit_error={exc}")

    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    model.to("cpu")
    model.eval()
    return model, tokenizer, four_bit_loaded


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _configure_offline_mode(args.offline)
    local_files_only = not args.allow_online

    model_path = os.path.abspath(args.model_path)
    model, tokenizer, four_bit_loaded = _load_model_and_tokenizer(
        model_path=model_path,
        local_files_only=local_files_only,
        try_4bit=args.try_4bit,
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
    print(f"loaded_4bit={four_bit_loaded}")
    print("predictions=")
    for item in results:
        print(item)


if __name__ == "__main__":
    main()
