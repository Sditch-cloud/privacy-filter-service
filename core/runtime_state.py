import argparse
import os
from typing import Any

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TokenClassificationPipeline,
    pipeline,
)

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

def get_ml_model(
    offline: bool,
    model_path: str,
    max_length: int,
    aggregation_strategy: str = "simple",
    device: int = -1,
) -> TokenClassificationPipeline:
    """Return a configured `TokenClassificationPipeline`.

    Parameters `aggregation_strategy` and `device` are forwarded to the
    transformers `pipeline` call so callers can control aggregation and
    whether the pipeline runs on CPU (`-1`) or a GPU index (e.g. `0`).
    """

    _configure_offline_mode(offline)
    model_path = os.path.abspath(model_path)
    model, tokenizer = _load_model_and_tokenizer(
        model_path=model_path,
        local_files_only=True,
        max_length=max_length,
    )

    classifier = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy=aggregation_strategy,
        device=device,
    )

    return classifier