from contextlib import asynccontextmanager
import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.model_downloader import download_model
from core.runtime_state import get_ml_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 下载模型（在服务正式 startup 之前执行）
    repo_id = "openai/privacy-filter"
    cache_dir = "./models-cache"
    mirror = "https://hf-mirror.com/"
    allow_patterns = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
        "model.sig",
        "viterbi_calibration.json",
    ]

    snapshot_path = await asyncio.to_thread(
        download_model,
        repo_id=repo_id,
        cache_dir=cache_dir,
        mirror=mirror,
        allow_patterns=allow_patterns,
    )

    if not snapshot_path:
        raise RuntimeError("Model download failed; aborting startup")

    # 初始化全局唯一模型实例（在后台线程中运行以避免阻塞 event loop）
    try:
        ml_model = await asyncio.to_thread(
            get_ml_model,
            True,  # offline
            snapshot_path,
            1024,  # max_length
            "simple",  # aggregation_strategy
            -1,  # device (CPU)
        )
    except Exception as e:
        raise RuntimeError(f"Model initialization failed: {e}")

    # 将模型作为应用状态保存，作为全局唯一实例使用
    app.state.ml_model = ml_model

    yield


app = FastAPI(lifespan=lifespan)


def _to_json_safe(value: Any) -> Any:
    """Recursively convert non-JSON-native scalars (e.g. numpy) to Python types."""
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_json_safe(v) for v in value)

    # numpy scalar types expose `.item()` to convert to Python primitives.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return item_method()
        except Exception:
            pass

    return value


@app.get("/")
def read_root():
    return {"Hello": "World"}


class TextRequest(BaseModel):
    text: str


@app.post("/analyze")
async def analyze_text(req: TextRequest):
    """Analyze `text` using the global ML model stored in `app.state.ml_model`.

    The pipeline invocation runs in a background thread to avoid blocking
    the async event loop.
    """
    classifier = getattr(app.state, "ml_model", None)
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        results = await asyncio.to_thread(classifier, req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    return {"predictions": _to_json_safe(results)}

