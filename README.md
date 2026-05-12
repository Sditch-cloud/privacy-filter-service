# privacy-filter-service

基于 Hugging Face 官方下载与离线模式说明的本地推理服务，目标是：

- 在线阶段下载并固定模型快照
- 离线阶段只从本地目录加载模型
- 通过 FastAPI 提供 `openai/privacy-filter` 的 token-classification HTTP 接口

## 1. 环境准备

推荐使用 `uv`：

```bash
uv sync
```

如果只安装 CPU PyTorch，可按官方建议使用 CPU wheel：

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e .
```

## 2. 启动 FastAPI 服务

服务启动时会在 `lifespan` 中执行：

- 下载模型（默认仓库：`openai/privacy-filter`）
- 初始化全局唯一 pipeline（CPU）
- 对外提供 `/analyze` 接口

```bash
uv run fastapi dev main.py
```

默认地址为 `http://127.0.0.1:8000`。

可用接口：

- `GET /`：健康检查示例，返回 `{"Hello": "World"}`
- `POST /analyze`：文本隐私信息检测

`POST /analyze` 请求示例：

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
	-H "Content-Type: application/json" \
	-d '{"text":"My email is alice@example.com and phone is 123-456-7890"}'
```

返回示例：

```json
{
	"predictions": [
		{
			"entity_group": "private_email",
			"score": 0.9998,
			"word": " alice@example.com",
			"start": 12,
			"end": 30
		}
	]
}
```

说明：

- 当前实现默认按离线方式加载模型（见 `main.py` 启动参数）
- 模型实例保存在 `app.state.ml_model`，请求复用同一个实例
- 若模型初始化失败，服务会在启动阶段直接报错


## 3. 官方文档对应关系

- 下载与缓存：`huggingface_hub.snapshot_download`（含 `local_dir`、过滤、`dry_run`）
- 离线模式：`HF_HUB_OFFLINE=1` + `local_files_only=True`

参考：

- https://huggingface.co/docs/huggingface_hub/guides/download
- https://huggingface.co/docs/transformers/en/installation#offline-mode
