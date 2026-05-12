# privacy-filter-service

基于 Hugging Face 官方下载与离线模式说明的本地推理方案，目标是：

- 在线阶段下载并固定模型快照
- 离线阶段只从本地目录加载模型
- CPU 上执行 `openai/privacy-filter` 的 token-classification 推理

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

## 2. 在线下载模型（官方 snapshot_download 流程）

先联网执行下载，将模型缓存到本地。

```bash
uv run python download_model.py --repo-id openai/privacy-filter --cache-dir ./models-cache
```

下载完成后会输出 `snapshot_path=...`。离线推理时优先把该路径传给 `--model-path`。

常见可选参数：

- `--revision <branch|tag|commit>` 固定版本
- `--local-dir ./models/privacy-filter` 以本地目录结构落盘
- `--allow-pattern` / `--ignore-pattern` 过滤下载文件
- `--dry-run` 只查看将下载的文件，不实际下载

如果模型受限访问，先设置 token：

```bash
set HF_TOKEN=your_token_here
```

## 3. 离线本地 CPU 推理

将 `--model-path` 指向本地 snapshot 路径或 `--local-dir` 输出目录。

```bash
uv run python main.py --model-path ./models-cache/models--openai--privacy-filter/snapshots/<commit_hash> --text "My name is Alice Smith and my email is alice@example.com" --offline
```
models-cache\models--openai--privacy-filter\snapshots\7ffa9a043d54d1be65afb281eddf0ffbe629385b
说明：

- `--offline` 会设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`
- 默认 `local_files_only=True`，不会在线回源
- 如需允许在线回源（不推荐离线验收时使用），加 `--allow-online`


## 4. 官方文档对应关系

- 下载与缓存：`huggingface_hub.snapshot_download`（含 `local_dir`、过滤、`dry_run`）
- 离线模式：`HF_HUB_OFFLINE=1` + `local_files_only=True`

参考：

- https://huggingface.co/docs/huggingface_hub/guides/download
- https://huggingface.co/docs/transformers/en/installation#offline-mode

## 6. 注意事项

- `openai/privacy-filter` 主仓是基础模型，不等于 INT4 成品；INT4 通常在 Quantizations 子模型里。
- 该模型用于隐私检测辅助，不等于合规保证；生产场景应保留人工复核。
