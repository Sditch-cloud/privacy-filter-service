from typing import Optional, List
import os
import inspect
from pathlib import Path

from huggingface_hub import snapshot_download


def download_model(
  repo_id: str = "openai/privacy-filter",
  revision: Optional[str] = None,
  cache_dir: str = "./models-cache",
  local_dir: Optional[str] = None,
  allow_patterns: Optional[List[str]] = None,
  ignore_patterns: Optional[List[str]] = None,
  token: Optional[str] = None,
  mirror: Optional[str] = None,
  dry_run: bool = False,
 ) -> Optional[str]:
  """Download a Hugging Face model snapshot.

  Returns a single normalized `snapshot_path` string on success. When
  `dry_run=True` the returned string will be a comma-separated list of
  normalized paths. Returns `None` on failure. The function also prints
  the resulting `snapshot_path` for CLI visibility.
  """

  token = token or os.getenv("HF_TOKEN")
  # Convert empty lists to None so huggingface_hub treats them as unset
  allow_patterns = allow_patterns or None
  ignore_patterns = ignore_patterns or None

  endpoint_kwargs = {}
  if mirror:
    mirror = mirror.rstrip("/")
    os.environ.setdefault("HF_HUB_URL", mirror)
    os.environ.setdefault("HF_ENDPOINT", mirror)
    sig = inspect.signature(snapshot_download)
    if "endpoint" in sig.parameters:
      endpoint_kwargs["endpoint"] = mirror

  try:
    snapshot_path = snapshot_download(
      repo_id=repo_id,
      repo_type="model",
      revision=revision,
      cache_dir=cache_dir,
      local_dir=local_dir,
      allow_patterns=allow_patterns,
      ignore_patterns=ignore_patterns,
      token=token,
      dry_run=dry_run,
      **endpoint_kwargs,
    )

    if snapshot_path:
      # Normalize to forward slashes for consistent, platform-independent output.
      # `snapshot_download` returns a `str` path on success, but when
      # `dry_run=True` it may return a list of `DryRunFileInfo` objects.
      if isinstance(snapshot_path, (list, tuple)):
        try:
          posix_list = [Path(str(p)).as_posix() for p in snapshot_path]
        except Exception:
          posix_list = [str(p).replace("\\", "/") for p in snapshot_path]
        joined = ",".join(posix_list)
        print(f"snapshot_path=[{joined}]")
        return joined
      else:
        try:
          posix_path = Path(snapshot_path).as_posix()
        except Exception:
          posix_path = str(snapshot_path).replace("\\", "/")
        print(f"snapshot_path={posix_path}")
        return posix_path

    return None
  except Exception as e:
    print(f"Model download failed: {e}")
    return None


