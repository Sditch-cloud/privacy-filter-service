import argparse
import os
from typing import Optional
import inspect

from huggingface_hub import snapshot_download


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Download and cache a Hugging Face model snapshot for local/offline inference."
	)
	parser.add_argument(
		"--repo-id",
		default="openai/privacy-filter",
		help="Model repository id on Hugging Face Hub.",
	)
	parser.add_argument(
		"--revision",
		default=None,
		help="Optional branch/tag/commit revision to pin.",
	)
	parser.add_argument(
		"--cache-dir",
		default="./models-cache",
		help="Cache directory used by huggingface_hub.",
	)
	parser.add_argument(
		"--local-dir",
		default=None,
		help="If set, materialize files in this local directory (git-like layout).",
	)
	parser.add_argument(
		"--allow-pattern",
		action="append",
		default=[],
		help="Glob pattern to include. Repeat this flag to include multiple patterns.",
	)
	parser.add_argument(
		"--ignore-pattern",
		action="append",
		default=[],
		help="Glob pattern to exclude. Repeat this flag to exclude multiple patterns.",
	)
	parser.add_argument(
		"--token",
		default=None,
		help="HF token string. If omitted, uses HF_TOKEN env var when available.",
	)
	parser.add_argument(
		"--mirror",
		default=None,
		help="Optional mirror/endpoint URL (e.g. https://hf-mirror.com/) to override HF endpoint.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Only show which files would be downloaded.",
	)
	return parser


def _normalize_patterns(values: list[str]) -> Optional[list[str]]:
	if not values:
		return None
	return values


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	token = args.token or os.getenv("HF_TOKEN")
	allow_patterns = _normalize_patterns(args.allow_pattern)
	ignore_patterns = _normalize_patterns(args.ignore_pattern)
	# If a mirror/endpoint is provided, try to pass it to snapshot_download
	# (newer huggingface_hub versions accept an `endpoint` kwarg). Also
	# export common env vars to help older versions pick up the mirror.
	endpoint_kwargs = {}
	if args.mirror:
		mirror = args.mirror.rstrip('/')
		# set environment variables that some HF tooling respects
		os.environ.setdefault("HF_HUB_URL", mirror)
		os.environ.setdefault("HF_ENDPOINT", mirror)
		# If snapshot_download accepts `endpoint`, pass it directly.
		sig = inspect.signature(snapshot_download)
		if "endpoint" in sig.parameters:
			endpoint_kwargs["endpoint"] = mirror

	snapshot_path = snapshot_download(
		repo_id=args.repo_id,
		repo_type="model",
		revision=args.revision,
		cache_dir=args.cache_dir,
		local_dir=args.local_dir,
		allow_patterns=allow_patterns,
		ignore_patterns=ignore_patterns,
		token=token,
		dry_run=args.dry_run,
		**endpoint_kwargs,
	)

	print(f"repo_id={args.repo_id}")
	if args.revision:
		print(f"revision={args.revision}")
	print(f"cache_dir={os.path.abspath(args.cache_dir)}")
	if args.local_dir:
		print(f"local_dir={os.path.abspath(args.local_dir)}")
	print(f"snapshot_path={snapshot_path}")


if __name__ == "__main__":
	main()
