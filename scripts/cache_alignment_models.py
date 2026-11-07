"""Cache the ASR/alignment models used by the unchanged WhisperX wrapper."""

from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
from pathlib import Path
import time

from huggingface_hub import HfApi, snapshot_download
import requests


MODELS = (
    ("Systran/faster-whisper-small", "model.bin"),
    ("facebook/wav2vec2-lv-60-espeak-cv-ft", None),
)
MAX_ATTEMPTS = 8


def cache_model(item):
    repo_id, weight_name = item
    info = HfApi().model_info(repo_id, files_metadata=True)
    names = {entry.rfilename for entry in info.siblings}
    if weight_name is None:
        weight_name = "model.safetensors" if "model.safetensors" in names else "pytorch_model.bin"
    if weight_name not in names:
        raise RuntimeError(f"No expected model weight in {repo_id}: {weight_name}")
    allowed = sorted(name for name in names
                     if name.endswith((".json", ".txt")) or name == weight_name)
    expected_bytes = next((getattr(entry, "size", None) for entry in info.siblings
                           if entry.rfilename == weight_name), None)
    print(f"Caching {repo_id}@{info.sha}: {weight_name}", flush=True)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            snapshot = snapshot_download(repo_id, revision=info.sha,
                                         allow_patterns=allowed, max_workers=4)
            # Hub may return a cached snapshot directory after a network error
            # even when its weight was not downloaded. Verify actual files.
            missing = [name for name in allowed if not (Path(snapshot) / name).is_file()]
            if missing:
                raise OSError(f"Incomplete snapshot for {repo_id}: {missing}")
            actual_bytes = (Path(snapshot) / weight_name).stat().st_size
            if expected_bytes is not None and actual_bytes != expected_bytes:
                raise OSError(f"Incomplete weight for {repo_id}: {actual_bytes}/{expected_bytes}")
            break
        except (OSError, requests.exceptions.RequestException) as error:
            response = getattr(error, "response", None)
            if response is not None and response.status_code < 500 and response.status_code not in {408, 429}:
                raise
            print(f"{repo_id}: transfer attempt {attempt}/{MAX_ATTEMPTS} failed: {type(error).__name__}; resuming cache", flush=True)
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(min(attempt * 2, 10))
    print(f"Cached {repo_id}", flush=True)
    return {"repo_id": repo_id, "revision": info.sha, "snapshot": snapshot,
            "files": allowed, "weight_bytes": expected_bytes,
            "verified_weight_bytes": actual_bytes}


def main():
    with ThreadPoolExecutor(max_workers=2) as executor:
        models = list(executor.map(cache_model, MODELS))
    output = Path("outputs/improvement/model_cache_manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "models": models, "torchcodec": version("torchcodec"),
    }, indent=2) + "\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
