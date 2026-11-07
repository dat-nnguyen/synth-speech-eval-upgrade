#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config_name="${1:-input_sample_scores}"

if [[ ! -f "$project_dir/configs/$config_name.yaml" ]]; then
    echo "Missing config: configs/$config_name.yaml" >&2
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "Start Docker Desktop, then rerun this script." >&2
    exit 1
fi

if ! docker image inspect synth-speech-eval-cpu:local >/dev/null 2>&1; then
    docker build --platform linux/amd64 -f "$project_dir/Dockerfile.cpu" \
        -t synth-speech-eval-cpu:local "$project_dir"
fi

# Check only the dimensions selected by the merged config before downloading models.
docker run --rm --init --platform linux/amd64 \
    -v "$project_dir:/work" -w /work synth-speech-eval-cpu:local \
    python -c '
import sys
from pathlib import Path
import yaml
from src.util.config_util import deep_merge
base = yaml.safe_load(Path("configs/default.yaml").read_text())
config = deep_merge(base, yaml.safe_load(Path(f"configs/{sys.argv[1]}.yaml").read_text()))
missing = [f"src/models/weights/{dim}.pth" for dim in config["sq_ast_dims"]
           if not Path(f"src/models/weights/{dim}.pth").is_file()]
if missing:
    print("Missing SQ-AST weights:\n" + "\n".join(missing), file=sys.stderr)
    print("Download weights from https://github.com/WafaaWardah/SQ-AST", file=sys.stderr)
    sys.exit(1)
' "$config_name"

exec docker run --rm --init --platform linux/amd64 \
    -e HF_HUB_DISABLE_XET=1 \
    -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
    -e OMP_NUM_THREADS="${SQ_AST_CPU_THREADS:-4}" \
    -e MKL_NUM_THREADS="${SQ_AST_CPU_THREADS:-4}" \
    -e OPENBLAS_NUM_THREADS="${SQ_AST_CPU_THREADS:-4}" \
    -v "$project_dir:/work" \
    -v synth-speech-eval-cache:/root/.cache \
    -w /work synth-speech-eval-cpu:local \
    python -m src.eval "$config_name"
