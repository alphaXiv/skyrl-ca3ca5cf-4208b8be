#!/usr/bin/env bash
# Stage 1: build the HotpotQA retrieval-subagent dataset and write EVAL.md.
# No training in this stage — the artifact (EVAL.md + data/*.parquet stats) is the gate.
set -euxo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv .venv-data --python 3.12 --allow-existing
uv pip install --python .venv-data/bin/python -q datasets pandas pyarrow numpy

.venv-data/bin/python chroma/build_dataset.py --out data --eval-md EVAL.md

echo "=== EVAL.md ==="
cat EVAL.md
ls -lh data/
