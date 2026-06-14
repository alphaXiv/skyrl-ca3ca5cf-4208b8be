#!/usr/bin/env bash
# RGSD repro (2606.12507) — BASELINE (root/control): base Qwen-2.5-3B on RubricHub-med-300.
# No training. Establishes the +0 reference for the GRPO / RGSD children AND the
# rubric-conditioning lift (paper Table 1, Qwen-2.5-3B medical: +44.0pp).
# Gate: judge runs cheaply (a few cents) and a clearly positive conditioning lift.
set -euxo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

MODEL="Qwen/Qwen2.5-3B-Instruct"
NUM_GPUS=$(nvidia-smi -L | wc -l)
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export RGSD_JUDGE_MODEL="openai/gpt-4o-mini"
export RGSD_JUDGE_CACHE="$ROOT/judge_cache.jsonl"
export RGSD_JUDGE_MAX_USD="2.0"   # hard cap on the baseline judge bill

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "FATAL: OPENROUTER_API_KEY not set in the run environment." >&2
  echo "Add it in the OpenResearch project/org env vars, then re-run." >&2
  exit 3
fi

# ---------- environment setup (bare-pod fix ladder, same as Chroma) ----------
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
apt-get install -y -qq libnuma1 2>/dev/null || true

uv sync --extra fsdp
uv pip install --python .venv/bin/python -q openai datasets pandas pyarrow

# ---------- dataset (val only needed here; deterministic, gitignored) ----------
.venv/bin/python rgsd/build_dataset.py --out data --domain Medical \
  --n-train 64 --n-val 300 --eval-md data_EVAL.md

# ---------- baseline eval + conditioning lift ----------
.venv/bin/python rgsd/eval_base.py --model "$MODEL" --val "$ROOT/data/validation.parquet" \
  --out EVAL.md --dump eval_rollouts.jsonl

# ---------- surface artifacts to OpenResearch ----------
mkdir -p .openresearch/artifacts
cp -f EVAL.md .openresearch/artifacts/EVAL.md || true
cp -f data_EVAL.md .openresearch/artifacts/data_EVAL.md || true
cp -f eval_rollouts.jsonl .openresearch/artifacts/eval_rollouts.jsonl || true
cat EVAL.md
