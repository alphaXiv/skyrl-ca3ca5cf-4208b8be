#!/usr/bin/env bash
# RGSD repro (2606.12507) — RGSD ARM (paper's method, verifier-free).
# Rubric-guided self-distillation on Qwen-2.5-3B via a lean on-policy loop:
#   student (LoRA on, prompt only) rolls out y; teacher (base weights, adapter
#   disabled, rubric-conditioned prompt) scores y; per-token clipped JSD(beta=0.5).
# NO judge calls during training. Eval uses the SAME OpenRouter rubric judge as the
# GRPO arm / baseline -> apples-to-apples. Matched data/schedule with GRPO (512 train,
# 2 epochs). Gate: eval rubric-sat rises above base 0.2355, comparable to the GRPO arm.
set -euxo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

MODEL="Qwen/Qwen2.5-3B-Instruct"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export RGSD_JUDGE_MODEL="openai/gpt-4o-mini"
export RGSD_JUDGE_CACHE="$ROOT/judge_cache.jsonl"
export RGSD_JUDGE_MAX_USD="3.0"   # eval-only judge bill (training is verifier-free)

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "FATAL: OPENROUTER_API_KEY not set in the run environment." >&2
  exit 3
fi

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
apt-get install -y -qq libnuma1 2>/dev/null || true

uv sync --extra fsdp
uv pip install --python .venv/bin/python -q openai datasets pandas pyarrow peft

# matched data with the GRPO arm (deterministic, gitignored)
.venv/bin/python rgsd/build_dataset.py --out data --domain Medical \
  --n-train 512 --n-val 300 --eval-md data_EVAL.md

# RGSD training + in-loop eval; writes EVAL.md
.venv/bin/python rgsd/train_rgsd.py \
  --model "$MODEL" --train "$ROOT/data/train.parquet" --val "$ROOT/data/validation.parquet" \
  --epochs 2 --batch-prompts 16 --lora-rank 32 --lora-alpha 64 --lr 1e-4 \
  --beta 0.5 --gen-temp 1.0 --max-new-tokens 768 \
  --eval-interval 16 --eval-prompts 300 \
  --out EVAL.md --save-lora "$HOME/rgsd_lora"

mkdir -p .openresearch/artifacts
cp -f EVAL.md data_EVAL.md .openresearch/artifacts/ 2>/dev/null || true
cat EVAL.md
