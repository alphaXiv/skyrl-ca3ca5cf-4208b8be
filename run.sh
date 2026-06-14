#!/usr/bin/env bash
# RGSD repro (2606.12507) — GRPO ARM (paper's judge-based comparison method).
# Single-turn rubric env on Qwen-2.5-3B: model emits one response, reward = rubric
# satisfaction scored by gpt-4o-mini (OpenRouter). GRPO group-normalizes over G=8.
# Gate: eval/all/avg_score (mean rubric-sat) rises above the step-0 / baseline anchor
# (~0.236), no collapse. This is the GRPO side of the GRPO-vs-RGSD comparison.
set -euxo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

MODEL="Qwen/Qwen2.5-3B-Instruct"
RUN_NAME="grpo-qwen2.5-3b-rubric-med"
NUM_GPUS=$(nvidia-smi -L | wc -l)
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export RGSD_JUDGE_MODEL="openai/gpt-4o-mini"
export RGSD_JUDGE_CACHE="$ROOT/judge_cache.jsonl"
export RGSD_JUDGE_USAGE_OUT="$ROOT/judge_usage.json"
export RGSD_JUDGE_MAX_USD="10.0"   # training judge cap (won't be hit at this scale)

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "FATAL: OPENROUTER_API_KEY not set in the run environment." >&2
  exit 3
fi

# ---------- environment setup (bare-pod fix ladder) ----------
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
apt-get install -y -qq libnuma1 2>/dev/null || true

uv sync --extra fsdp
uv pip install --python .venv/bin/python -q openai datasets pandas pyarrow

.venv/bin/ray stop 2>/dev/null || true
.venv/bin/ray start --head --num-gpus="$NUM_GPUS"

# ---------- dataset (deterministic; gitignored) ----------
.venv/bin/python rgsd/build_dataset.py --out data --domain Medical \
  --n-train 512 --n-val 300 --eval-md data_EVAL.md

# ---------- GRPO training ----------
set +e
.venv/bin/python -m skyrl.train.entrypoints.main_base \
  data.train_data="['$ROOT/data/train.parquet']" \
  data.val_data="['$ROOT/data/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="regular" \
  trainer.algorithm.use_kl_loss=false \
  trainer.policy.model.path="$MODEL" \
  trainer.policy.model.lora.rank=32 \
  trainer.policy.model.lora.alpha=64 \
  trainer.policy.optimizer_config.lr=1.0e-5 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=2 \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  generator.inference_engine.num_engines="$NUM_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.45 \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.distributed_executor_backend=mp \
  trainer.epochs=2 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=4096 \
  generator.sampling_params.max_generate_length=1024 \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.step_wise_trajectories=true \
  generator.n_samples_per_prompt=8 \
  generator.max_turns=1 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.eval_n_samples_per_prompt=1 \
  generator.eval_sampling_params.temperature=0.0 \
  generator.eval_sampling_params.max_generate_length=1024 \
  environment.env_class="rubric" \
  environment.skyrl_gym.max_env_workers=16 \
  environment.skyrl_gym.rubric.judge_model="$RGSD_JUDGE_MODEL" \
  trainer.logger="wandb" \
  trainer.project_name="rgsd-skyrl" \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_interval=100 \
  trainer.hf_save_interval=100 \
  trainer.max_ckpts_to_keep=1 \
  trainer.resume_mode=none \
  trainer.ckpt_path="$HOME/ckpts/$RUN_NAME" \
  trainer.export_path="$HOME/ckpts/$RUN_NAME/exports" \
  trainer.eval_batch_size=300 \
  trainer.eval_before_train=true \
  trainer.eval_interval=8 \
  2>&1 | tee train.log
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

# ---------- EVAL.md ----------
{
  echo "# GRPO arm — $RUN_NAME (exit $TRAIN_EXIT)"
  echo
  echo "Baseline (no-train) reference: base rubric-sat 0.2355, base+rubric 0.8236."
  echo
  echo "## eval/all/avg_score (mean rubric-sat) over training"
  echo '```'
  grep -oE "'eval/all/avg_score': [0-9.]+" train.log | tail -10 || echo "none"
  echo '```'
  echo "## eval/env/rubric_sat + resp_chars"
  echo '```'
  grep -oE "'eval/env/(rubric_sat|resp_chars)': [0-9.]+" train.log | tail -10 || echo "none"
  echo '```'
  echo "## train reward"
  echo '```'
  grep -oE "avg_raw_reward[^,]*" train.log | tail -8 || true
  echo '```'
  echo "## Judge cost (OpenRouter, main process)"
  echo '```'
  cat judge_usage.json 2>/dev/null || echo "no usage file"
  echo '```'
  echo "## NaN check"
  if grep -qE "(loss|reward|grad_norm)[^a-zA-Z]*(nan|inf)" train.log; then
    echo "WARNING: possible nan/inf"
  else
    echo "no nan/inf in loss/reward/grad_norm"
  fi
} > EVAL.md

mkdir -p .openresearch/artifacts
cp -f EVAL.md train.log data_EVAL.md judge_usage.json .openresearch/artifacts/ 2>/dev/null || true
cat EVAL.md
exit "$TRAIN_EXIT"
