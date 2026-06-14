"""Baseline eval + conditioning-lift check for the RGSD repro (2606.12507).

This is the project's root/control. NO training. It establishes:
  1. base-model rubric satisfaction on RubricHub-med-300 (the "+0" reference for
     the GRPO / RGSD children), and
  2. the rubric-conditioning gap (paper Table 1): base vs base+rubric-in-prompt.
     A large gap (paper: +44pp at Qwen-2.5-3B) is the premise RGSD relies on.

Generation: offline vLLM, greedy, on the same model the children train.
Grading: the shared OpenRouter judge (rgsd/judge.py) — identical to the GRPO reward.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from rgsd import judge
from rgsd.prompts import SYSTEM_PROMPT, conditioned_user, unconditioned_user


def _load_val(path):
    import pandas as pd

    df = pd.read_parquet(path)
    items = []
    for _, row in df.iterrows():
        gt = row["reward_spec"]["ground_truth"]
        items.append({"question": gt["question"], "rubrics": list(gt["rubrics"])})
    return items


def _chat(tokenizer, user_content):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _grade_all(items, responses, label):
    scores = [0.0] * len(items)

    def work(i):
        scores[i] = judge.grade(items[i]["question"], responses[i], items[i]["rubrics"])

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(work, range(len(items))))
    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"[eval] {label}: mean rubric-sat {mean:.4f} over {len(scores)} prompts")
    return scores, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--val", default="data/validation.parquet")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=-1, help="cap #prompts (debug); -1 = all")
    ap.add_argument("--out", default="EVAL.md")
    ap.add_argument("--dump", default="eval_rollouts.jsonl")
    args = ap.parse_args()

    items = _load_val(args.val)
    if args.limit >= 0:
        items = items[: args.limit]
    print(f"[eval] {len(items)} validation prompts")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem_util, max_model_len=8192, dtype="bfloat16")
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    uncond_prompts = [_chat(tok, unconditioned_user(it["question"])) for it in items]
    cond_prompts = [_chat(tok, conditioned_user(it["question"], it["rubrics"])) for it in items]

    print("[eval] generating UNCONDITIONED (base)...")
    uncond = [o.outputs[0].text for o in llm.generate(uncond_prompts, sp)]
    print("[eval] generating CONDITIONED (base + rubric)...")
    cond = [o.outputs[0].text for o in llm.generate(cond_prompts, sp)]

    u_scores, u_mean = _grade_all(items, uncond, "UNCONDITIONED")
    c_scores, c_mean = _grade_all(items, cond, "CONDITIONED  ")
    lift_pp = (c_mean - u_mean) * 100

    u_len = sum(len(t) for t in uncond) / max(1, len(uncond))
    c_len = sum(len(t) for t in cond) / max(1, len(cond))
    usage = judge.usage_summary()

    with open(args.dump, "w") as f:
        for i, it in enumerate(items):
            f.write(json.dumps({
                "question": it["question"][:500],
                "uncond_resp": uncond[i][:1500], "uncond_score": u_scores[i],
                "cond_resp": cond[i][:1500], "cond_score": c_scores[i],
            }, ensure_ascii=False) + "\n")

    with open(args.out, "w") as f:
        f.write("# RGSD baseline — Qwen-2.5-3B on RubricHub-med (no training)\n\n")
        f.write(f"Model: `{args.model}`  |  prompts: {len(items)}  |  judge: `{judge._model()}` (OpenRouter)\n\n")
        f.write("## Rubric satisfaction (the +0 reference for GRPO / RGSD children)\n\n")
        f.write("| condition | rubric-sat | mean resp chars |\n|---|---|---|\n")
        f.write(f"| base (unconditioned) | {u_mean:.4f} | {u_len:.0f} |\n")
        f.write(f"| base + rubric in prompt | {c_mean:.4f} | {c_len:.0f} |\n\n")
        f.write(f"**Conditioning lift: +{lift_pp:.1f}pp** "
                f"(paper Table 1 for Qwen-2.5-3B medical: +44.0pp). ")
        f.write("A large positive lift confirms the base model can satisfy rubrics when shown them — "
                "the latent capability RGSD distills.\n\n")
        f.write("## Judge cost (OpenRouter)\n\n")
        f.write(f"- calls: {usage['calls']}  (cache hits: {usage['cache_hits']}, errors: {usage['errors']})\n")
        f.write(f"- tokens: {usage['in_tok']} in / {usage['out_tok']} out\n")
        f.write(f"- **approx spend: ${usage['usd']:.3f}**\n")
    print(f"[eval] wrote {args.out}  (conditioning lift +{lift_pp:.1f}pp, judge ${usage['usd']:.3f})")


if __name__ == "__main__":
    main()
