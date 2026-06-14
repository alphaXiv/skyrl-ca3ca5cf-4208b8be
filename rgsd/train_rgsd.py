"""RGSD — Rubric-Guided Self-Distillation (2606.12507), lean on-policy loop.

Teacher = frozen base weights (LoRA adapter DISABLED) conditioned on prompt+rubric.
Student = base + trainable LoRA adapter, conditioned on prompt only.

Per step, per prompt q with rubric R:
  1. student samples an on-policy rollout y ~ pi_S(.|q)   (adapter ON, prompt only)
  2. student forward (adapter ON, grad)  on [q, y]        -> log pi_S(.|q, y_<t)
  3. teacher forward (adapter OFF, no grad) on [q,R, y]   -> log pi_T(.|q,R, y_<t)
  4. loss = mean_t clipped-JSD_beta( pi_S(.|q,y_<t) || pi_T(.|q,R,y_<t) ),  beta=0.5
  5. backprop into LoRA only.   NO judge calls during training (verifier-free).

Eval (greedy, adapter ON, prompt only) is scored by the SAME OpenRouter rubric judge
as the GRPO arm + baseline -> apples-to-apples comparison.
"""

import argparse
import json
import os
import random

import torch

from rgsd import judge
from rgsd.prompts import SYSTEM_PROMPT, conditioned_user, unconditioned_user

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _load_rows(path):
    import pandas as pd

    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        gt = r["reward_spec"]["ground_truth"]
        rows.append({"question": str(gt["question"]), "rubrics": [dict(x) for x in gt["rubrics"]]})
    return rows


def _ids(tok, user_content):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]


def gen_jsd(student_logits, teacher_logits, beta=0.5, clip=None):
    """Generalized clipped JSD over response tokens.

    student_logits/teacher_logits: [N, V] raw logits aligned to the N response tokens.
    Student carries grad; teacher is detached. Computed in fp32.
    """
    sp = torch.log_softmax(student_logits.float(), dim=-1)
    tp = torch.log_softmax(teacher_logits.float(), dim=-1).detach()
    p = sp.exp()
    q = tp.exp()
    m = (beta * p + (1.0 - beta) * q).clamp_min(1e-12)
    logm = m.log()
    kl_pm = (p * (sp - logm)).sum(-1)
    kl_qm = (q * (tp - logm)).sum(-1)
    jsd = beta * kl_pm + (1.0 - beta) * kl_qm  # [N]
    if clip is not None and clip > 0:
        jsd = jsd.clamp(max=clip)
    return jsd.mean()


@torch.no_grad()
def evaluate(model, tok, rows, max_new_tokens, n_prompts, device):
    model.eval()
    sub = rows[:n_prompts]
    scores = []
    bs = 16
    gen_texts = []
    for i in range(0, len(sub), bs):
        batch = sub[i : i + bs]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": unconditioned_user(r["question"])}],
            tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        for j in range(len(batch)):
            resp = out[j, enc.input_ids.shape[1]:]
            gen_texts.append(tok.decode(resp, skip_special_tokens=True))
    from concurrent.futures import ThreadPoolExecutor

    scores = [0.0] * len(sub)
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(lambda k: scores.__setitem__(k, judge.grade(sub[k]["question"], gen_texts[k], sub[k]["rubrics"])), range(len(sub))))
    model.train()
    return sum(scores) / len(scores) if scores else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--train", default="data/train.parquet")
    ap.add_argument("--val", default="data/validation.parquet")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-prompts", type=int, default=8, help="prompts per optimizer step (grad accum)")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--jsd-clip", type=float, default=0.0, help="per-token JSD clamp; 0=off")
    ap.add_argument("--gen-temp", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--eval-interval", type=int, default=8, help="optimizer steps between evals")
    ap.add_argument("--eval-prompts", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--out", default="EVAL.md")
    ap.add_argument("--save-lora", default="rgsd_lora")
    args = ap.parse_args()

    device = "cuda"
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # for batched generation

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                                             target_modules=LORA_TARGETS, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.config.use_cache = True  # needed for generate; forward passes don't rely on cache

    train_rows = _load_rows(args.train)
    val_rows = _load_rows(args.val)
    eos = tok.eos_token_id
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    history = []
    step = 0
    base_eval = evaluate(model, tok, val_rows, args.max_new_tokens, args.eval_prompts, device)
    history.append((0, base_eval))
    print(f"[rgsd] step 0 (pre-train) eval rubric-sat {base_eval:.4f}")

    for epoch in range(args.epochs):
        random.Random(1234 + epoch).shuffle(train_rows)
        for i in range(0, len(train_rows), args.batch_prompts):
            batch = train_rows[i : i + args.batch_prompts]
            opt.zero_grad()
            tok.padding_side = "left"
            # --- 1. on-policy student rollouts (adapter ON) ---
            model.eval()
            with torch.no_grad():
                sprompts = [tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": unconditioned_user(r["question"])}],
                    tokenize=False, add_generation_prompt=True) for r in batch]
                enc = tok(sprompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                gen = model.generate(**enc, do_sample=True, temperature=args.gen_temp, top_p=1.0,
                                     max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
            responses = []
            for j in range(len(batch)):
                full = gen[j, enc.input_ids.shape[1]:]
                eos_pos = (full == eos).nonzero()
                if eos_pos.numel() > 0:
                    full = full[: int(eos_pos[0, 0]) + 1]  # keep through the first real EOS
                responses.append(full)
            model.train()

            # --- 2-4. per-example student/teacher forwards + JSD (grad accum) ---
            step_loss = 0.0
            n_ok = 0
            for j, r in enumerate(batch):
                resp = responses[j]
                if resp.numel() == 0:
                    continue
                sp_ids = _ids(tok, unconditioned_user(r["question"])).to(device)
                tp_ids = _ids(tok, conditioned_user(r["question"], r["rubrics"])).to(device)
                s_seq = torch.cat([sp_ids, resp]).unsqueeze(0)
                t_seq = torch.cat([tp_ids, resp]).unsqueeze(0)
                Lr = resp.numel()
                # student (grad, adapter ON)
                s_logits = model(s_seq, use_cache=False).logits[0]  # [T,V]
                s_resp = s_logits[sp_ids.numel() - 1 : sp_ids.numel() + Lr - 1]  # [Lr,V]
                # teacher (no grad, adapter OFF)
                with torch.no_grad():
                    with model.disable_adapter():
                        t_logits = model(t_seq, use_cache=False).logits[0]
                t_resp = t_logits[tp_ids.numel() - 1 : tp_ids.numel() + Lr - 1]
                loss = gen_jsd(s_resp, t_resp, beta=args.beta, clip=args.jsd_clip) / len(batch)
                loss.backward()
                step_loss += loss.item() * len(batch)
                n_ok += 1
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            opt.step()
            step += 1
            print(f"[rgsd] epoch {epoch} step {step} jsd {step_loss / max(1, n_ok):.4f} ({n_ok}/{len(batch)} rollouts)")

            if step % args.eval_interval == 0:
                ev = evaluate(model, tok, val_rows, args.max_new_tokens, args.eval_prompts, device)
                history.append((step, ev))
                print(f"[rgsd] step {step} eval rubric-sat {ev:.4f}")
            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps > 0 and step >= args.max_steps:
            break

    final_eval = evaluate(model, tok, val_rows, args.max_new_tokens, args.eval_prompts, device)
    history.append((step, final_eval))
    model.save_pretrained(args.save_lora)
    usage = judge.usage_summary()

    with open(args.out, "w") as f:
        f.write("# RGSD arm — Qwen-2.5-3B rubric self-distillation (RubricHub-med)\n\n")
        f.write("Baseline (no-train) reference: base 0.2355, base+rubric 0.8236.\n\n")
        f.write("## eval rubric-sat over training (step: score)\n\n")
        for s, v in history:
            f.write(f"- step {s}: {v:.4f}\n")
        f.write(f"\n**peak {max(v for _, v in history):.4f}, final {final_eval:.4f}** "
                f"(vs base 0.2355). Verifier-free: 0 judge calls in training.\n\n")
        f.write("## Judge cost (eval only; OpenRouter)\n\n")
        f.write(f"- calls {usage['calls']} (cache {usage['cache_hits']}, err {usage['errors']}), "
                f"tokens {usage['in_tok']}/{usage['out_tok']}, **${usage['usd']:.3f}**\n")
    print(f"[rgsd] done. peak {max(v for _, v in history):.4f} final {final_eval:.4f}  judge ${usage['usd']:.3f}")


if __name__ == "__main__":
    main()
