"""Build RubricHub-medical train/val parquet for the RGSD repro (2606.12507).

Source: sojuL/RubricHub_v1 (Li et al., 2026; Apache-2.0), per-domain file
`RuRL/rurbichub_v1_Medical.parquet`. Each source row carries a per-prompt rubric
(`reward_model.rubrics` = list of {criterion, points}). We:

  - extract the user question + the weighted rubric,
  - build a SkyRL row: prompt=[system,user(question)], env_class="rubric",
    reward_spec.ground_truth={question, rubrics}, extra_info,
  - split seeded into train.parquet + a 300-prompt held-out validation.parquet
    (mirroring the paper's held-out RubricHub validation subset).

Deterministic (SEED=42) so every node regenerates byte-identical data.
"""

import argparse
import json
import os
import random

from rgsd.prompts import SYSTEM_PROMPT, unconditioned_user

SEED = 42


def _extract_question(prompt_msgs) -> str:
    """Pull the user-facing question text from the source `prompt` messages."""
    if isinstance(prompt_msgs, list):
        users = [m.get("content", "") for m in prompt_msgs if m.get("role") == "user"]
        if users:
            return "\n\n".join(u.strip() for u in users if u and u.strip()).strip()
        # fall back to any content
        return "\n\n".join(str(m.get("content", "")).strip() for m in prompt_msgs).strip()
    return str(prompt_msgs).strip()


def _clean_rubrics(rm) -> list:
    """reward_model.rubrics -> [{criterion, points}] with positive points only."""
    out = []
    rubrics = (rm or {}).get("rubrics") or []
    for r in rubrics:
        crit = str(r.get("criterion", "")).strip()
        try:
            pts = int(r.get("points", 1))
        except Exception:
            pts = 1
        if crit and pts > 0:
            out.append({"criterion": crit, "points": pts})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--domain", default="Medical")
    ap.add_argument("--n-train", type=int, default=2000, help="cap on train rows (cost control); -1 = all")
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--min-rubrics", type=int, default=3)
    ap.add_argument("--max-prompt-chars", type=int, default=6000)
    ap.add_argument("--eval-md", default="data_EVAL.md")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from datasets import load_dataset

    fname = f"RuRL/rurbichub_v1_{args.domain}.parquet"
    print(f"[build] loading sojuL/RubricHub_v1 :: {fname}")
    ds = load_dataset("sojuL/RubricHub_v1", data_files=fname, split="train")
    print(f"[build] raw rows: {len(ds)}")

    records = []
    skipped = 0
    for row in ds:
        question = _extract_question(row.get("prompt"))
        rubrics = _clean_rubrics(row.get("reward_model"))
        if not question or len(question) > args.max_prompt_chars or len(rubrics) < args.min_rubrics:
            skipped += 1
            continue
        records.append(
            {
                "data_source": f"rubrichub_{args.domain.lower()}",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": unconditioned_user(question)},
                ],
                "env_class": "rubric",
                "reward_spec": {
                    "method": "reward_model",
                    "ground_truth": {"question": question, "rubrics": rubrics},
                },
                "extra_info": {
                    "domain": args.domain.lower(),
                    "n_rubrics": len(rubrics),
                    "total_points": sum(r["points"] for r in rubrics),
                },
            }
        )

    rng = random.Random(SEED)
    rng.shuffle(records)
    print(f"[build] kept {len(records)} rows (skipped {skipped})")

    n_val = min(args.n_val, len(records) // 5)
    val = records[:n_val]
    train_pool = records[n_val:]
    if args.n_train >= 0:
        train = train_pool[: args.n_train]
    else:
        train = train_pool

    import pandas as pd

    train_path = os.path.join(args.out, "train.parquet")
    val_path = os.path.join(args.out, "validation.parquet")
    pd.DataFrame(train).to_parquet(train_path)
    pd.DataFrame(val).to_parquet(val_path)
    print(f"[build] wrote {len(train)} train -> {train_path}")
    print(f"[build] wrote {len(val)} val   -> {val_path}")

    # ---- EVAL.md gate artifact: dataset stats + one rendered sample ----
    import statistics

    n_rub = [r["extra_info"]["n_rubrics"] for r in records]
    pts = [r["extra_info"]["total_points"] for r in records]
    sample = val[0] if val else records[0]
    with open(args.eval_md, "w") as f:
        f.write(f"# RubricHub-{args.domain} dataset\n\n")
        f.write(f"- source: sojuL/RubricHub_v1 :: {fname} (Apache-2.0)\n")
        f.write(f"- kept rows: {len(records)} (skipped {skipped})\n")
        f.write(f"- train: {len(train)}   val: {len(val)} (seed {SEED})\n")
        f.write(f"- rubrics/prompt: mean {statistics.mean(n_rub):.1f}, "
                f"min {min(n_rub)}, max {max(n_rub)}\n")
        f.write(f"- total points/prompt: mean {statistics.mean(pts):.1f}\n\n")
        f.write("## Sample validation row\n\n")
        f.write("Question:\n\n")
        f.write("> " + sample["reward_spec"]["ground_truth"]["question"][:800].replace("\n", "\n> ") + "\n\n")
        f.write("Rubric:\n\n")
        for r in sample["reward_spec"]["ground_truth"]["rubrics"][:8]:
            f.write(f"- (w{r['points']}) {r['criterion']}\n")
    print(f"[build] wrote {args.eval_md}")


if __name__ == "__main__":
    main()
