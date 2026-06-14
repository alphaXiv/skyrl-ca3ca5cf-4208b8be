"""Shared prompts for the RGSD reproduction (2606.12507).

Single source of truth so the dataset builder, the training env, the teacher
conditioning, and the LLM judge never drift.

Two response conditions:
  - UNCONDITIONED (student / base): the model sees only the prompt q.
  - CONDITIONED (teacher / base+rubric): the model additionally sees the per-prompt
    rubric as a structured criteria list plus a "transition" instruction telling it
    to satisfy the criteria *naturally* without explicitly referencing them
    (paper Appendix C.1). This is the privileged signal RGSD distills from, and the
    "+rubric in prompt" column of the paper's Table 1 conditioning-lift table.
"""

SYSTEM_PROMPT = (
    "You are a knowledgeable, careful assistant. Answer the user's question "
    "thoroughly and accurately."
)

# Appended to the user turn to build the rubric-conditioned (teacher) prompt.
RUBRIC_CONDITION_TEMPLATE = (
    "{question}\n\n"
    "---\n"
    "When writing your answer, make sure it satisfies the following criteria. "
    "Address them naturally as part of a coherent response; do NOT mention, list, "
    "or explicitly reference these criteria in your answer.\n"
    "{rubric_block}"
)


def format_rubric_block(rubrics) -> str:
    """rubrics: list of {'criterion': str, 'points': int}. Render as a weighted list."""
    lines = []
    for i, r in enumerate(rubrics, 1):
        crit = str(r.get("criterion", "")).strip()
        pts = r.get("points", 1)
        lines.append(f"{i}. (weight {pts}) {crit}")
    return "\n".join(lines)


def unconditioned_user(question: str) -> str:
    """Student / base-model user turn: prompt only."""
    return question


def conditioned_user(question: str, rubrics) -> str:
    """Teacher / base+rubric user turn: prompt + rubric criteria + transition."""
    return RUBRIC_CONDITION_TEMPLATE.format(
        question=question, rubric_block=format_rubric_block(rubrics)
    )


# ----------------------------------------------------------------------------- judge

JUDGE_SYSTEM = (
    "You are a strict, fair grader. You are given a user prompt, a candidate "
    "response, and a list of rubric criteria. For EACH criterion, decide whether the "
    "candidate response satisfies it. Judge only what the response actually contains. "
    "Be conservative: if a criterion is not clearly satisfied, mark it false."
)


def judge_user(question: str, response: str, rubrics) -> str:
    """One judge call grades ALL criteria for a response (cost-efficient).

    The grader returns a JSON object {"met": [bool, ...]} aligned to the rubric order.
    Score is computed caller-side as sum(points where met) / sum(positive points).
    """
    crit_lines = []
    for i, r in enumerate(rubrics):
        crit_lines.append(f'{i}: {str(r.get("criterion", "")).strip()}')
    crits = "\n".join(crit_lines)
    return (
        f"# User prompt\n{question}\n\n"
        f"# Candidate response\n{response}\n\n"
        f"# Rubric criteria (index: criterion)\n{crits}\n\n"
        "For each criterion index above, output whether the candidate response "
        "satisfies it. Respond with ONLY a JSON object of the form "
        '{"met": [true, false, ...]} where the array has exactly one boolean per '
        "criterion, in index order. No prose, no explanation."
    )
