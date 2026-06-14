"""Single-turn rubric-graded environment for the RGSD repro (2606.12507).

The judge-based GRPO arm: the model emits ONE response to an open-ended prompt; the
reward is its rubric satisfaction as scored by an LLM judge (OpenRouter gpt-4o-mini),
identical to the baseline eval metric. GRPO then group-normalizes these scalar
rewards across the n_samples_per_prompt rollouts.

Per-prompt question + rubric arrive via the `extras` (env_extras) channel:
    extras["reward_spec"]["ground_truth"] = {"question": str, "rubrics": [{criterion, points}]}
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

# Make the repo-root `rgsd` package (judge.py) importable from inside the gym worker.
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class RubricEnvConfig:
    judge_model: str = "openai/gpt-4o-mini"


class RubricEnv(BaseTextEnv):
    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()
        assert "reward_spec" in extras and "ground_truth" in extras["reward_spec"], "reward_spec.ground_truth required"
        gt = extras["reward_spec"]["ground_truth"]
        self.question = str(gt["question"])
        self.rubrics = [dict(r) for r in gt["rubrics"]]
        self._score = 0.0
        self._chars = 0.0

    def step(self, action: str) -> BaseTextEnvStepOutput:
        from rgsd.judge import grade  # imported lazily so the env module loads even pre-install

        self._score = grade(self.question, action, self.rubrics)
        self._chars = float(len(action))
        return BaseTextEnvStepOutput(
            observations=[],
            reward=self._score,
            done=True,  # single-turn: one response -> graded -> done
            metadata=self.get_metrics(),
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {"rubric_sat": float(self._score), "resp_chars": float(self._chars)}
