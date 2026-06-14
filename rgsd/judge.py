"""OpenRouter LLM judge for rubric satisfaction (RGSD repro, 2606.12507).

Used by BOTH the baseline eval and the GRPO training env so the reward signal and
the eval metric are computed identically.

- One judge call grades ALL criteria of a response (cheap: 1 call/response, not 1/criterion).
- Score = sum(points where criterion met) / sum(positive points)  -> rubric satisfaction in [0,1].
- Disk cache keyed by (model, prompt, response, rubrics) avoids re-grading identical inputs.
- Token usage is accumulated and dumped so the run reports the exact OpenRouter bill.
- Thread-safe: the SkyRL env grades up to `max_env_workers` episodes concurrently.

Config via env vars:
  OPENROUTER_API_KEY   (required)
  RGSD_JUDGE_MODEL     (default "openai/gpt-4o-mini")
  RGSD_JUDGE_CACHE     (default "judge_cache.jsonl" in CWD; "" disables)
  RGSD_JUDGE_MAX_USD   (default 5.0; hard stop to protect the bill)
"""

import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from rgsd.prompts import JUDGE_SYSTEM, judge_user

# gpt-4o-mini pricing (USD per token); used only for an approximate running cost.
_PRICE_IN = 0.15 / 1_000_000
_PRICE_OUT = 0.60 / 1_000_000

_LOCK = threading.Lock()
_CLIENT = None
_CACHE: Dict[str, float] = {}
_CACHE_LOADED = False

# accumulated accounting (process-wide)
_USAGE = {"calls": 0, "cache_hits": 0, "in_tok": 0, "out_tok": 0, "usd": 0.0, "errors": 0}


def _model() -> str:
    return os.environ.get("RGSD_JUDGE_MODEL", "openai/gpt-4o-mini")


def _cache_path() -> str:
    return os.environ.get("RGSD_JUDGE_CACHE", "judge_cache.jsonl")


def _max_usd() -> float:
    return float(os.environ.get("RGSD_JUDGE_MAX_USD", "5.0"))


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        _CLIENT = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    return _CLIENT


def _load_cache() -> None:
    global _CACHE_LOADED
    if _CACHE_LOADED:
        return
    path = _cache_path()
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    _CACHE[d["k"]] = d["v"]
        except Exception:
            pass
    _CACHE_LOADED = True


def _cache_put(key: str, val: float) -> None:
    _CACHE[key] = val
    path = _cache_path()
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps({"k": key, "v": val}) + "\n")
        except Exception:
            pass


def _key(question: str, response: str, rubrics: List[Dict[str, Any]]) -> str:
    crit = "|".join(f'{r.get("criterion","")}:{r.get("points",1)}' for r in rubrics)
    raw = f"{_model()}\x00{question}\x00{response}\x00{crit}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _parse_met(text: str, n: int) -> Optional[List[bool]]:
    if not text:
        return None
    s = text.strip()
    # tolerate ```json fences / surrounding prose
    if "```" in s:
        s = s.split("```")[1] if s.count("```") >= 2 else s
        s = s.replace("json", "", 1).strip() if s.lstrip().startswith("json") else s
    try:
        obj = json.loads(s)
    except Exception:
        # last resort: find the first {...} blob
        i, j = s.find("{"), s.rfind("}")
        if i == -1 or j == -1:
            return None
        try:
            obj = json.loads(s[i : j + 1])
        except Exception:
            return None
    met = obj.get("met") if isinstance(obj, dict) else None
    if not isinstance(met, list):
        return None
    out = [bool(x) for x in met][:n]
    while len(out) < n:
        out.append(False)
    return out


def _score_from_met(met: List[bool], rubrics: List[Dict[str, Any]]) -> float:
    total = sum(max(0, int(r.get("points", 1))) for r in rubrics)
    if total <= 0:
        return 0.0
    got = sum(int(r.get("points", 1)) for r, m in zip(rubrics, met) if m and int(r.get("points", 1)) > 0)
    return max(0.0, min(1.0, got / total))


def usage_summary() -> Dict[str, Any]:
    with _LOCK:
        return dict(_USAGE)


def grade(question: str, response: str, rubrics: List[Dict[str, Any]], max_retries: int = 4) -> float:
    """Return rubric-satisfaction score in [0,1]. Robust to API/parse failures.

    On hard failure (no key, budget exceeded, repeated errors) returns 0.0 so a
    single bad grade never crashes a training step; failures are counted in usage.
    """
    if not rubrics:
        return 0.0
    _load_cache()
    k = _key(question, response, rubrics)
    with _LOCK:
        if k in _CACHE:
            _USAGE["cache_hits"] += 1
            return _CACHE[k]
        if _USAGE["usd"] >= _max_usd():
            _USAGE["errors"] += 1
            return 0.0

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": judge_user(question, response, rubrics)},
    ]
    last_err = None
    for attempt in range(max_retries):
        try:
            client = _get_client()
            resp = client.chat.completions.create(
                model=_model(),
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content
            usage = getattr(resp, "usage", None)
            it = getattr(usage, "prompt_tokens", 0) or 0
            ot = getattr(usage, "completion_tokens", 0) or 0
            met = _parse_met(text, len(rubrics))
            with _LOCK:
                _USAGE["calls"] += 1
                _USAGE["in_tok"] += it
                _USAGE["out_tok"] += ot
                _USAGE["usd"] += it * _PRICE_IN + ot * _PRICE_OUT
            if met is None:
                last_err = f"unparseable: {text!r:.200}"
                continue
            score = _score_from_met(met, rubrics)
            _cache_put(k, score)
            return score
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)
            time.sleep(min(2.0 * (2**attempt), 20.0))
    with _LOCK:
        _USAGE["errors"] += 1
    return 0.0
