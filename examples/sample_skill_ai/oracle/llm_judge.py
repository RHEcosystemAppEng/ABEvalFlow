# #ai-generated-oracle
"""LLM-as-judge evaluator for URL shortener skill.

Assesses the quality of the implementation beyond what deterministic
tests can cover: code style, error messages, documentation.
"""

import importlib
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path("/workspace")


def judge() -> dict:
    """Run LLM-based quality assessment and return score + rationale."""
    sys.path.insert(0, str(WORKSPACE))
    if "shortener" in sys.modules:
        del sys.modules["shortener"]

    try:
        mod = importlib.import_module("shortener")
    except ImportError:
        return {"score": 0.0, "rationale": "shortener module not found"}

    checks = []
    score = 1.0

    if not mod.__doc__:
        checks.append("Missing module docstring")
        score -= 0.1

    for fn_name in ("shorten", "expand"):
        fn = getattr(mod, fn_name, None)
        if fn is None:
            checks.append(f"Missing function: {fn_name}")
            score -= 0.3
        elif not fn.__doc__:
            checks.append(f"Missing docstring for {fn_name}")
            score -= 0.05

    try:
        short = mod.shorten("https://example.com")
        if not short.isalnum() or len(short) != 6:
            checks.append("Short ID format is non-standard")
            score -= 0.1
    except Exception as exc:
        checks.append(f"shorten raised unexpected error: {exc}")
        score -= 0.2

    score = max(0.0, min(1.0, score))
    rationale = "; ".join(checks) if checks else "Implementation meets all quality criteria"

    return {"score": round(score, 2), "rationale": rationale}


if __name__ == "__main__":
    result = judge()
    print(json.dumps(result, indent=2))
