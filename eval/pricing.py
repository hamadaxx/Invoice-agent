"""Per-token USD prices for the models the eval runner can bill against.

Source: https://platform.openai.com/docs/pricing (redirects to
https://developers.openai.com/api/docs/pricing), fetched 2026-11-09. These are
OpenAI's standard (non-batch, non-cached) API rates, in USD per 1,000,000 tokens:

    model          input $/1M   output $/1M
    gpt-4o-mini        0.15         0.60
    gpt-4.1-mini       0.40         1.60

AGENT_MODEL defaults to "gpt-4o-mini" (agent/model.py), so that's the rate the
eval runner bills against unless AGENT_MODEL is overridden. Prices change; if
AGENT_MODEL is pointed at something not listed here, cost comes back as None
rather than a silently wrong number.
"""
import os

# USD per 1,000,000 tokens, standard (non-cached) rate.
PRICES_PER_MILLION = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Return the dollar cost of a call, or None if `model` has no listed price."""
    price = PRICES_PER_MILLION.get(model)
    if price is None:
        return None
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


def current_model() -> str:
    return os.environ.get("AGENT_MODEL", "gpt-4o-mini")
