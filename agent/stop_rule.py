"""Stop rule. Written first, tested before the agent existed.

One counter, `steps`, goes up by one on every model call and every tool call.
The loop checks `exceeded()` before each model call and before each tool call.
Because `steps` only grows and `max_steps` is finite, the loop always ends.
That is the proof that the agent cannot run forever.

Two more limits stop it earlier:
- max_tokens, so a run cannot burn money
- max_payout, the batch budget. A payment that would cross it is refused and the run stops.
"""
from dataclasses import dataclass


@dataclass
class StopRule:
    max_steps: int = 60
    max_tokens: int = 200_000
    max_payout: float = 100_000.0

    steps: int = 0
    tokens: int = 0
    payout: float = 0.0
    budget_hit: bool = False

    def add_step(self) -> None:
        self.steps += 1

    def add_tokens(self, n: int) -> None:
        self.tokens += n

    def can_pay(self, amount: float) -> bool:
        return self.payout + amount <= self.max_payout

    def add_payout(self, amount: float) -> None:
        self.payout += amount

    def exceeded(self) -> str | None:
        """Return why the run must stop, or None if it can go on."""
        if self.budget_hit:
            return "max_payout"
        if self.steps >= self.max_steps:
            return "max_steps"
        if self.tokens >= self.max_tokens:
            return "max_tokens"
        return None
