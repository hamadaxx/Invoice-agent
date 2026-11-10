from agent.checkpoint import AutoApprover
from agent.loop import run_agent
from agent.model import LoopingModel
from agent.stop_rule import StopRule


def test_starts_clear():
    assert StopRule().exceeded() is None


def test_step_limit():
    s = StopRule(max_steps=3)
    for _ in range(3):
        s.add_step()
    assert s.exceeded() == "max_steps"


def test_token_limit():
    s = StopRule(max_tokens=1000)
    s.add_tokens(1000)
    assert s.exceeded() == "max_tokens"


def test_budget():
    s = StopRule(max_payout=10_000)
    assert s.can_pay(10_000)
    s.add_payout(6_000)
    assert not s.can_pay(5_000)


def test_looping_model_is_stopped(toolbox):
    """A model that never stops calling tools still ends, at exactly max_steps."""
    for limit in (5, 20, 77):
        r = run_agent(LoopingModel(), toolbox, AutoApprover(), StopRule(max_steps=limit), "go")
        assert r.stop_reason == "max_steps"
        assert r.steps == limit


def test_token_budget_stops_looping_model(toolbox):
    r = run_agent(LoopingModel(), toolbox, AutoApprover(), StopRule(max_steps=10_000, max_tokens=500), "go")
    assert r.stop_reason == "max_tokens"
