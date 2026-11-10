"""The agent loop: plan, call a tool, read the result, decide, stop.

Every model call and every tool call adds one step to the stop rule.
The stop rule is checked before each of them, so the loop always ends.
"""
import time
from dataclasses import dataclass, field

from agent.checkpoint import Approver
from agent.prompt import SYSTEM_PROMPT
from agent.stop_rule import StopRule
from agent.tools import Toolbox, ToolError


@dataclass
class RunResult:
    stop_reason: str  # "done", "max_steps", "max_tokens", "max_payout"
    final_text: str
    trajectory: list = field(default_factory=list)  # one entry per tool call
    steps: int = 0
    tokens: int = 0
    payout: float = 0.0
    seconds: float = 0.0
    unresolved: list = field(default_factory=list)  # invoice_ids a batch never got to decide


def run_agent(model, toolbox: Toolbox, approver: Approver, stop: StopRule, task: str,
              system: str = SYSTEM_PROMPT, verbose: bool = False, only_invoice: str | None = None) -> RunResult:
    """Runs the loop for one task/conversation. `only_invoice`, when set, restricts every
    invoice-keyed tool call (read_invoice, check_payment_history, schedule_payment,
    flag_for_review) to that one invoice_id -- see `handle_call`. Used by `run_batch` to give
    each invoice its own isolated context; a bare `run_agent` call still works exactly as
    before when `only_invoice` is omitted.
    """
    start = time.perf_counter()
    messages = [{"role": "user", "content": task}]
    trajectory = []
    final_text = ""
    reason = None

    def log(event: dict):
        trajectory.append(event)
        if verbose:
            print(f"[{event['step']:>2}] {event['tool']}({event['input']}) -> {event['output'][:160]}")

    while True:
        reason = stop.exceeded()
        if reason:
            break

        # ---- model call ----
        stop.add_step()
        turn = model.respond(system, messages, toolbox.schemas())
        stop.add_tokens(turn.input_tokens + turn.output_tokens)
        final_text = turn.text or final_text
        messages.append({"role": "assistant", "content": turn.content_blocks() or [{"type": "text", "text": "."}]})

        if not turn.tool_calls:
            reason = "done"
            break

        # ---- tool calls ----
        results = []
        for call in turn.tool_calls:
            if stop.exceeded():
                output = "ERROR: step limit reached, this call was not run."
            else:
                stop.add_step()
                t0 = time.perf_counter()
                output = handle_call(call, toolbox, approver, stop, only_invoice=only_invoice)
                log({
                    "step": stop.steps, "tool": call.name, "input": call.input, "output": output,
                    "ms": round((time.perf_counter() - t0) * 1000, 1),
                })
            results.append({
                "type": "tool_result", "tool_use_id": call.id, "content": output,
                "is_error": output.startswith("ERROR"),
            })
        messages.append({"role": "user", "content": results})

    return RunResult(
        stop_reason=reason, final_text=final_text, trajectory=trajectory, steps=stop.steps,
        tokens=stop.tokens, payout=stop.payout, seconds=round(time.perf_counter() - start, 3),
    )


def _sorted_by_due_date(toolbox: Toolbox, invoice_ids: list[str]) -> list[str]:
    """Soonest-due first. Invoices with no known due date (e.g. an ID that doesn't exist)
    sort last rather than crashing or sorting first."""
    due = toolbox.invoice_due_dates(invoice_ids)
    return sorted(invoice_ids, key=lambda i: due.get(i) or "9999-99-99")


def run_batch(model, toolbox: Toolbox, approver: Approver, stop: StopRule, invoice_ids: list[str],
              isolate: bool = True, system: str = SYSTEM_PROMPT, verbose: bool = False) -> RunResult:
    """Processes `invoice_ids`, always soonest-due-first (see `_sorted_by_due_date`): if the
    shared budget can't cover every invoice, the ones already paid when it runs out are the
    most urgent ones, and `unresolved` holds the rest -- the normal, incidental consequence
    of StopRule.budget_hit being sticky (agent/stop_rule.py), not special-cased here.

    With isolate=True (the default), each invoice gets its own run_agent() call with a fresh
    message history ("Process invoice INV-xxxx only."), so text injected into one invoice
    never sits in the same context as another invoice -- it can't reference or influence an
    invoice the model has no memory of reading. `stop` is shared across every invoice: steps,
    tokens and payout all accumulate across the whole batch, so the batch budget and step
    limit still hold exactly as they did for one big conversation.

    With isolate=False, falls back to today's single shared conversation over the whole list,
    for comparison.
    """
    invoice_ids = _sorted_by_due_date(toolbox, invoice_ids)

    if not isolate:
        task = (f"Process these invoices: {', '.join(invoice_ids)}." if invoice_ids
                else "Process all pending invoices.")
        return run_agent(model, toolbox, approver, stop, task, system=system, verbose=verbose)

    start = time.perf_counter()
    merged_trajectory = []
    final_parts = []
    reason = "done"
    unresolved: list = []

    for i, inv in enumerate(invoice_ids):
        if stop.exceeded():
            unresolved.extend(invoice_ids[i:])
            reason = stop.exceeded()
            break
        if verbose:
            print(f"\n--- {inv} (isolated) ---")
        r = run_agent(model, toolbox, approver, stop, f"Process invoice {inv} only.",
                      system=system, verbose=verbose, only_invoice=inv)
        for e in r.trajectory:
            merged_trajectory.append({**e, "invoice": inv})
        if r.final_text:
            final_parts.append(f"{inv}: {r.final_text}")
        reason = r.stop_reason
        if reason != "done":
            # the shared stop rule tripped mid-invoice: this one is unresolved too, and so is
            # everything after it -- the batch budget is what stopped the batch, not a decision.
            unresolved.extend(invoice_ids[i:])
            break

    return RunResult(
        stop_reason=reason, final_text="\n".join(final_parts), trajectory=merged_trajectory,
        steps=stop.steps, tokens=stop.tokens, payout=stop.payout,
        seconds=round(time.perf_counter() - start, 3), unresolved=unresolved,
    )


def handle_call(call, toolbox: Toolbox, approver: Approver, stop: StopRule, only_invoice: str | None = None) -> str:
    if only_invoice is not None:
        target = call.input.get("invoice_id")
        if target is not None and target != only_invoice:
            return f"ERROR: this run only handles {only_invoice}."

    args, err = toolbox.parse(call.name, call.input)
    if err:
        return err

    if call.name != "schedule_payment":
        return toolbox.execute(call.name, args)

    # ---- the write path: guard, budget, human, then pay ----
    try:
        proposal = toolbox.build_proposal(args)
    except ToolError as e:
        return f"ERROR: {e}"

    if not stop.can_pay(args.amount):
        stop.budget_hit = True
        return (f"ERROR: paying {args.amount} TND would exceed the batch budget of {stop.max_payout} TND "
                f"(already paid {stop.payout} TND). The run is stopping.")

    if not approver.approve(proposal):
        return f"Payment for {args.invoice_id} was REJECTED by the human reviewer. Do not retry it. Move on."

    output = toolbox.execute("schedule_payment", args)
    if not output.startswith("ERROR") and '"status": "paid"' in output:
        stop.add_payout(args.amount)
    return output
