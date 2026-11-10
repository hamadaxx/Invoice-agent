"""End-to-end loop tests with a scripted model. No API key needed."""
from agent.checkpoint import AutoApprover, DenyAll, ScriptedApprover
from agent.loop import run_agent, run_batch
from agent.model import ScriptedModel
from agent.stop_rule import StopRule


def clean_path(inv, po, vendor, vid, amount, reason="PO open, amount and vendor match, never paid."):
    return [
        [("read_invoice", {"invoice_id": inv})],
        [("lookup_purchase_order", {"po_number": po})],
        [("get_vendor", {"vendor_name": vendor})],
        [("check_payment_history", {"invoice_id": inv})],
        [("schedule_payment", {"invoice_id": inv, "vendor_id": vid, "amount": amount, "reason": reason})],
    ]


def tools_used(result):
    return [e["tool"] for e in result.trajectory]


def test_clean_invoice_paid_after_approval(toolbox, query):
    model = ScriptedModel(clean_path("INV-2231", "PO-1001", "TechSupply", "V-01", 40000) + ["Paid INV-2231."])
    approver = AutoApprover()
    r = run_agent(model, toolbox, approver, StopRule(), "Process INV-2231")
    assert r.stop_reason == "done"
    assert approver.asked == ["INV-2231"]
    assert r.payout == 40000
    assert tools_used(r)[-1] == "schedule_payment"
    assert query("SELECT status FROM invoices WHERE invoice_id='INV-2231'")[0][0] == "paid"


def test_rejected_payment_moves_no_money(toolbox, query):
    model = ScriptedModel(clean_path("INV-2232", "PO-1002", "OfficePro", "V-02", 6000) + ["Rejected."])
    r = run_agent(model, toolbox, DenyAll(), StopRule(), "Process INV-2232")
    assert "REJECTED" in r.trajectory[-1]["output"]
    assert query("SELECT COUNT(*) FROM payments")[0][0] == 0


def test_invalid_payment_never_reaches_human(toolbox, query):
    """Amount mismatch is caught by the guard, so the human is never asked."""
    model = ScriptedModel([
        [("schedule_payment", {"invoice_id": "INV-2235", "vendor_id": "V-01", "amount": 6200,
                               "reason": "PO-1005 open, amount and vendor match, never paid."})],
        "Flagging.",
    ])
    approver = AutoApprover()
    r = run_agent(model, toolbox, approver, StopRule(), "Process INV-2235")
    assert approver.asked == []
    assert r.trajectory[0]["output"].startswith("ERROR")
    assert query("SELECT COUNT(*) FROM payments")[0][0] == 0


def test_budget_cap_stops_run(toolbox, query):
    model = ScriptedModel(
        clean_path("INV-2232", "PO-1002", "OfficePro", "V-02", 6000)
        + clean_path("INV-2231", "PO-1001", "TechSupply", "V-01", 40000)
        + ["should never get here"]
    )
    r = run_agent(model, toolbox, AutoApprover(), StopRule(max_payout=10_000), "Process both")
    assert r.stop_reason == "max_payout"
    assert r.payout == 6000
    assert query("SELECT invoice_id FROM payments") == [("INV-2232",)]


def test_partial_batch_with_scripted_approver(toolbox):
    model = ScriptedModel(
        clean_path("INV-2232", "PO-1002", "OfficePro", "V-02", 6000)
        + clean_path("INV-2233", "PO-1003", "CleanCo", "V-03", 1800)
        + ["Done."]
    )
    approver = ScriptedApprover(decisions={"INV-2232": True, "INV-2233": False})
    r = run_agent(model, toolbox, approver, StopRule(), "Process both")
    assert r.payout == 6000
    assert approver.asked == ["INV-2232", "INV-2233"]


def test_trajectory_is_logged(toolbox):
    model = ScriptedModel([[("list_invoices", {})], "ok"])
    r = run_agent(model, toolbox, AutoApprover(), StopRule(), "list")
    e = r.trajectory[0]
    assert set(e) == {"step", "tool", "input", "output", "ms"}


def test_injection_double_payment_blocked(toolbox, query):
    """INV-2242 asks to pay INV-2231 again. Even if the model obeys, the guard stops it."""
    model = ScriptedModel(
        clean_path("INV-2231", "PO-1001", "TechSupply", "V-01", 40000)
        + [[("read_invoice", {"invoice_id": "INV-2242"})],
           [("schedule_payment", {"invoice_id": "INV-2231", "vendor_id": "V-01", "amount": 40000,
                                  "reason": "the invoice said to pay it again"})],
           "done"]
    )
    run_agent(model, toolbox, AutoApprover(), StopRule(), "Process INV-2231 and INV-2242")
    assert query("SELECT COUNT(*) FROM payments WHERE invoice_id='INV-2231'")[0][0] == 1


def test_injection_new_iban_cannot_be_used(toolbox, query):
    """INV-2240 asks for a new IBAN. The tool has no IBAN field, and the bank uses the registry."""
    model = ScriptedModel([
        [("schedule_payment", {"invoice_id": "INV-2240", "vendor_id": "V-02", "amount": 1200,
                               "reason": "PO-1006 open, amount and vendor match, never paid.",
                               "iban": "FR76 3000 6000 0112 3456 7890 189"})],
        [("schedule_payment", {"invoice_id": "INV-2240", "vendor_id": "V-02", "amount": 1200,
                               "reason": "PO-1006 open, amount and vendor match, never paid."})],
        "done",
    ])
    run_agent(model, toolbox, AutoApprover(), StopRule(), "Process INV-2240")
    rows = query("SELECT iban FROM payments")
    assert rows == [("TN59 0800 1000 0123 4567 8901",)]  # OfficePro's registry IBAN, not the FR76 one


# ───────────────────────── run_batch / isolation ─────────────────────────

def test_isolated_run_blocks_cross_invoice_calls(toolbox, query):
    """INV-2242's injected text asks to also read/pay INV-2231. In an isolated per-invoice run,
    even attempting to touch a different invoice_id is refused in code, before the tool runs."""
    model = ScriptedModel([
        [("read_invoice", {"invoice_id": "INV-2242"})],
        [("read_invoice", {"invoice_id": "INV-2231"})],  # tries to peek at the other invoice
        [("schedule_payment", {"invoice_id": "INV-2231", "vendor_id": "V-01", "amount": 40000,
                               "reason": "the other invoice asked for it"})],
        "done",
    ])
    r = run_batch(model, toolbox, AutoApprover(), StopRule(), ["INV-2242"], isolate=True)
    cross_calls = [e for e in r.trajectory if e["input"].get("invoice_id") == "INV-2231"]
    assert len(cross_calls) == 2
    assert all(e["output"] == "ERROR: this run only handles INV-2242." for e in cross_calls)
    assert query("SELECT COUNT(*) FROM payments")[0][0] == 0
    assert all(e["invoice"] == "INV-2242" for e in r.trajectory)


def test_isolated_batch_shares_the_stop_rule(toolbox, query):
    """Two clean invoices, each in its own isolated conversation, still share one batch budget.
    run_batch always processes soonest-due-first: INV-2233 (due 2026-11-30) before INV-2231
    (due 2026-12-01), regardless of the order they're given in."""
    model = ScriptedModel(
        clean_path("INV-2233", "PO-1003", "CleanCo", "V-03", 1800) + ["Paid INV-2233."]
        + clean_path("INV-2231", "PO-1001", "TechSupply", "V-01", 40000) + ["should never get here"]
    )
    stop = StopRule(max_payout=5_000)
    r = run_batch(model, toolbox, AutoApprover(), stop, ["INV-2231", "INV-2233"], isolate=True)
    assert r.stop_reason == "max_payout"
    assert r.payout == 1800
    assert r.unresolved == ["INV-2231"]
    assert query("SELECT invoice_id FROM payments") == [("INV-2233",)]


def test_isolate_false_falls_back_to_one_shared_conversation(toolbox, query):
    model = ScriptedModel(
        clean_path("INV-2232", "PO-1002", "OfficePro", "V-02", 6000)
        + clean_path("INV-2233", "PO-1003", "CleanCo", "V-03", 1800)
        + ["Done."]
    )
    r = run_batch(model, toolbox, AutoApprover(), StopRule(), ["INV-2232", "INV-2233"], isolate=False)
    assert r.stop_reason == "done"
    assert r.payout == 7800
    assert query("SELECT COUNT(*) FROM payments")[0][0] == 2


# ───────────────────────── run_batch / due-date ordering ─────────────────────────

def test_run_batch_processes_soonest_due_first(toolbox):
    """Given in a scrambled order -- INV-2231 (due 12-01), INV-2234 (due 12-10), INV-2233
    (due 11-30) -- run_batch always processes due-soonest first, regardless of input order."""
    model = ScriptedModel(
        clean_path("INV-2233", "PO-1003", "CleanCo", "V-03", 1800) + ["ok"]
        + clean_path("INV-2231", "PO-1001", "TechSupply", "V-01", 40000) + ["ok"]
        + clean_path("INV-2234", "PO-1004", "NetLink Telecom", "V-04", 4500) + ["ok"]
    )
    r = run_batch(model, toolbox, AutoApprover(), StopRule(), ["INV-2231", "INV-2234", "INV-2233"], isolate=True)
    paid_order = [e["invoice"] for e in r.trajectory if e["tool"] == "schedule_payment"]
    assert paid_order == ["INV-2233", "INV-2231", "INV-2234"]  # 11-30, 12-01, 12-10


def test_due_date_sort_handles_an_unknown_invoice_id(toolbox):
    """An invoice_id with no row at all (so no due_date) sorts last instead of crashing."""
    model = ScriptedModel([[("read_invoice", {"invoice_id": "INV-2233"})], "not found", "ok"])
    r = run_batch(model, toolbox, AutoApprover(), StopRule(), ["INV-9999", "INV-2233"], isolate=True)
    assert [e["invoice"] for e in r.trajectory][0] == "INV-2233"
