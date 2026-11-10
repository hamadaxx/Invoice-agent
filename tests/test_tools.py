import json

import pytest

from agent.tools import PaymentIn, ToolError


def call(toolbox, name, **kw):
    return toolbox.call(name, kw)


def test_list_pending(toolbox):
    out = json.loads(call(toolbox, "list_invoices"))
    assert len(out["invoice_ids"]) == 12


def test_read_invoice_wraps_untrusted_text(toolbox):
    out = json.loads(call(toolbox, "read_invoice", invoice_id="INV-2240"))
    assert out["raw_text"].startswith("<untrusted_vendor_text>")


def test_unknown_tool(toolbox):
    assert call(toolbox, "delete_registry").startswith("ERROR: unknown tool")


def test_bad_arguments_give_clear_error(toolbox):
    out = call(toolbox, "schedule_payment", invoice_id="INV-2231", vendor_id="V-01", amount="lots")
    assert out.startswith("ERROR: invalid arguments for schedule_payment")
    assert "amount" in out


def test_extra_iban_argument_rejected(toolbox):
    out = call(toolbox, "schedule_payment", invoice_id="INV-2240", vendor_id="V-02", amount=1200,
               reason="matches PO-1006, clean invoice.", iban="FR76 3000 6000 0112 3456 7890 189")
    assert out.startswith("ERROR: invalid arguments")


def test_missing_po(toolbox):
    assert "does not exist" in call(toolbox, "lookup_purchase_order", po_number="PO-9999")


def test_unknown_vendor(toolbox):
    assert "not in the registry" in call(toolbox, "get_vendor", vendor_name="QuickParts SARL")


def test_vendor_lookup_ignores_case(toolbox):
    assert json.loads(call(toolbox, "get_vendor", vendor_name="techsupply"))["vendor_id"] == "V-01"


@pytest.mark.parametrize("invoice_id,vendor_id,amount,expected", [
    ("INV-2235", "V-01", 6200, "differs from PO amount"),           # invoice matches amount, PO doesn't
    ("INV-2236", "V-02", 950, "no purchase order"),                 # no PO
    ("INV-2237", "V-02", 22000, "does not match the purchase order vendor"),
    ("INV-2238", "V-03", 1800, "missing or closed"),                # closed PO
    ("INV-2239", "V-05", 2500, "unknown or blocked"),                # blocked vendor
    ("INV-2231", "V-01", 1, "differs from the invoice amount"),     # wrong amount on a clean invoice
    ("INV-2237", "V-01", 22000, "comes from 'QuickParts SARL'"),    # unknown sender, PO vendor matches
])
def test_payment_guard(toolbox, invoice_id, vendor_id, amount, expected):
    with pytest.raises(ToolError, match=expected):
        toolbox.build_proposal(PaymentIn(invoice_id=invoice_id, vendor_id=vendor_id, amount=amount,
                                          reason="testing the payment guard checks."))


def test_unknown_sender_cannot_borrow_po_vendor(toolbox, query):
    """INV-2237 is from QuickParts SARL but quotes TechSupply's PO-1010. Paying V-01 would send
    money to TechSupply for an invoice nobody at TechSupply sent."""
    out = call(toolbox, "schedule_payment", invoice_id="INV-2237", vendor_id="V-01", amount=22000,
               reason="PO-1010 open, amount matches, vendor matches PO.")
    assert out.startswith("ERROR")
    assert query("SELECT COUNT(*) FROM payments")[0][0] == 0


def test_one_payment_per_po(toolbox, query):
    out = json.loads(call(toolbox, "schedule_payment", invoice_id="INV-2242", vendor_id="V-01", amount=22000,
                           reason="PO-1010 open, amount and vendor match, never paid."))
    assert out["status"] == "paid"
    assert query("SELECT status FROM purchase_orders WHERE po_number='PO-1010'")[0][0] == "closed"
    with pytest.raises(ToolError):
        toolbox.build_proposal(PaymentIn(invoice_id="INV-2237", vendor_id="V-01", amount=22000,
                                          reason="testing the one-payment-per-PO guard."))


def test_pay_then_cannot_pay_again(toolbox, query):
    out = json.loads(call(toolbox, "schedule_payment", invoice_id="INV-2231", vendor_id="V-01", amount=40000,
                           reason="PO-1001 open, amount and vendor match, never paid."))
    assert out["status"] == "paid"
    again = call(toolbox, "schedule_payment", invoice_id="INV-2231", vendor_id="V-01", amount=40000,
                 reason="PO-1001 open, amount and vendor match, never paid.")
    assert again.startswith("ERROR")
    assert query("SELECT COUNT(*) FROM payments")[0][0] == 1


def test_bank_down_once_is_retried(toolbox, bank):
    bank.post("/_chaos", json={"fail_next": 1})
    out = json.loads(call(toolbox, "schedule_payment", invoice_id="INV-2231", vendor_id="V-01", amount=40000,
                           reason="PO-1001 open, amount and vendor match, never paid."))
    assert out["status"] == "paid"


def test_bank_down_twice_gives_clear_error(toolbox, bank, query):
    bank.post("/_chaos", json={"fail_next": 2})
    out = call(toolbox, "schedule_payment", invoice_id="INV-2231", vendor_id="V-01", amount=40000,
               reason="PO-1001 open, amount and vendor match, never paid.")
    assert out.startswith("ERROR: bank temporarily unavailable")
    assert "Flag the invoice" in out
    assert query("SELECT status FROM invoices WHERE invoice_id='INV-2231'")[0][0] == "pending"


def test_flag_is_idempotent(toolbox, query):
    call(toolbox, "flag_for_review", invoice_id="INV-2235", reason="amount mismatch")
    call(toolbox, "flag_for_review", invoice_id="INV-2235", reason="amount mismatch")
    assert query("SELECT COUNT(*) FROM reviews")[0][0] == 1


def test_schemas_are_valid_for_api(toolbox):
    schemas = toolbox.schemas()
    assert len(schemas) == 8
    assert all(s["input_schema"]["type"] == "object" for s in schemas)


def test_invoice_due_dates_returns_mapping(toolbox):
    dates = toolbox.invoice_due_dates(["INV-2231", "INV-2233"])
    assert dates == {"INV-2231": "2026-12-01", "INV-2233": "2026-11-30"}


def test_invoice_due_dates_skips_unknown_ids(toolbox):
    assert toolbox.invoice_due_dates(["INV-9999"]) == {}


def test_invoice_due_dates_empty_list(toolbox):
    assert toolbox.invoice_due_dates([]) == {}
