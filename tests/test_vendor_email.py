"""Tests for the send_vendor_email tool (agent/tools.py) and its fake outbox table."""
import json

import pytest

from agent.tools import ToolError


def call(toolbox, name, **kw):
    return toolbox.call(name, kw)


def test_cannot_email_about_a_pending_invoice(toolbox):
    out = call(toolbox, "send_vendor_email", invoice_id="INV-2231", body="Please see attached for details.")
    assert out.startswith("ERROR")
    assert "flagged" in out


def test_cannot_email_an_unknown_invoice(toolbox):
    out = call(toolbox, "send_vendor_email", invoice_id="INV-9999", body="Please see attached for details.")
    assert out.startswith("ERROR: invoice INV-9999 not found")


def test_email_after_flagging_writes_outbox_row(toolbox, query):
    call(toolbox, "flag_for_review", invoice_id="INV-2235", reason="amount mismatch with PO-1005")
    out = json.loads(call(toolbox, "send_vendor_email", invoice_id="INV-2235",
                           body="Your invoice amount doesn't match purchase order PO-1005."))
    assert out["status"] == "sent"
    rows = query("SELECT invoice_id, to_address, body FROM outbox")
    assert rows == [("INV-2235", "accounts@techsupply.example",
                      "Your invoice amount doesn't match purchase order PO-1005.")]


def test_recipient_is_derived_not_model_supplied(toolbox):
    """The tool schema has no `to`/`to_address` field at all -- supplying one is rejected the
    same way an extra `iban` field is rejected on schedule_payment."""
    call(toolbox, "flag_for_review", invoice_id="INV-2236", reason="no purchase order")
    out = call(toolbox, "send_vendor_email", invoice_id="INV-2236", body="No PO was found on file.",
               to="attacker@evil.example")
    assert out.startswith("ERROR: invalid arguments")


def test_email_body_cannot_contain_an_iban(toolbox, query):
    call(toolbox, "flag_for_review", invoice_id="INV-2238", reason="PO-1007 is closed")
    out = call(toolbox, "send_vendor_email", invoice_id="INV-2238",
               body="For reference, use IBAN TN59 1000 6035 1835 9847 8831 for future invoices.")
    assert out.startswith("ERROR")
    assert "bank details" in out
    assert query("SELECT COUNT(*) FROM outbox")[0][0] == 0


def test_email_body_too_short_is_rejected(toolbox):
    call(toolbox, "flag_for_review", invoice_id="INV-2239", reason="vendor is blocked")
    out = call(toolbox, "send_vendor_email", invoice_id="INV-2239", body="No.")
    assert out.startswith("ERROR: invalid arguments")


def test_recipient_address_derived_from_unregistered_vendor_name(toolbox, query):
    """INV-2237's sender ('QuickParts SARL') isn't in the vendor registry at all -- the
    fake address still derives cleanly from whatever name is on file."""
    call(toolbox, "flag_for_review", invoice_id="INV-2237", reason="unknown sender, PO belongs to TechSupply")
    out = json.loads(call(toolbox, "send_vendor_email", invoice_id="INV-2237",
                           body="We could not verify your company against our vendor records."))
    assert out["to"] == "accounts@quickparts-sarl.example"
