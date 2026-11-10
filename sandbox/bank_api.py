"""Mock bank API. The only place money moves.

Run on its own:  uvicorn sandbox.bank_api:app --port 8001
The agent can also use it in-process (see agent/run.py), so no server is needed for tests.

Safety rules enforced HERE, whatever the agent asks:
- The IBAN always comes from the vendor registry. The request has no IBAN field.
- One payment per invoice_id. A second call returns "already_paid" and moves nothing.
- Blocked vendors cannot be paid.
"""
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from sandbox.reset import DEFAULT_DB, reset


class PaymentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # an "iban" field is rejected
    invoice_id: str = Field(pattern=r"^INV-\d{4}$")
    vendor_id: str = Field(pattern=r"^V-\d{2}$")
    amount: float = Field(gt=0)


class ChaosRequest(BaseModel):
    fail_next: int = Field(ge=0, le=100)


def create_app(db_path: str | Path) -> FastAPI:
    app = FastAPI(title="Mock Bank")
    app.state.fail_next = 0  # chaos switch for error tests

    def connect():
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        return con

    def get_payment(con, invoice_id):
        row = con.execute("SELECT * FROM payments WHERE invoice_id = ?", (invoice_id,)).fetchone()
        return dict(row) if row else None

    @app.post("/payments")
    def make_payment(req: PaymentRequest):
        if app.state.fail_next > 0:
            app.state.fail_next -= 1
            raise HTTPException(503, "Bank temporarily unavailable")

        with closing(connect()) as con:
            existing = get_payment(con, req.invoice_id)
            if existing:
                return {"status": "already_paid", "payment": existing}

            vendor = con.execute(
                "SELECT * FROM vendors WHERE vendor_id = ?", (req.vendor_id,)
            ).fetchone()
            if vendor is None:
                raise HTTPException(404, f"Unknown vendor {req.vendor_id}")
            if not vendor["active"]:
                raise HTTPException(403, f"Vendor {req.vendor_id} is blocked")

            try:
                con.execute(
                    "INSERT INTO payments (invoice_id, vendor_id, iban, amount) VALUES (?, ?, ?, ?)",
                    (req.invoice_id, req.vendor_id, vendor["iban"], req.amount),
                )
                con.commit()
            except sqlite3.IntegrityError:
                # Two calls raced. The first one won, so this one is a duplicate.
                return {"status": "already_paid", "payment": get_payment(con, req.invoice_id)}

            return {"status": "paid", "payment": get_payment(con, req.invoice_id)}

    @app.get("/payments")
    def list_payments():
        with closing(connect()) as con:
            return [dict(r) for r in con.execute("SELECT * FROM payments ORDER BY payment_id")]

    @app.post("/reset")
    def reset_sandbox():
        reset(Path(db_path))
        app.state.fail_next = 0
        return {"status": "reset"}

    @app.post("/_chaos")
    def chaos(req: ChaosRequest):
        app.state.fail_next = req.fail_next
        return {"fail_next": req.fail_next}

    return app


app = create_app(os.environ.get("SANDBOX_DB", str(DEFAULT_DB)))
