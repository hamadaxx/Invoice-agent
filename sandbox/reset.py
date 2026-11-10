"""Wipe the sandbox database and reload the seed data.

Usage:  python -m sandbox.reset [--db path] [--seed path]
"""
import argparse
import json
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_DB = HERE / "sandbox.db"
DEFAULT_SEED = HERE / "seed.json"


def apply_seed_patch(db_path: Path, patch: dict) -> None:
    """Insert extra rows into an already-reset DB, for eval scenarios that need a
    situation the base seed doesn't cover (a new vendor, a PO that's already closed,
    an invoice that's already paid, ...). Never touches seed.json itself.

    `patch` keys are all optional lists of row dicts: "vendors", "purchase_orders",
    "invoices", "payments". Each dict may omit fields that have a sensible default
    (vendor active=1, PO status="open", invoice status="pending").
    """
    con = sqlite3.connect(db_path)
    try:
        for v in patch.get("vendors", []):
            con.execute(
                "INSERT INTO vendors (vendor_id, name, iban, active) VALUES (?, ?, ?, ?)",
                (v["vendor_id"], v["name"], v["iban"], v.get("active", 1)),
            )
        for po in patch.get("purchase_orders", []):
            con.execute(
                "INSERT INTO purchase_orders (po_number, vendor_id, description, amount, status) "
                "VALUES (?, ?, ?, ?, ?)",
                (po["po_number"], po["vendor_id"], po["description"], po["amount"], po.get("status", "open")),
            )
        for inv in patch.get("invoices", []):
            con.execute(
                "INSERT INTO invoices (invoice_id, vendor_name, po_number, amount, due_date, raw_text, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (inv["invoice_id"], inv["vendor_name"], inv.get("po_number"), inv["amount"], inv["due_date"],
                 inv["raw_text"], inv.get("status", "pending")),
            )
        for p in patch.get("payments", []):
            con.execute(
                "INSERT INTO payments (invoice_id, vendor_id, iban, amount) VALUES (?, ?, ?, ?)",
                (p["invoice_id"], p["vendor_id"], p["iban"], p["amount"]),
            )
        con.commit()
    finally:
        con.close()


def reset(db_path: Path = DEFAULT_DB, seed_path: Path = DEFAULT_SEED) -> None:
    seed = json.loads(Path(seed_path).read_text())
    con = sqlite3.connect(db_path)
    try:
        con.executescript((HERE / "schema.sql").read_text())
        con.executemany(
            "INSERT INTO vendors VALUES (:vendor_id, :name, :iban, :active)",
            seed["vendors"],
        )
        con.executemany(
            "INSERT INTO purchase_orders VALUES "
            "(:po_number, :vendor_id, :description, :amount, :status)",
            seed["purchase_orders"],
        )
        con.executemany(
            "INSERT INTO invoices (invoice_id, vendor_name, po_number, amount, due_date, raw_text) "
            "VALUES (:invoice_id, :vendor_name, :po_number, :amount, :due_date, :raw_text)",
            seed["invoices"],
        )
        con.commit()
    finally:
        con.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    a = p.parse_args()
    reset(a.db, a.seed)
    print(f"Sandbox reset: {a.db}")
