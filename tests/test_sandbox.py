import sqlite3

import pytest

from sandbox.reset import reset


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    reset(path)
    con = sqlite3.connect(path)
    yield con, path
    con.close()


def count(con, table):
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_seed_loads(db):
    con, _ = db
    assert count(con, "vendors") == 5
    assert count(con, "purchase_orders") == 10
    assert count(con, "invoices") == 12
    assert count(con, "payments") == 0


def test_reset_wipes_payments(db):
    con, path = db
    con.execute(
        "INSERT INTO payments (invoice_id, vendor_id, iban, amount) VALUES ('INV-2231','V-01','x',1)"
    )
    con.commit()
    assert count(con, "payments") == 1
    reset(path)
    assert count(con, "payments") == 0


def test_same_invoice_cannot_be_paid_twice(db):
    con, _ = db
    sql = "INSERT INTO payments (invoice_id, vendor_id, iban, amount) VALUES ('INV-2231','V-01','x',40000)"
    con.execute(sql)
    with pytest.raises(sqlite3.IntegrityError):
        con.execute(sql)
