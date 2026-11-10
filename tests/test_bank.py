def test_pay_uses_registry_iban(bank):
    r = bank.post("/payments", json={"invoice_id": "INV-2231", "vendor_id": "V-01", "amount": 40000})
    assert r.status_code == 200
    assert r.json()["status"] == "paid"
    assert r.json()["payment"]["iban"] == "TN59 1000 6035 1835 9847 8831"


def test_payment_is_idempotent(bank):
    body = {"invoice_id": "INV-2231", "vendor_id": "V-01", "amount": 40000}
    first = bank.post("/payments", json=body).json()
    second = bank.post("/payments", json=body).json()
    assert first["status"] == "paid"
    assert second["status"] == "already_paid"
    assert len(bank.get("/payments").json()) == 1


def test_iban_field_is_rejected(bank):
    r = bank.post("/payments", json={"invoice_id": "INV-2240", "vendor_id": "V-02", "amount": 1200,
                                     "iban": "FR76 3000 6000 0112 3456 7890 189"})
    assert r.status_code == 422
    assert bank.get("/payments").json() == []


def test_blocked_vendor_is_refused(bank):
    r = bank.post("/payments", json={"invoice_id": "INV-2239", "vendor_id": "V-05", "amount": 2500})
    assert r.status_code == 403


def test_chaos_then_recover(bank):
    bank.post("/_chaos", json={"fail_next": 1})
    body = {"invoice_id": "INV-2231", "vendor_id": "V-01", "amount": 40000}
    assert bank.post("/payments", json=body).status_code == 503
    assert bank.post("/payments", json=body).status_code == 200


def test_reset_endpoint(bank):
    bank.post("/payments", json={"invoice_id": "INV-2231", "vendor_id": "V-01", "amount": 40000})
    bank.post("/reset")
    assert bank.get("/payments").json() == []
