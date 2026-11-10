-- Fake company data for the Invoice Payment Agent sandbox.
-- Everything here is resettable with `make reset`.

DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS outbox;
DROP TABLE IF EXISTS invoices;
DROP TABLE IF EXISTS purchase_orders;
DROP TABLE IF EXISTS vendors;

-- Trusted list of suppliers. The ONLY source of bank details.
CREATE TABLE vendors (
    vendor_id   TEXT PRIMARY KEY,          -- e.g. V-01
    name        TEXT NOT NULL UNIQUE,
    iban        TEXT NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1 -- 0 = blocked vendor
);

-- What the company actually ordered.
CREATE TABLE purchase_orders (
    po_number   TEXT PRIMARY KEY,          -- e.g. PO-1001
    vendor_id   TEXT NOT NULL REFERENCES vendors(vendor_id),
    description TEXT NOT NULL,
    amount      REAL NOT NULL CHECK (amount > 0),
    status      TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed'))
);

-- Invoices received. `raw_text` is UNTRUSTED: it comes from outside.
CREATE TABLE invoices (
    invoice_id  TEXT PRIMARY KEY,          -- e.g. INV-2231
    vendor_name TEXT NOT NULL,             -- as printed on the invoice, may not match registry
    po_number   TEXT,                      -- may be missing or wrong
    amount      REAL NOT NULL,
    due_date    TEXT NOT NULL,
    raw_text    TEXT NOT NULL,             -- full invoice body, attack surface
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'paid', 'flagged', 'rejected'))
);

-- Payments made by the mock bank. invoice_id UNIQUE = idempotency key.
CREATE TABLE payments (
    payment_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id  TEXT NOT NULL UNIQUE REFERENCES invoices(invoice_id),
    vendor_id   TEXT NOT NULL REFERENCES vendors(vendor_id),
    iban        TEXT NOT NULL,             -- copied from vendors at payment time
    amount      REAL NOT NULL CHECK (amount > 0),
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Invoices the agent flagged for a human.
CREATE TABLE reviews (
    review_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id  TEXT NOT NULL REFERENCES invoices(invoice_id),
    reason      TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Fake outbox for vendor notifications. Nothing is actually emailed; this just records what
-- WOULD be sent, so the tool is a real write with a real attack surface (see send_vendor_email
-- in agent/tools.py) without needing a real mail server.
CREATE TABLE outbox (
    email_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id  TEXT NOT NULL REFERENCES invoices(invoice_id),
    to_address  TEXT NOT NULL,              -- derived from the vendor on file, never model-supplied
    body        TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
