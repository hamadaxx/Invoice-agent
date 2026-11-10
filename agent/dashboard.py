"""Read-only view of the sandbox DB's current state: every invoice (status, due date,
amount), what it paid or was flagged for, and any vendor notification sent about it.

Two ways to see it:
  - Embedded in the live approval page (agent/web_approver.py) when running with
    --approver web, updating in real time as a run proceeds.
  - Standalone, any time, even with no agent run active:
        python -m agent.dashboard --db sandbox/sandbox.db [--port 8788]

Both read the same `query_state()` and share the same rendering (DASHBOARD_SCRIPT below),
so there's exactly one place that knows how to turn the DB into a dashboard.
"""
import argparse
import sqlite3
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

DEFAULT_PORT = 8788


def query_state(db_path) -> dict:
    """Everything the dashboard shows, in one call. Joins payments/reviews onto invoices
    (at most one of each per invoice in practice -- flag_for_review and schedule_payment are
    both idempotent) and counts outbox emails per invoice separately, since a vendor could in
    principle be emailed more than once."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        invoices = [dict(r) for r in con.execute("""
            SELECT i.invoice_id, i.vendor_name, i.po_number, i.amount, i.due_date, i.status,
                   p.vendor_id AS paid_vendor_id, p.iban AS paid_iban, p.created_at AS paid_at,
                   rv.reason AS review_reason,
                   (SELECT COUNT(*) FROM outbox o WHERE o.invoice_id = i.invoice_id) AS emails_sent
            FROM invoices i
            LEFT JOIN payments p ON p.invoice_id = i.invoice_id
            LEFT JOIN reviews rv ON rv.invoice_id = i.invoice_id
            ORDER BY i.due_date, i.invoice_id
        """)]
        outbox = [dict(r) for r in con.execute(
            "SELECT invoice_id, to_address, body, created_at FROM outbox ORDER BY email_id DESC"
        )]
    finally:
        con.close()

    totals = {"pending": 0, "paid": 0, "flagged": 0, "rejected": 0}
    paid_amount = 0.0
    for inv in invoices:
        totals[inv["status"]] = totals.get(inv["status"], 0) + 1
        if inv["status"] == "paid":
            paid_amount += inv["amount"]
    totals["paid_amount"] = paid_amount

    return {"invoices": invoices, "outbox": outbox, "totals": totals}


# ───────────────────────── shared rendering ─────────────────────────

DASHBOARD_STYLE = """
  .dash-summary { display: flex; gap: .6rem; flex-wrap: wrap; margin-bottom: 1rem; }
  .pill {
    background: #fff; border: 1px solid #e3e5e9; border-radius: 8px; padding: .5rem .9rem;
    font-size: .82rem; color: #6b7280;
  }
  .pill b { color: #1b1f24; font-size: 1rem; display: block; }
  table.dash { width: 100%; border-collapse: collapse; font-size: .83rem; }
  table.dash th, table.dash td { text-align: left; padding: .5rem .5rem; border-bottom: 1px solid #edeef1; }
  table.dash th { color: #6b7280; font-weight: 600; font-size: .75rem; text-transform: uppercase; }
  .status { padding: .12rem .55rem; border-radius: 20px; font-size: .74rem; font-weight: 650; white-space: nowrap; }
  .status.pending { background: #fff3cd; color: #7a5b00; }
  .status.paid { background: #dafbe1; color: #116329; }
  .status.flagged { background: #ffebe9; color: #82071e; }
  .status.rejected { background: #eee; color: #555; }
  .mono { font-family: ui-monospace, Consolas, monospace; font-size: .8rem; }
  .muted { color: #9aa1ac; }
  .email-card {
    background: #fff; border: 1px solid #e3e5e9; border-radius: 8px; padding: .75rem .9rem;
    margin-bottom: .6rem; font-size: .85rem;
  }
  .email-card .meta { color: #6b7280; font-size: .78rem; margin-bottom: .3rem; }
  .last-updated { color: #9aa1ac; font-size: .78rem; margin-top: .5rem; }
"""

DASHBOARD_SCRIPT = """
function escD(s) { return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function money(n) { return Number(n).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2}); }

function renderDashboard(data) {
  const t = data.totals;
  document.getElementById('dash-summary').innerHTML = `
    <div class="pill"><b>${t.pending}</b>pending</div>
    <div class="pill"><b>${t.paid}</b>paid</div>
    <div class="pill"><b>${t.flagged}</b>flagged</div>
    <div class="pill"><b>${t.rejected || 0}</b>rejected</div>
    <div class="pill"><b>${money(t.paid_amount)}</b>TND paid</div>
    <div class="pill"><b>${data.outbox.length}</b>emails sent</div>
  `;

  const rows = data.invoices.map(inv => {
    let note = '<span class="muted">&mdash;</span>';
    if (inv.status === 'paid') {
      note = `IBAN ${escD(inv.paid_iban || '')}`;
    } else if (inv.status === 'flagged') {
      note = escD(inv.review_reason || '');
      if (inv.emails_sent > 0) note += ` <span class="muted">(vendor emailed)</span>`;
    }
    return `<tr>
      <td class="mono">${escD(inv.invoice_id)}</td>
      <td>${escD(inv.vendor_name)}</td>
      <td class="mono">${escD(inv.po_number || '-')}</td>
      <td>${money(inv.amount)}</td>
      <td class="mono">${escD(inv.due_date)}</td>
      <td><span class="status ${inv.status}">${escD(inv.status)}</span></td>
      <td>${note}</td>
    </tr>`;
  }).join('');
  document.getElementById('dash-invoices').innerHTML = `
    <table class="dash"><thead><tr>
      <th>Invoice</th><th>Vendor</th><th>PO</th><th>Amount</th><th>Due</th><th>Status</th><th>Note</th>
    </tr></thead><tbody>${rows || '<tr><td colspan="7" class="muted">No invoices.</td></tr>'}</tbody></table>`;

  const emails = data.outbox.map(e => `
    <div class="email-card">
      <div class="meta">${escD(e.created_at)} &middot; ${escD(e.invoice_id)} &rarr; ${escD(e.to_address)}</div>
      ${escD(e.body)}
    </div>`).join('');
  document.getElementById('dash-outbox').innerHTML =
    emails || '<p class="muted">No vendor emails sent yet.</p>';
}
"""


def dashboard_section_html(standalone: bool) -> str:
    """The <h2>/<div> blocks for the dashboard. `standalone` just changes the top margin
    since the embedded version sits below the approval card/history."""
    top_margin = "0" if standalone else "2rem"
    return f"""
<div style="margin-top:{top_margin}">
  <h2>Batch overview</h2>
  <div id="dash-summary" class="dash-summary"></div>
  <div id="dash-invoices"></div>
  <h2>Vendor emails sent</h2>
  <div id="dash-outbox"></div>
  <p class="last-updated" id="dash-updated"></p>
</div>
"""


STANDALONE_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Invoice Dashboard</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 2rem 1rem; background: #f4f5f7; min-height: 100vh;
    font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; color: #1b1f24;
  }}
  .wrap {{ max-width: 920px; margin: 0 auto; }}
  h1 {{ font-size: 1.25rem; margin: 0 0 1.25rem; }}
  h2 {{ font-size: .95rem; color: #6b7280; margin: 1.75rem 0 .5rem; font-weight: 600; }}
  {style}
</style>
</head>
<body>
<div class="wrap">
  <h1>Invoice Dashboard</h1>
  {section}
</div>
<script>
{script}
async function poll() {{
  try {{
    const r = await fetch('/api/dashboard');
    renderDashboard(await r.json());
    document.getElementById('dash-updated').textContent = 'Updated ' + new Date().toLocaleTimeString();
  }} catch (e) {{ /* DB momentarily locked mid-write; just retry */ }}
  setTimeout(poll, 1500);
}}
poll();
</script>
</body>
</html>
"""


def standalone_page() -> str:
    return STANDALONE_PAGE_TEMPLATE.format(
        style=DASHBOARD_STYLE, script=DASHBOARD_SCRIPT, section=dashboard_section_html(standalone=True)
    )


def create_dashboard_app(db_path) -> FastAPI:
    app = FastAPI(title="Invoice Dashboard")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return standalone_page()

    @app.get("/api/dashboard")
    def dashboard():
        return query_state(db_path)

    @app.get("/api/ping")
    def ping():
        return {"ok": True}

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="sandbox/sandbox.db")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()

    db_path = Path(a.db)
    if not db_path.exists():
        print(f"No DB at {db_path} yet -- run `make reset` or `python -m agent.run` first.")
        return

    print(f"\n=== Dashboard: http://127.0.0.1:{a.port}/ (db: {db_path}) ===")
    if not a.no_browser:
        webbrowser.open(f"http://127.0.0.1:{a.port}/")
    uvicorn.Server(uvicorn.Config(create_dashboard_app(db_path), host="127.0.0.1",
                                  port=a.port, log_level="warning")).run()


if __name__ == "__main__":
    main()
