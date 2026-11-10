"""Web-based human checkpoint: a local page with Approve/Reject buttons instead of
typing y/N in the terminal. The agent loop still calls `approve(proposal)` and blocks
until a decision comes back -- only *where* the human looks changes.

This runs its own tiny FastAPI server in a background thread, so it can serve pages
while the main thread (running the agent loop) blocks on `decide()` waiting for a
click. The approval card itself only ever sees the fields already on a PaymentProposal,
the same ones ConsoleApprover prints -- it needs no DB access. When `db_path` is given,
the page also embeds the live batch dashboard (agent/dashboard.py): every invoice's
status, what it paid or was flagged for, and any vendor email sent about it, polling
the DB directly and updating as the run proceeds -- so one page shows the whole run,
not just whatever's waiting on a decision right now.

The approval card deliberately shows the invoice sender right next to the registry
vendor name and flags it if they differ -- see postmortem.md: last run's incident
happened because the human approval screen had no way to show that mismatch, so an
approval looked routine when it wasn't. It also shows the agent's own one-line reason
for paying, and any warnings from the regex tripwire or the quarantine extraction
(agent/quarantine.py) even if the agent's reason doesn't mention them.
"""
import queue
import threading
import time
import webbrowser

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from agent.checkpoint import Approver, PaymentProposal
from agent.dashboard import DASHBOARD_SCRIPT, DASHBOARD_STYLE, dashboard_section_html, query_state

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Payment Approval</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2rem 1rem; background: #f4f5f7; min-height: 100vh;
    font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    color: #1b1f24;
  }
  .wrap { max-width: 640px; margin: 0 auto; }
  h1 { font-size: 1.25rem; margin: 0 0 1.25rem; display: flex; align-items: center; gap: .5rem; }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: #9aa1ac; display: inline-block; }
  .dot.live { background: #2ea043; box-shadow: 0 0 0 0 rgba(46,160,67,.5); animation: pulse 1.6s infinite; }
  @keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(46,160,67,.45); }
    70% { box-shadow: 0 0 0 7px rgba(46,160,67,0); }
    100% { box-shadow: 0 0 0 0 rgba(46,160,67,0); }
  }
  .idle {
    background: #fff; border: 1px solid #e3e5e9; border-radius: 10px;
    padding: 2.5rem 1.5rem; text-align: center; color: #6b7280;
  }
  .card {
    background: #fff; border: 1px solid #e3e5e9; border-radius: 10px;
    padding: 1.5rem; box-shadow: 0 1px 2px rgba(0,0,0,.04);
  }
  .amount { font-size: 1.9rem; font-weight: 650; margin: 0 0 .15rem; }
  .sub { color: #6b7280; font-size: .9rem; margin: 0 0 1.1rem; }
  .rows { display: grid; grid-template-columns: auto 1fr; gap: .45rem 1rem; font-size: .92rem; }
  .rows dt { color: #6b7280; }
  .rows dd { margin: 0; font-weight: 500; word-break: break-word; }
  .warn { color: #b42318; font-weight: 650; }
  .warn-box {
    margin-top: 1.1rem; padding: .7rem .9rem; border-radius: 8px;
    background: #fef2f2; border: 1px solid #fecaca; color: #b42318; font-size: .88rem;
  }
  .reason-box {
    margin-top: 1.1rem; padding: .7rem .9rem; border-radius: 8px;
    background: #f4f5f7; border: 1px solid #e3e5e9; font-size: .88rem;
  }
  .reason-box .label { color: #6b7280; font-weight: 600; margin-right: .3rem; }
  .warn-box ul { margin: .4rem 0 0; padding-left: 1.1rem; }
  .btns { display: flex; gap: .75rem; margin-top: 1.4rem; }
  button {
    flex: 1; padding: .7rem 1rem; border-radius: 8px; border: none;
    font-size: .95rem; font-weight: 600; cursor: pointer; transition: opacity .15s;
  }
  button:hover { opacity: .88; }
  button:disabled { opacity: .5; cursor: default; }
  .approve { background: #2ea043; color: #fff; }
  .reject { background: #d1242f; color: #fff; }
  table { width: 100%; border-collapse: collapse; font-size: .85rem; margin-top: .75rem; }
  th, td { text-align: left; padding: .5rem .4rem; border-bottom: 1px solid #edeef1; }
  th { color: #6b7280; font-weight: 600; }
  .tag { padding: .12rem .5rem; border-radius: 20px; font-size: .76rem; font-weight: 650; }
  .tag.approved { background: #dafbe1; color: #116329; }
  .tag.rejected { background: #ffebe9; color: #82071e; }
  h2 { font-size: .95rem; color: #6b7280; margin: 1.75rem 0 .5rem; font-weight: 600; }
  .empty-hist { color: #9aa1ac; font-size: .85rem; }
""" + DASHBOARD_STYLE + """
</style>
</head>
<body>
<div class="wrap">
  <h1><span class="dot" id="dot"></span>Payment Approval</h1>
  <div id="area"><div class="idle">Waiting for the next payment&hellip;</div></div>
  <h2>Decisions this run</h2>
  <div id="history"><div class="empty-hist">Nothing decided yet.</div></div>
""" + dashboard_section_html(standalone=False) + """
</div>
<script>
""" + DASHBOARD_SCRIPT + """

async function poll() {
  try {
    const r = await fetch('/api/state');
    const data = await r.json();
    document.getElementById('dot').classList.add('live');
    render(data);
  } catch (e) {
    document.getElementById('dot').classList.remove('live');
  }
  try {
    const dr = await fetch('/api/dashboard');
    const ddata = await dr.json();
    if (ddata.available !== false) {
      renderDashboard(ddata);
      document.getElementById('dash-updated').textContent = 'Updated ' + new Date().toLocaleTimeString();
    }
  } catch (e) { /* no --db configured for this approver, or DB momentarily locked mid-write */ }
  setTimeout(poll, 900);
}

function esc(s) { return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function render(data) {
  const area = document.getElementById('area');
  const p = data.pending;
  if (!p) {
    area.innerHTML = '<div class="idle">Waiting for the next payment&hellip;</div>';
  } else {
    const mismatchBox = p.sender_mismatch ? `
      <div class="warn-box">⚠ The invoice was sent by <strong>${esc(p.invoice_sender)}</strong> but this
      would pay <strong>${esc(p.vendor_name)}</strong>. Only approve if that is actually expected.</div>` : '';
    const otherWarnings = (p.warnings || []).length ? `
      <div class="warn-box">⚠ Automated checks found:
        <ul>${p.warnings.map(w => `<li>${esc(w)}</li>`).join('')}</ul>
      </div>` : '';
    const reasonBox = p.reason ? `
      <div class="reason-box"><span class="label">Agent's reason:</span>${esc(p.reason)}</div>` : '';
    area.innerHTML = `
      <div class="card">
        <p class="amount">${p.amount.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})} TND</p>
        <p class="sub">to ${esc(p.vendor_name)} (${esc(p.vendor_id)})</p>
        <dl class="rows">
          <dt>Invoice</dt><dd>${esc(p.invoice_id)}</dd>
          <dt>Sent by</dt><dd class="${p.sender_mismatch ? 'warn' : ''}">${esc(p.invoice_sender)}</dd>
          <dt>Purchase order</dt><dd>${esc(p.po_number)}</dd>
          <dt>IBAN (registry)</dt><dd>${esc(p.iban)}</dd>
        </dl>
        ${reasonBox}
        ${mismatchBox}
        ${otherWarnings}
        <div class="btns">
          <button class="approve" onclick="decide(true)">Approve</button>
          <button class="reject" onclick="decide(false)">Reject</button>
        </div>
      </div>`;
  }

  const hist = document.getElementById('history');
  if (!data.history.length) {
    hist.innerHTML = '<div class="empty-hist">Nothing decided yet.</div>';
  } else {
    const rows = data.history.map(h => `
      <tr>
        <td>${esc(h.decided_at)}</td>
        <td>${esc(h.invoice_id)}</td>
        <td>${esc(h.vendor_name)}</td>
        <td>${h.amount.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}</td>
        <td><span class="tag ${h.decision}">${h.decision}</span></td>
      </tr>`).join('');
    hist.innerHTML = `<table><thead><tr><th>Time</th><th>Invoice</th><th>Vendor</th><th>Amount</th><th></th>
      </tr></thead><tbody>${rows}</tbody></table>`;
  }
}

let busy = false;
async function decide(approved) {
  if (busy) return;
  busy = true;
  try { await fetch('/api/decide', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({approved})}); } finally { busy = false; }
}

poll();
</script>
</body>
</html>
"""


class WebApprover(Approver):
    """Serves the approval page and blocks `decide()` until the browser answers."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8787, open_browser: bool = True, db_path=None):
        """`db_path`, when given, embeds the live batch dashboard (agent/dashboard.py) on the
        same page -- every invoice's status, what it paid or was flagged for, and any vendor
        email sent about it. Without it, the page is just the approval card and history, as
        before."""
        super().__init__()
        self.host, self.port = host, port
        self.db_path = db_path
        self._pending: dict | None = None
        self._history: list[dict] = []
        self._decision_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self._wait_until_up()
        print(f"\n=== Approval page: http://{self.host}:{self.port}/ ===")
        if open_browser:
            webbrowser.open(f"http://{self.host}:{self.port}/")

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Payment Approval")

        @app.get("/", response_class=HTMLResponse)
        def index():
            return PAGE

        @app.get("/api/ping")
        def ping():
            return {"ok": True}

        @app.get("/api/state")
        def state():
            with self._lock:
                return {"pending": self._pending, "history": list(reversed(self._history))}

        @app.get("/api/dashboard")
        def dashboard():
            if self.db_path is None:
                return {"available": False}
            return query_state(self.db_path)

        @app.post("/api/decide")
        async def decide_endpoint(request: Request):
            body = await request.json()
            approved = bool(body.get("approved"))
            with self._lock:
                if self._pending is None:
                    return JSONResponse({"ok": False, "error": "nothing pending"}, status_code=409)
                record = dict(self._pending)
                record["decision"] = "approved" if approved else "rejected"
                record["decided_at"] = time.strftime("%H:%M:%S")
                self._history.append(record)
                self._pending = None
            self._decision_queue.put(approved)
            return {"ok": True}

        return app

    def _serve(self) -> None:
        config = uvicorn.Config(self._build_app(), host=self.host, port=self.port, log_level="warning")
        uvicorn.Server(config).run()

    def _wait_until_up(self, timeout: float = 5.0) -> None:
        url = f"http://{self.host}:{self.port}/api/ping"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                httpx.get(url, timeout=0.3)
                return
            except httpx.HTTPError:
                time.sleep(0.1)

    def decide(self, proposal: PaymentProposal) -> bool:
        with self._lock:
            self._pending = {
                "invoice_id": proposal.invoice_id,
                "vendor_id": proposal.vendor_id,
                "vendor_name": proposal.vendor_name,
                "iban": proposal.iban,
                "amount": proposal.amount,
                "po_number": proposal.po_number,
                "invoice_sender": proposal.invoice_sender,
                "sender_mismatch": proposal.invoice_sender.strip().lower() != proposal.vendor_name.strip().lower(),
                "reason": proposal.reason,
                "warnings": list(proposal.warnings),
            }
        print(f"\n>>> Waiting in the browser for a decision on {proposal.invoice_id} "
              f"({proposal.amount:,.2f} TND to {proposal.vendor_name}) ...")
        approved = self._decision_queue.get()
        print(f">>> {'Approved' if approved else 'Rejected'} {proposal.invoice_id}.")
        return approved
