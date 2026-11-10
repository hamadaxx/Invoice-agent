"""Run the agent from the command line.

  python -m agent.run                          # all pending invoices, you approve in the terminal
  python -m agent.run --invoices INV-2231 INV-2232
  python -m agent.run --approver auto --reset  # no prompts, fresh sandbox
  python -m agent.run --approver web --reset   # approve from a browser page instead of the terminal
  python -m agent.run --bank-url http://localhost:8001   # use a running bank server
  python -m agent.run --no-isolate             # one shared conversation for the whole batch (old behavior)
  python -m agent.run --no-quarantine          # let the main agent read raw invoice text directly

Each invoice gets its own fresh conversation by default (see run_batch in agent/loop.py), so
text injected into one invoice can't reference or influence another invoice in the same batch.
The main agent never reads raw invoice text directly either, by default: read_invoice hands it
to a quarantined extraction call first (see agent/quarantine.py) and the agent only ever sees
the structured result.

Needs OPENAI_API_KEY in the environment (put it in .env, never commit it).
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from agent.checkpoint import AutoApprover, ConsoleApprover, DenyAll
from agent.loop import run_batch
from agent.model import OpenAIModel
from agent.stop_rule import StopRule
from agent.tools import Toolbox
from agent.web_approver import WebApprover
from sandbox.bank_api import create_app
from sandbox.reset import DEFAULT_DB, reset

STEPS_PER_INVOICE = 16  # read, lookup PO, lookup vendor, check history, schedule/flag(+email), each
# preceded by its own model turn, plus slack for retries -- and, since isolation (the default,
# see run_batch in agent/loop.py) gives each invoice its own conversation, one more model turn
# per invoice just to end that conversation. 8 was calibrated before isolation existed; a real
# 12-invoice run under isolation measured ~10.6 steps/invoice on actual usage (and didn't even
# hit the flag+email path on every invoice), so 8 undercounts it -- this leaves real headroom.


def load_dotenv(path=".env"):
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--invoices", nargs="*", help="invoice IDs to process (default all pending)")
    p.add_argument("--approver", choices=["console", "auto", "deny", "web"], default="console")
    p.add_argument("--web-port", type=int, default=8787, help="port for --approver web")
    p.add_argument("--no-browser", action="store_true", help="with --approver web, don't auto-open a browser tab")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--bank-url", help="URL of a running bank API (default in-process)")
    p.add_argument("--reset", action="store_true", help="reset the sandbox first")
    p.add_argument("--max-steps", type=int, default=None,
                    help=f"default: 10 + {STEPS_PER_INVOICE} * number of invoices to process")
    p.add_argument("--max-payout", type=float, default=100_000)
    p.add_argument("--max-tokens", type=int, default=200_000)
    p.add_argument("--no-isolate", action="store_true",
                    help="process the whole batch in one shared conversation instead of "
                         "one fresh, isolated conversation per invoice (the default)")
    p.add_argument("--no-quarantine", action="store_true",
                    help="let the main agent read raw invoice text directly, instead of only "
                         "a quarantined extraction of it (the default is quarantine on)")
    a = p.parse_args()
    a.quarantine = not a.no_quarantine

    load_dotenv()
    if a.reset or not Path(a.db).exists():
        reset(Path(a.db))

    bank = httpx.Client(base_url=a.bank_url) if a.bank_url else TestClient(create_app(a.db))
    stop = StopRule(max_tokens=a.max_tokens, max_payout=a.max_payout)  # max_steps patched in below
    quarantine_model = OpenAIModel() if a.quarantine else None
    toolbox = Toolbox(a.db, bank, quarantine=a.quarantine, quarantine_model=quarantine_model, stop=stop)
    if a.approver == "web":
        approver = WebApprover(port=a.web_port, open_browser=not a.no_browser, db_path=a.db)
    else:
        approver = {"console": ConsoleApprover, "auto": AutoApprover, "deny": DenyAll}[a.approver]()

    if a.invoices:
        invoice_ids = a.invoices
    else:
        pending = json.loads(toolbox.call("list_invoices", {}))
        invoice_ids = pending["invoice_ids"]

    if a.max_steps is None:
        a.max_steps = 10 + STEPS_PER_INVOICE * len(invoice_ids)
    stop.max_steps = a.max_steps

    result = run_batch(OpenAIModel(), toolbox, approver, stop, invoice_ids,
                        isolate=not a.no_isolate, verbose=True)

    print("\n=== RESULT ===")
    print(result.final_text)
    if result.unresolved:
        print(f"unresolved: {', '.join(result.unresolved)}")
    print(f"\nstop: {result.stop_reason} (limit {a.max_steps}) | steps: {result.steps} | tokens: {result.tokens} "
          f"| paid: {result.payout:,.2f} TND | {result.seconds}s "
          f"| isolate={not a.no_isolate} quarantine={a.quarantine}")

    Path("logs").mkdir(exist_ok=True)
    log_path = Path("logs") / f"run_{datetime.now():%Y%m%d_%H%M%S}.json"
    log_path.write_text(json.dumps(result.__dict__, indent=2, default=str))
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
