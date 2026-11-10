# Invoice Payment Agent

CS496 AI Engineering, Project 3 (The Operator).

An agent that reads supplier invoices, checks them against purchase orders and a trusted vendor registry, and pays them through a mock bank after a human approves.

## Setup

```
make setup
make reset
```

## Run

Put your key in a `.env` file (it is git-ignored).

```
echo 'OPENAI_API_KEY=sk-...' > .env
make run
```

`make run` resets the sandbox and processes all pending invoices. You approve each payment in the terminal.
Use `python -m agent.run --invoices INV-2231 INV-2232` to run on a few invoices.
Every run writes a trajectory log to `logs/`.

To approve from a browser page instead of typing `y/N`:

```
python -m agent.run --reset --approver web
```

This opens `http://127.0.0.1:8787/`, which shows each pending payment (invoice, PO, amount,
registry IBAN, the agent's own one-line reason, and any warnings) and an Approve/Reject
button, plus a running history of decisions made so far in the run. It flags when the
invoice's sender doesn't match the vendor the payment would actually go to -- see
`postmortem.md` for why that specific check is on the screen. Use `--web-port` to change the
port and `--no-browser` to not auto-open a tab.

By default, each invoice gets its own isolated conversation (`run_batch` in `agent/loop.py`)
and the main agent never reads an invoice's raw text directly -- it only sees a quarantined
extraction of it (`agent/quarantine.py`). Use `--no-isolate` / `--no-quarantine` to turn
either off for comparison; see `threat_model.md` for why both exist.

## Tests

```
make test
```

## Layout

- `agent/` tools, loop (incl. `run_batch` isolation), stop rule, checkpoint, quarantine
  (`agent/quarantine.py`), web approver
- `sandbox/` schema, seed data, reset, mock bank API
- `eval/` scenarios, trajectory eval (`run_eval.py`), isolate x quarantine comparison
  (`compare.py`)
- `errors.md`, `threat_model.md`, `postmortem.md`

## Results

To fill after the eval runs.

## Contributions

Youssef Hamada, everything.
