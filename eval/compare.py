"""Compares isolate on/off x quarantine on/off across the injection and batch scenarios.

    python -m eval.compare [--model scripted|real] [--only ID]

Reuses eval/run_eval.py's scoring: for each of the 4 settings, every injection/batch scenario
runs once and gets scored exactly as `python -m eval.run_eval` would score it. Writes
eval/results/comparison.md: one row per setting with task success, attacks blocked, attacks
flagged by the model, mean steps, and cost per task.

In `--model scripted` (the default), only scenarios with a `.script.json` run (same as
eval-scripted), and quarantine is always a no-op -- there's no real model there to extract
anything from. That makes the scripted comparison thin by construction; `--model real` is
where isolate/quarantine actually show a difference.
"""
import argparse
import statistics
from datetime import datetime
from pathlib import Path

from eval.run_eval import RESULTS_DIR, load_scenarios, run_scenario

SETTINGS = [
    ("isolate=on,  quarantine=on", True, True),
    ("isolate=on,  quarantine=off", True, False),
    ("isolate=off, quarantine=on", False, True),
    ("isolate=off, quarantine=off", False, False),
]


def run_comparison(model_mode: str = "scripted", only: str | None = None) -> list[dict]:
    scenarios = [s for s in load_scenarios(only) if s["category"] in ("injection", "batch")]
    rows = []
    for label, isolate, quarantine in SETTINGS:
        results, skipped = [], []
        for scenario in scenarios:
            r = run_scenario(scenario, model_mode, isolate=isolate, quarantine=quarantine)
            if r is None:
                skipped.append(scenario["id"])
            else:
                results.append(r)

        injection_results = [r for r in results if r["attack_blocked"] is not None]
        steps = [r["steps"] for r in results]
        costs = [r["cost_usd"] for r in results if r["cost_usd"] is not None]
        rows.append({
            "label": label, "isolate": isolate, "quarantine": quarantine,
            "n": len(results), "skipped": skipped, "success": sum(r["success"] for r in results),
            "n_injection": len(injection_results),
            "blocked": sum(r["attack_blocked"] for r in injection_results),
            "flagged": sum(r["model_flagged"] for r in injection_results),
            "mean_steps": statistics.mean(steps) if steps else None,
            "mean_cost": statistics.mean(costs) if costs else None,
            "results": results,
        })
    return rows


def write_comparison_md(rows: list[dict], out_path: Path, model_mode: str, scenario_count: int) -> None:
    lines = ["# Isolation x quarantine comparison", ""]
    lines.append(f"Model mode: `{model_mode}` | scenarios considered: {scenario_count} (injection + batch) | "
                 f"generated: {datetime.now():%Y-%m-%d %H:%M}")
    if model_mode == "scripted":
        lines.append("")
        lines.append("Scripted mode only runs scenarios that have a `.script.json` (see "
                      "`eval/scenarios/`), and quarantine is always a no-op in scripted mode -- "
                      "there is no real model to extract anything from (its own logic is covered "
                      "directly in `tests/test_quarantine.py`). Run with `--model real` for a "
                      "comparison where isolate and quarantine actually change behavior.")
    lines.append("")
    lines.append("| setting | task success | attacks blocked | attacks flagged by model | "
                  "mean steps | cost per task |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        succ_str = f"{row['success']}/{row['n']}" if row["n"] else "0/0"
        if row["n_injection"]:
            blocked_str = f"{row['blocked']}/{row['n_injection']}"
            flagged_str = f"{row['flagged']}/{row['n_injection']}"
        else:
            blocked_str = flagged_str = "n/a"
        steps_str = f"{row['mean_steps']:.1f}" if row["mean_steps"] is not None else "-"
        cost_str = f"${row['mean_cost']:.5f}" if row["mean_cost"] is not None else "-"
        lines.append(f"| {row['label']} | {succ_str} | {blocked_str} | {flagged_str} | "
                      f"{steps_str} | {cost_str} |")

    if rows and rows[0]["skipped"]:
        lines.append("")
        lines.append(f"- Skipped per setting (no `.script.json`): {', '.join(rows[0]['skipped'])}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["scripted", "real"], default="scripted")
    p.add_argument("--only")
    a = p.parse_args()

    if a.model == "real":
        from agent.run import load_dotenv
        load_dotenv()

    scenario_count = len([s for s in load_scenarios(a.only) if s["category"] in ("injection", "batch")])
    rows = run_comparison(a.model, a.only)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "comparison.md"
    write_comparison_md(rows, out_path, a.model, scenario_count)

    for row in rows:
        skip_note = f", {len(row['skipped'])} skipped" if row["skipped"] else ""
        print(f"{row['label']}: {row['success']}/{row['n']} succeeded{skip_note}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
