#!/usr/bin/env python3
"""Print or execute the deterministic paper matrix."""
import argparse, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-key", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--method", required=True,
                   help="method served by the already-running server")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    cfg = json.loads((ROOT / "nova_kv/protocols/paper.json").read_text())
    task, model = cfg["tasks"][a.task], cfg["models"][a.model_key]
    allowed = task.get("methods", cfg["methods"])
    if a.method not in allowed:
        p.error(f"method {a.method!r} is not configured for {a.task}; choose from {allowed}")
    for seed in task["seeds"]:
        out = ROOT / "results" / a.task / a.model_key / a.method / f"seed-{seed}.jsonl"
        cmd = ["python3", str(ROOT / "scripts/run_cell.py"), "--task", a.task,
               "--model", model, "--method", a.method, "--seed", str(seed), "--input", a.input,
               "--output", str(out)]
        print(" ".join(cmd), flush=True)
        if a.execute: subprocess.run(cmd, check=True)

if __name__ == "__main__": main()
