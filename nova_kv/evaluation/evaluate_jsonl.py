#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from .score import contains_answer, exact_match

p=argparse.ArgumentParser(); p.add_argument("input", type=Path); p.add_argument("--metric", choices=["exact","contains"], default="exact"); a=p.parse_args()
score = exact_match if a.metric == "exact" else contains_answer
rows=[json.loads(line) for line in a.input.open()]
values=[score(r["prediction"], r["answers"]) for r in rows]
print(json.dumps({"file":str(a.input),"metric":a.metric,"n":len(values),"score":sum(values)/len(values) if values else 0.0},indent=2))
