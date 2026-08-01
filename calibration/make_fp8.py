#!/usr/bin/env python3
"""fp8-quantize a VQ codebook, matching the milestone recipe: per-group amax scale to the
fp8 max, cast to fp8 (e4m3fn default), dequantize back to fp16. Deploy format for the fast
gather kernel. Same structure/keys as the input; only codebook values change."""
import argparse, torch

FP8 = {"e4m3": (torch.float8_e4m3fn, 448.0), "e5m2": (torch.float8_e5m2, 57344.0)}

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fmt", choices=["e4m3", "e5m2"], default="e4m3")
args = ap.parse_args()
dt, fmax = FP8[args.fmt]

p = torch.load(args.inp, map_location="cpu", weights_only=False)
out = {}
for (l, h), cbs in p["codebooks"].items():
    q = []
    for c in cbs:
        c = c.float()
        s = c.abs().amax().clamp_min(1e-12) / fmax          # per-group scale
        q.append(((c / s).to(dt).to(torch.float32) * s).half())
    out[(l, h)] = q
p2 = dict(p); p2["codebooks"] = out; p2["fp8_fmt"] = args.fmt
torch.save(p2, args.out)
print(f"SAVED {args.out} ({args.fmt}) from {args.inp}", flush=True)
