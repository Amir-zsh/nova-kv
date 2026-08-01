#!/usr/bin/env python3
"""Build a simulated-TurboQuant baseline bundle (SGLANG_SIMQUANT_PATH input).

The served arm applies TurboQuant's rotation + Lloyd-Max roundtrip inside the
BF16 write path: only accuracy transfers, no memory is saved (that is the
point — it isolates the quantizer's accuracy from any serving change). The
shipped `turboquant_k3v3.pt` bundles were built with:

  gpt-oss-20b:   --head-dim 64  --layers 12 --k-bits 3 --v-bits 3
  Qwen3-8B / Llama-3.1-8B:
                 --head-dim 128 --layers 36 --k-bits 3 --v-bits 3

Uses the vendored public TurboQuant implementation in ../vendor (MIT).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))
from turboquant_pytorch.compressors_v3 import MSECompressor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--k-bits", type=int, default=3)
    ap.add_argument("--v-bits", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42, help="TurboQuant's default")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bundle = {
        "head_dim": args.head_dim,
        "k_bits": args.k_bits,
        "v_bits": args.v_bits,
        "seed": args.seed,
        "n_layers": args.layers,
        # Per-layer because MSECompressor's seed is seed + layer_idx*1000 (K)
        # and +500 (V); one rotation per layer, SHARED across kv heads.
        "k_rotation": torch.empty(args.layers, args.head_dim, args.head_dim),
        "v_rotation": torch.empty(args.layers, args.head_dim, args.head_dim),
        "note": "simulated (fake) quant: dequantized BF16 is stored; "
                "no memory saving by design",
    }

    for L in range(args.layers):
        seed_base = args.seed + L * 1000
        kc = MSECompressor(args.head_dim, args.k_bits, seed=seed_base, device="cpu")
        vc = MSECompressor(args.head_dim, args.v_bits, seed=seed_base + 500, device="cpu")
        bundle["k_rotation"][L] = kc.Pi
        bundle["v_rotation"][L] = vc.Pi
        if L == 0:
            # Centroids depend only on (head_dim, bits), so one copy each.
            bundle["k_centroids"] = kc.centroids.clone()
            bundle["v_centroids"] = vc.centroids.clone()

    # Orthogonality is what makes (x @ Pi.T) @ Pi an inverse; a non-orthogonal
    # rotation would silently inflate error at every bit-width.
    for name in ("k_rotation", "v_rotation"):
        Pi = bundle[name][0]
        err = (Pi.T @ Pi - torch.eye(args.head_dim)).abs().max().item()
        assert err < 1e-4, f"{name} not orthogonal: {err:.2e}"

    torch.save(bundle, args.out)
    print(f"wrote {args.out}  (K {bundle['k_centroids'].numel()} levels, "
          f"V {bundle['v_centroids'].numel()} levels)")


if __name__ == "__main__":
    main()
