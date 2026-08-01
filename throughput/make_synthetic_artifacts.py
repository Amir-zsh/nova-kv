#!/usr/bin/env python3
"""Generate throughput-only OSCAR rotations and a vq2 codebook for any model.

The quant arms need per-model calibrated artifacts, and requiring them is the
main obstacle to benchmarking a new model. Throughput does not depend on their
*contents* -- the decode kernel issues the same gathers and the same GEMM shapes
whatever the numbers are -- so for speed measurements we synthesize them from the
model's (layers, kv_heads, head_dim) alone.

THESE ARTIFACTS ARE THROUGHPUT-VALID AND ACCURACY-INVALID. Random centroids
reconstruct garbage; generated text is meaningless. That is harmless here only
because ignore_eos fixes the token count regardless. Never read an accuracy
number off a run that used them -- accuracy work must use the calibrated bundles
under artifacts/.

Usage:
  python throughput/make_synthetic_artifacts.py --model Qwen/Qwen3-8B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capacity import model_geometry  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CODEBOOK_SIZE = 256      # K: keeps VQ indices in uint8 (2.0 b/coord at G=4)
GROUP_DIM = 4            # G: vq_codebook.py asserts G == 4 (packed-int32 codewords)


def random_orthogonal(n: int, gen: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float32))
    # QR's sign convention is arbitrary; fixing it keeps Q a proper rotation.
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def write_rotations(out: Path, geom: dict, gen: torch.Generator) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, dim in (("k_rotation_qqt_r_h_pbr.pt", geom["head_dim"]),
                      ("v_rotation_sst_r_h_pbr.pt", geom["v_head_dim"])):
        layers = {l: {"rotation": random_orthogonal(dim, gen)}
                  for l in range(geom["layers"])}
        torch.save({"layers": layers}, out / name)
        print(f"  wrote {out / name}  [{geom['layers']} x {dim}x{dim}]")


def write_vq_bundle(path: Path, geom: dict, gen: torch.Generator) -> None:
    L, H, D = geom["layers"], geom["kv_heads"], geom["head_dim"]
    ng = D // GROUP_DIM

    fwd = torch.empty(L, H, D, D, dtype=torch.float32)
    inv = torch.empty(L, H, D, D, dtype=torch.float32)
    for l in range(L):
        for h in range(H):
            q = random_orthogonal(D, gen)
            fwd[l, h] = q
            inv[l, h] = q.T          # exact inverse, since orthogonal

    # Centroids live in the space of per-token-normalised, rotated K, where a
    # unit-norm row has per-coordinate RMS 1/sqrt(D). Matching that scale keeps
    # reconstruction finite -- wildly scaled centroids risk NaN/denormals, which
    # would change timing rather than merely degrading quality.
    scale = D ** -0.5
    codebooks = {
        (l, h): [torch.randn(CODEBOOK_SIZE, GROUP_DIM, generator=gen) * scale
                 for _ in range(ng)]
        for l in range(L) for h in range(H)
    }

    torch.save({
        "forward": fwd,
        "inverse": inv,
        "mean": torch.zeros(L, H, D, dtype=torch.float32),
        "bounds": [(g * GROUP_DIM, (g + 1) * GROUP_DIM, None) for g in range(ng)],
        "codebooks": codebooks,
        "pertoken_norm": True,
        "synthetic": True,          # tripwire: accuracy tooling can refuse these
    }, path)
    print(f"  wrote {path}  [L={L} H={H} D={D} NG={ng} K={CODEBOOK_SIZE} G={GROUP_DIM}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-root", type=Path,
                    default=REPO / "artifacts/synthetic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    geom = model_geometry(a.model)
    # Tag by geometry as well as name so a bundle can never be silently paired
    # with a model it does not fit.
    tag = (f"{a.model.split('/')[-1]}_L{geom['layers']}"
           f"H{geom['kv_heads']}D{geom['head_dim']}")
    out = a.out_root / tag
    bundle = out / "codebook.pt"

    if bundle.exists() and not a.force:
        print(f"{tag}: already present at {out} (use --force to regenerate)")
        return 0

    print(f"{tag}: generating synthetic THROUGHPUT-ONLY artifacts")
    gen = torch.Generator().manual_seed(a.seed)
    # Flat artifact-dir layout: rotations and codebook side by side, exactly
    # what scripts/serve_method.sh expects an ARTIFACT_DIR to contain.
    write_rotations(out, geom, gen)
    write_vq_bundle(bundle, geom, gen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
