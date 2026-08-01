"""Group-VQ codebook bundle loader + torch-side encode helpers for the vq2
K quant tier of the unified mixed HP+int2 pool.

Bundle schema (produced by the ``calibration/fit_vq_codebook.py`` trainer):

    forward   [L, H, D, D]   residual map: r = (k - mean) @ forward  (per head)
    inverse   [L, H, D, D]   recon map:    k_hat = r_hat @ inverse + mean
    mean      [L, H, D]
    codebooks {(l, h): list of NG tensors [K, G]}   (fp16 or fp8_e4m3fn)
    bounds    [(start, end, bits)] * NG   -- flat allocation, contiguous groups
    pertoken_norm bool -- per-token RMS scale on r before lookup

Engine storage convention: BOTH tiers hold the residual ``r`` (HP as bf16
rows, quant as VQ indices + per-token RMS scale). Queries are mapped with
``q @ inverse.T`` so ``q_m . r = q . (k - mean)``; the ``-q . mean`` term is
constant across keys for a given query, hence softmax-invariant, and it is
identical for both tiers. Models with a learned attention sink are the
exception: the sink is an extra denominator logit, not a real key, so it must
receive the same per-query ``-q . mean`` shift. Prefill adjusts it explicitly;
decode fuses the adjustment into unified stage 2.

The decode kernel reconstructs codewords from a packed int32 (4x fp8-e5m2
bytes, little-endian => byte i is coord i of the group; e5m2 because sm80
Triton only bitcasts fp8e5). Encode assigns against the *fp8-dequantized*
centroids so the encoder is a true nearest-neighbor for what the decoder
actually reconstructs.
"""

from __future__ import annotations

import functools
import logging
import os

import triton
import triton.language as tl
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def resolve_vq_fp8_fmt() -> str:
    """Pick the fp8 format for centroid storage: "e5m2" or "e4m3".

    SGLANG_VQ_FP8_FMT=e5m2|e4m3|auto; DEFAULT e5m2, matching the A100 record.

    e5m2 is the default so H100 runs stay directly comparable to the A100
    accuracy results. That is not a free choice on A100 -- Triton gates fp8e4nv
    at capability >= 8.9 (backends/nvidia/compiler.py:188) and A100 is sm80, so
    e4m3 could not be used there at all. Every recorded baseline is therefore
    e5m2 by construction.

    e4m3 is available and strictly better where the hardware allows it: on the
    shipped gpqacc64k bundle its rel L2 centroid error is 2.643% vs e5m2's
    5.668%, and with centroid |max| = 11.2 against e4m3's 448 ceiling, 0 of
    1.57M values saturate -- the narrower exponent range costs nothing here.
    Downstream, though, an A/B measured it NEUTRAL (NIAH +0.22 = noise, 0.00
    speed), so it buys comparability-breaking, not accuracy. Opt in with
    SGLANG_VQ_FP8_FMT=e4m3 (or "auto" to take it wherever sm >= 8.9).

    The resolved format is logged at load ("fp8=%s"), so any run's choice is
    recoverable from its server log. Forcing e4m3 below sm89 raises here rather
    than failing later at kernel compile.
    """
    import os

    want = os.environ.get("SGLANG_VQ_FP8_FMT", "e5m2").lower()
    if want not in ("auto", "e5m2", "e4m3"):
        raise ValueError(f"SGLANG_VQ_FP8_FMT must be auto|e5m2|e4m3, got {want!r}")
    if want == "auto":
        cap = torch.cuda.get_device_capability()
        return "e4m3" if cap >= (8, 9) else "e5m2"
    if want == "e4m3" and torch.cuda.get_device_capability() < (8, 9):
        raise ValueError(
            "SGLANG_VQ_FP8_FMT=e4m3 requires compute capability >= 8.9 "
            f"(this device is sm{''.join(map(str, torch.cuda.get_device_capability()))})"
        )
    return want


@dataclass
class VQCodebook:
    forward: torch.Tensor    # [L, H, D, D] bf16 -- k-side map (applied to k - mean)
    q_map: torch.Tensor      # [L, H, D, D] bf16 -- inverse.transpose(-1,-2), q-side map
    mean: torch.Tensor       # [L, H, D] bf16
    cb16: torch.Tensor       # [L, H, NG, K, G] fp16 -- fp8-dequantized centroids
    cb_sq: torch.Tensor      # [L, H, NG, K] fp32 -- 0.5 * ||c||^2 per centroid
    cb_packed: torch.Tensor  # [L, H, NG, K] int32 -- fp8 bytes packed little-endian
    num_groups: int          # NG
    codebook_size: int       # K
    group_dim: int           # G
    pertoken_norm: bool
    source_fp8_fmt: str | None      # fp8 format the bundle declares, if any
    centroid_resnap_rel_rmse: float  # error added by re-snapping at load
    fp8_fmt: str = "e5m2"    # "e5m2" | "e4m3" -- decode kernel must bitcast to match

    def layer(self, idx: int):
        return (
            self.forward[idx],
            self.q_map[idx],
            self.mean[idx],
            self.cb16[idx],
            self.cb_sq[idx],
            self.cb_packed[idx],
        )


def load_vq_codebook(
    path: str,
    *,
    layer_num: int,
    start_layer: int,
    head_num: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    head_start: int = 0,
) -> VQCodebook:
    """``head_num`` is the pool's LOCAL head count; under tensor parallelism
    the bundle holds all global KV heads and ``head_start`` selects this
    rank's contiguous slice (Megatron-style sharding: rank r owns heads
    [r*local, (r+1)*local))."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    F = blob["forward"]
    inv = blob["inverse"]
    mean = blob["mean"]
    bounds = blob["bounds"]
    ptn = bool(blob.get("pertoken_norm", False))
    source_fp8_fmt = blob.get("fp8_fmt")

    L_total, H_total, D, D2 = F.shape
    assert D == D2 == head_dim, f"codebook head_dim {D} != pool head_dim {head_dim}"
    head_end = head_start + head_num
    assert head_end <= H_total and (head_start == 0 or H_total % head_num == 0), (
        f"codebook has {H_total} heads, rank wants [{head_start}, {head_end}) "
        f"— TP size must divide the KV head count (no replication support)"
    )
    H = head_num
    F = F[:, head_start:head_end]
    inv = inv[:, head_start:head_end]
    mean = mean[:, head_start:head_end]
    end_layer = start_layer + layer_num
    assert end_layer <= L_total, (
        f"codebook has {L_total} layers, pool wants [{start_layer}, {end_layer})"
    )

    # Flat contiguous groups only (the stratified permutation is folded into
    # ``forward``); the decode kernel assumes uniform (K, G) across groups.
    starts = [s for (s, _e, _b) in bounds]
    ends = [e for (_s, e, _b) in bounds]
    NG = len(bounds)
    G = ends[0] - starts[0]
    assert starts == list(range(0, D, G)) and all(
        e - s == G for s, e in zip(starts, ends)
    ), f"vq2 requires uniform contiguous groups, got bounds={bounds[:4]}..."

    cbs = blob["codebooks"]
    K = cbs[(0, 0)][0].shape[0]
    cb = torch.empty((layer_num, H, NG, K, G), dtype=torch.float16)
    for l in range(layer_num):
        for h in range(H):
            entry = cbs[(start_layer + l, head_start + h)]
            for g in range(NG):
                c = entry[g]
                assert c.shape == (K, G), f"codebook ({l},{h},{g}) shape {c.shape}"
                cb[l, h, g] = c.to(torch.float16)

    # Snap centroids to their fp8 representation, then build both the packed
    # int32 decode view and the matching fp16 encode view from the SAME bytes.
    #
    # e5m2 has 2 mantissa bits; e4m3 has 3. Triton only admits fp8e4nv at
    # compute capability >= 8.9 (backends/nvidia/compiler.py gates it), so
    # sm80/A100 must stay on e5m2 -- that is what the A100 record was measured
    # with. On sm89+/sm90 e4m3 roughly halves the centroid representation error
    # (measured 5.51% -> 2.66% rel L2 on the gpqacc64k bundle).
    #
    # The loader's snap and the decode kernel's bitcast MUST agree: the encoder
    # assigns against the centroids the decoder reconstructs. fp8_fmt is carried
    # on VQCodebook and threaded to the kernel as a constexpr for exactly that.
    fmt = resolve_vq_fp8_fmt()
    cb_fp8 = cb.to(torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2)
    centroid_resnap_rel_rmse = float(
        (
            (cb_fp8.to(torch.float32) - cb.to(torch.float32)).pow(2).sum()
            / cb.to(torch.float32).pow(2).sum().clamp_min(1e-30)
        )
        .sqrt()
        .item()
    )
    if source_fp8_fmt is not None and centroid_resnap_rel_rmse > 0:
        logger.warning(
            "vq2: codebook %s declares fp8_fmt=%s but stores dequantized "
            "centroids; runtime %s packing re-snaps them (relative RMSE %.6f). "
            "Use a raw trained bundle to measure single-snap fidelity.",
            path,
            source_fp8_fmt,
            fmt.upper(),
            centroid_resnap_rel_rmse,
        )
    assert G == 4, f"packed-int32 codewords require G == 4, got G={G}"
    cb_packed = (
        cb_fp8.contiguous().view(torch.int32).squeeze(-1).contiguous()
    )  # [L, H, NG, K]
    cb16 = cb_fp8.to(torch.float16)
    cb_sq = 0.5 * cb16.to(torch.float32).pow(2).sum(-1)  # [L, H, NG, K]

    Fl = F[start_layer:end_layer].to(dtype)
    q_map = inv[start_layer:end_layer].transpose(-1, -2).to(dtype).contiguous()
    mean_l = mean[start_layer:end_layer].to(dtype)

    out = VQCodebook(
        forward=Fl.to(device),
        q_map=q_map.to(device),
        mean=mean_l.to(device),
        cb16=cb16.to(device),
        cb_sq=cb_sq.to(device),
        cb_packed=cb_packed.to(device),
        num_groups=NG,
        codebook_size=K,
        group_dim=G,
        pertoken_norm=ptn,
        source_fp8_fmt=source_fp8_fmt,
        centroid_resnap_rel_rmse=centroid_resnap_rel_rmse,
        fp8_fmt=fmt,
    )
    logger.info(
        "vq2: loaded codebook %s (layers=%d heads=[%d,%d)/%d NG=%d K=%d "
        "G=%d ptn=%s fp8=%s source_fp8=%s resnap_rel_rmse=%.6f)",
        path,
        layer_num,
        head_start,
        head_end,
        H_total,
        NG,
        K,
        G,
        ptn,
        fmt,
        source_fp8_fmt,
        centroid_resnap_rel_rmse,
    )
    return out


def vq_map_k(
    k: torch.Tensor, forward: torch.Tensor, mean: torch.Tensor
) -> torch.Tensor:
    """r = (k - mean) @ forward, per head. k: [T, H, D] (any float dtype).
    Contiguous output: pool writers view the result as [-1, row_dim].

    Dispatches here rather than at the call sites (unlike QMAP) because there
    are two of them -- the decode aging flush and the prefill VQ write -- and a
    gate in each is a drift hazard for no gain.
    """
    from sglang.srt.environ import envs as _envs

    if _envs.SGLANG_VQ_OPT_KMAP.get():
        return vq_map_k_fused(k, forward, mean)
    kd = k.to(forward.dtype)
    return torch.einsum(
        "thd,hde->the", kd - mean.unsqueeze(0), forward
    ).contiguous()


@triton.jit
def _vq_qmap_kernel(
    Q, QMAP, MEAN, OUT,
    n_rows, QH,
    D: tl.constexpr, GRP: tl.constexpr, BLOCK_T: tl.constexpr,
    HAS_MEAN: tl.constexpr, PREC: tl.constexpr,
):
    """out[t, h*GRP+g, :] = (q[t, h*GRP+g, :] - mean[h]) @ q_map[h]

    One program handles BLOCK_T rows of a single KV head, reading and writing
    q in place-strided form. This replaces the view/permute/reshape/bmm/
    permute/reshape/contiguous chain with a single kernel: the profile showed
    the copies, not the GEMM, were the bulk of the cost.

    HAS_MEAN + GRP=1 makes the same kernel serve the K map, whose only
    differences are the centering term and the absent GQA group.
    """
    pid = tl.program_id(0)
    h = tl.program_id(1)
    offs_i = pid * BLOCK_T + tl.arange(0, BLOCK_T)      # index into T*GRP
    mask_i = offs_i < n_rows
    t = offs_i // GRP
    g = offs_i % GRP
    qh = h * GRP + g
    offs_d = tl.arange(0, D)

    base = t[:, None] * (QH * D) + qh[:, None] * D + offs_d[None, :]
    qv = tl.load(Q + base, mask=mask_i[:, None], other=0.0)
    if HAS_MEAN:
        qv = qv - tl.load(MEAN + h * D + offs_d)[None, :]
    m = tl.load(QMAP + h * D * D + offs_d[:, None] * D + offs_d[None, :])
    # PREC="ieee" on fp32 inputs: tl.dot would otherwise default to TF32 and
    # make the flag a silent numerics change (rel ~8e-4). fp16/bf16 are
    # bit-identical to the torch path either way.
    o = tl.dot(qv, m, input_precision=PREC)              # fp32 accumulate
    tl.store(OUT + base, o.to(qv.dtype), mask=mask_i[:, None])


def vq_map_q_fused(q: torch.Tensor, q_map: torch.Tensor) -> torch.Tensor:
    """Fused equivalent of vq_map_q (SGLANG_VQ_OPT_QMAP)."""
    T, QH, D = q.shape
    H = q_map.shape[0]
    GRP = QH // H
    qc = q.to(q_map.dtype)
    if not qc.is_contiguous():
        qc = qc.contiguous()
    out = torch.empty_like(qc)
    n_rows = T * GRP
    BLOCK_T = 16 if n_rows >= 16 else 16   # tl.dot needs >= 16 rows; mask covers the tail
    grid = (triton.cdiv(n_rows, BLOCK_T), H)
    _vq_qmap_kernel[grid](
        qc, q_map, qc, out, n_rows, QH,
        D=D, GRP=GRP, BLOCK_T=BLOCK_T, HAS_MEAN=False,
        PREC="ieee" if qc.dtype == torch.float32 else "tf32",
        num_warps=4, num_stages=2,
    )
    return out


def vq_map_k_fused(
    k: torch.Tensor, forward: torch.Tensor, mean: torch.Tensor
) -> torch.Tensor:
    """Fused equivalent of vq_map_k (SGLANG_VQ_OPT_KMAP).

    The einsum form runs sub/bmm/contiguous as three kernels over the same
    small tensor; measured under CUDA-graph replay (the serving condition,
    where launch cost is already amortised) that is 2.5-2.9x the fused time
    across n=8..512.
    """
    T, H, D = k.shape
    mean = mean.to(forward.dtype)
    kc = k.to(forward.dtype)
    if not kc.is_contiguous():
        kc = kc.contiguous()
    out = torch.empty_like(kc)
    BLOCK_T = 16                     # tl.dot needs >= 16 rows; mask covers the tail
    grid = (triton.cdiv(T, BLOCK_T), H)
    _vq_qmap_kernel[grid](
        kc, forward, mean, out, T, H,
        D=D, GRP=1, BLOCK_T=BLOCK_T, HAS_MEAN=True,
        PREC="ieee" if kc.dtype == torch.float32 else "tf32",
        num_warps=4, num_stages=2,
    )
    return out


def vq_map_q(q: torch.Tensor, q_map: torch.Tensor) -> torch.Tensor:
    """q_m = q @ inverse.T per KV head, broadcast over the GQA group.

    q: [T, QH, D]; q_map: [H, D, D] with QH % H == 0. Batched-GEMM (bmm) form:
    ~1.5x faster than the einsum at decode (small T) and bit-identical; brings the
    per-head query map to OSCAR's shared-rotation GEMM efficiency.
    """
    T, QH, D = q.shape
    H = q_map.shape[0]
    grp = QH // H
    qd = (
        q.to(q_map.dtype)
        .view(T, H, grp, D)
        .permute(1, 0, 2, 3)
        .reshape(H, T * grp, D)
    )
    out = torch.bmm(qd, q_map).view(H, T, grp, D).permute(1, 0, 2, 3)
    return out.reshape(T, QH, D).contiguous()


def vq_encode(
    r: torch.Tensor,
    cb16: torch.Tensor,
    cb_sq: torch.Tensor,
    *,
    pertoken_norm: bool,
    token_chunk: int = 2048,
):
    """Assign residual rows to nearest centroids.

    r: [T, H, D] (bf16/fp16/fp32); cb16: [H, NG, K, G]; cb_sq: [H, NG, K].
    Returns (idx uint8 [T, H, NG], scale fp32 [T, H]).  (K=256 -> indices fit uint8.)

    Nearest neighbor via argmax(<c, x> - 0.5 ||c||^2); token-chunked so the
    [T, H, NG, K] score tensor stays bounded.
    """
    T, H, D = r.shape
    NG, K, G = cb16.shape[1], cb16.shape[2], cb16.shape[3]
    rf = r.to(torch.float32)
    if pertoken_norm:
        scale = rf.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        rn = (rf / scale).to(torch.float16)
        scale = scale.squeeze(-1)
    else:
        scale = torch.ones((T, H), dtype=torch.float32, device=r.device)
        rn = rf.to(torch.float16)
    rn = rn.view(T, H, NG, G)
    idx = torch.empty((T, H, NG), dtype=torch.uint8, device=r.device)
    for t0 in range(0, T, token_chunk):
        t1 = min(t0 + token_chunk, T)
        scores = torch.einsum("thgc,hgkc->thgk", rn[t0:t1], cb16).to(torch.float32)
        scores -= cb_sq.unsqueeze(0)
        idx[t0:t1] = scores.argmax(-1).to(torch.uint8)
    return idx, scale


@triton.jit
def _vq_encode_kernel(
    R, CB, CBSQ, SCALE, IDX,
    n_tok,
    stride_r_l, stride_r_n, stride_r_h,
    stride_s_l, stride_s_n,
    stride_i_l, stride_i_n, stride_i_h,
    H,
    NG: tl.constexpr, K: tl.constexpr, G: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Nearest-centroid assign, fused -- never materialises the score tensor.

    grid = (L*H*NG, cdiv(n_tok, BLOCK_N)). Each program owns one
    (layer, head, group) and BLOCK_N tokens: it loads that group's [K, G]
    centroids once, scores the rows against them in registers, and writes the
    argmax straight to the uint8 index arena. The torch path instead builds an
    [L, n, H, NG, K] fp32 tensor (~76 MB at n=64) and makes three passes over
    it to produce ~0.5 MB of indices.

    The per-token RMS scale is precomputed by the caller (a cheap reduction)
    so it is not recomputed redundantly in all NG group-programs.
    """
    lhg = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = lhg % NG
    h = (lhg // NG) % H
    l = lhg // (NG * H)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_tok
    offs_g = tl.arange(0, G)
    offs_k = tl.arange(0, K)

    x = tl.load(
        R + l * stride_r_l + offs_n[:, None] * stride_r_n + h * stride_r_h
        + (g * G + offs_g)[None, :],
        mask=mask_n[:, None], other=0.0,
    ).to(tl.float32)
    sc = tl.load(SCALE + l * stride_s_l + offs_n * stride_s_n + h,
                 mask=mask_n, other=1.0).to(tl.float32)
    x = x / sc[:, None]

    cbp = (CB + ((l * H + h) * NG + g) * (K * G)
           + offs_k[:, None] * G + offs_g[None, :])
    c = tl.load(cbp).to(tl.float32)                                  # [K, G]
    csq = tl.load(CBSQ + ((l * H + h) * NG + g) * K + offs_k).to(tl.float32)

    scores = tl.sum(x[:, None, :] * c[None, :, :], axis=2) - csq[None, :]
    best = tl.argmax(scores, axis=1)
    tl.store(
        IDX + l * stride_i_l + offs_n[:, None] * stride_i_n
        + h * stride_i_h + g,
        best[:, None].to(tl.uint8), mask=mask_n[:, None],
    )


def vq_encode_fused(r, cb16, cb_sq, *, pertoken_norm: bool):
    """Fused flush encode (SGLANG_VQ_OPT_FLUSH).

    r: [L, n, H, D]; cb16: [L, H, NG, K, G]; cb_sq: [L, H, NG, K].
    Returns (idx uint8 [L, n, H, NG], scale fp32 [L, n, H]) -- same contract as
    the torch path in vq_flush_k.
    """
    L, n, H, D = r.shape
    NG, K, G = cb16.shape[2], cb16.shape[3], cb16.shape[4]
    rc = r.contiguous()
    rf = rc.to(torch.float32)
    if pertoken_norm:
        scale = rf.pow(2).mean(-1).sqrt().clamp_min(1e-8)             # [L, n, H]
    else:
        scale = torch.ones((L, n, H), dtype=torch.float32, device=r.device)
    idx = torch.empty((L, n, H, NG), dtype=torch.uint8, device=r.device)
    cbc, cbsqc = cb16.contiguous(), cb_sq.contiguous()
    BLOCK_N = 16
    grid = (L * H * NG, triton.cdiv(n, BLOCK_N))
    _vq_encode_kernel[grid](
        rf, cbc, cbsqc, scale, idx,
        n,
        rf.stride(0), rf.stride(1), rf.stride(2),
        scale.stride(0), scale.stride(1),
        idx.stride(0), idx.stride(1), idx.stride(2),
        H, NG=NG, K=K, G=G, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return idx, scale


def vq_encode_single(r, cb16_l, cb_sq_l, *, pertoken_norm):
    """Single-layer adapter for :func:`vq_encode_fused`.

    Drop-in for ``vq_encode`` at the prefill/extend write sites, which work one
    layer at a time: r [T, H, D], cb16_l [H, NG, K, G], cb_sq_l [H, NG, K].
    The unfused path materialises a [chunk, H, NG, K] fp32 score tensor (537 MB
    at the 2048-token chunk) and makes three passes over it; the fused kernel
    keeps the reduction in registers.
    """
    idx, scale = vq_encode_fused(
        r.unsqueeze(0),
        cb16_l.unsqueeze(0),
        cb_sq_l.unsqueeze(0),
        pertoken_norm=pertoken_norm,
    )
    return idx.squeeze(0), scale.squeeze(0)


def vq_dequant(
    idx: torch.Tensor, scale: torch.Tensor, cb16: torch.Tensor
) -> torch.Tensor:
    """Reconstruct residual rows: idx [T, H, NG] uint8, scale [T, H],
    cb16 [H, NG, K, G] -> r_hat [T, H, NG*G] fp32."""
    T, H, NG = idx.shape
    G = cb16.shape[-1]
    h_ids = torch.arange(H, device=idx.device).view(1, H, 1)
    g_ids = torch.arange(NG, device=idx.device).view(1, 1, NG)
    cw = cb16[h_ids, g_ids, idx.long()]  # [T, H, NG, G]
    r_hat = cw.to(torch.float32).view(T, H, NG * G)
    return r_hat * scale.to(torch.float32).unsqueeze(-1)
