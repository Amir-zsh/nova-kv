#!/usr/bin/env bash
# Serve one (model, method) arm with the vendored engine in ../python.
#
#   scripts/serve_method.sh MODEL METHOD ARTIFACT_DIR [extra sglang args...]
#
# METHOD:
#   bf16        stock BF16 KV cache (baseline)
#   nova        the paper method: mixed-KV windows, calibrated rotations,
#               INT2 V tier, group-VQ K tier (ARTIFACT_DIR/codebook.pt)
#   oscar       nova without the VQ K tier (scalar INT2 K; `int2` is an alias)
#   quarot      baseline: plain per-token INT2, segmented Hadamard, no mixed-KV
#               band, no calibrated rotation (real INT2 memory footprint)
#   turboquant  baseline: quarot + Lloyd-Max quantization levels
#   turboquant_k3v3
#               baseline: simulated 3-bit TurboQuant on the BF16 write path
#               (ARTIFACT_DIR/turboquant_k3v3.pt; no memory saving)
#
# ARTIFACT_DIR must contain, for nova/int2:
#   k_rotation_qqt_r_h_pbr.pt  v_rotation_sst_r_h_pbr.pt  [codebook.pt]
#
# Everything after ARTIFACT_DIR is passed to sglang.launch_server verbatim
# (--port, --context-length, --mem-fraction-static, --disable-radix-cache, ...).
#
# Env knobs (all optional):
#   PREFILL_BACKEND=fa3|triton   fa3 needs Hopper+; triton works everywhere
#   TP=1                         tensor parallel size
#   KV_SPLITS                    decode split-K (default: 48 for nova, 8 otherwise)
#   MAX_REQS=16                  --max-running-requests
#   MAX_TOKENS                   pin --max-total-tokens (default: derive from
#                                mem-fraction; pinning erases the capacity
#                                advantage the quantized pool exists to buy)
#   RADIX_CACHE=0|1              prefix cache (default off: eval rows are unique;
#                                prefix-reuse benchmarks set 1)
#   QUANT_GROUP_SIZE             min-max scale group (default: 128; hybrid-SWA
#                                models such as gpt-oss require per-head scales
#                                and default to 0 = omit the flag)
#   SGLANG_VQ2_CUDA=1            opt into the CUDA decode stage-1 kernel
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 MODEL METHOD ARTIFACT_DIR [sglang serve arguments...]" >&2
  exit 2
fi

model=$1
method=$2
artifact_dir=$3
shift 3

repo="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$repo/python:${PYTHONPATH:-}"

unset SGLANG_ENABLE_MIXED_KV_WINDOWS SGLANG_VQ_CODEBOOK_PATH SGLANG_SIMQUANT_PATH
unset SGLANG_OSCAR_K_ROTATION_PATH SGLANG_OSCAR_V_ROTATION_PATH

# Long-context evals exceed some models' config-derived limit (RoPE
# extrapolation, matching the HF-harness behaviour at 64K+).
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# expandable_segments lets the allocator return freed segments to the driver,
# which the capacity-sized quant pools rely on. Incompatible with the custom
# all-reduce's cudaIpc export under TP (vllm#42609), so TP>1 drops it.
if [[ "${TP:-1}" == "1" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi

# Hybrid-SWA models (gpt-oss) need per-head scales (group flag omitted) and the
# runtime V-rotation path: the weight-folding helper assumes a dense layer
# range, but hybrid rotation bundles cover the full-attention layers only.
if [[ "$model" == *gpt-oss* ]]; then
  qgs_default=0
  absorb_v_default=0
  hadamard_default=64     # segmented Hadamard order = head_dim
else
  qgs_default=128
  absorb_v_default=1
  hadamard_default=128
fi

args=(
  --model-path "$model"
  --trust-remote-code
  --host 127.0.0.1
  --prefill-attention-backend "${PREFILL_BACKEND:-fa3}"
  --decode-attention-backend triton
  --sampling-backend pytorch
  --tp-size "${TP:-1}"
  --max-running-requests "${MAX_REQS:-16}"
)
[[ -n "${MAX_TOKENS:-}" ]] && args+=(--max-total-tokens "$MAX_TOKENS")
[[ "${RADIX_CACHE:-0}" == "1" ]] || args+=(--disable-radix-cache)

kv_splits_default=8

case "$method" in
  bf16)
    ;;
  turboquant_k3v3)
    bundle="$artifact_dir/turboquant_k3v3.pt"
    if [[ ! -f $bundle ]]; then
      echo "missing simulated-baseline bundle: $bundle" >&2
      exit 2
    fi
    export SGLANG_SIMQUANT_PATH="$bundle"
    ;;
  quarot|turboquant)
    # Plain per-token INT2 pool: no mixed-KV band, no calibrated rotation.
    # The pool always applies a segmented Hadamard of order HADAMARD_ORDER;
    # order = head_dim is the QuaRot rotation. No percentile clip.
    export HADAMARD_ORDER="${HADAMARD_ORDER:-$hadamard_default}"
    export SGLANG_OSCAR_K_CLIP_RATIO="${SGLANG_OSCAR_K_CLIP_RATIO:-1.0}"
    export SGLANG_OSCAR_V_CLIP_RATIO="${SGLANG_OSCAR_V_CLIP_RATIO:-1.0}"
    [[ "$method" == turboquant ]] && export SGLANG_LLOYD_MAX=1
    export SGLANG_SWA_KEEP_PREFIX_TAIL="${SGLANG_SWA_KEEP_PREFIX_TAIL:-1}"
    args+=(--kv-cache-dtype int2)
    qgs="${QUANT_GROUP_SIZE:-$qgs_default}"
    [[ "$qgs" != "0" ]] && args+=(--kv-cache-quant-group-size "$qgs")
    ;;
  nova|oscar|int2)
    for f in k_rotation_qqt_r_h_pbr.pt v_rotation_sst_r_h_pbr.pt; do
      if [[ ! -f "$artifact_dir/$f" ]]; then
        echo "missing rotation bundle: $artifact_dir/$f" >&2
        exit 2
      fi
    done
    export SGLANG_ENABLE_MIXED_KV_WINDOWS=1
    export SGLANG_OSCAR_K_ROTATION_PATH="$artifact_dir/k_rotation_qqt_r_h_pbr.pt"
    export SGLANG_OSCAR_V_ROTATION_PATH="$artifact_dir/v_rotation_sst_r_h_pbr.pt"
    export SGLANG_OSCAR_K_CLIP_RATIO="${SGLANG_OSCAR_K_CLIP_RATIO:-0.96}"
    export SGLANG_OSCAR_V_CLIP_RATIO="${SGLANG_OSCAR_V_CLIP_RATIO:-0.92}"
    export SGLANG_OSCAR_ABSORB_V_ROTATION="${ABSORB_V_ROT:-$absorb_v_default}"
    export SGLANG_MIXED_KV_PREFIX_TOKENS="${PREFIX_TOKENS:-64}"
    export SGLANG_MIXED_KV_RECENT_TOKENS="${RECENT_TOKENS:-256}"
    export SGLANG_MIXED_KV_HP_DTYPE=bfloat16
    export SGLANG_MIXED_KV_SCALE_DTYPE="${SCALE_DTYPE:-float32}"
    # Retaining cached prefixes at ~(window + ring) SWA tokens instead of full
    # length; hybrid-SWA only, inert on dense models. bf16 stays stock.
    export SGLANG_SWA_KEEP_PREFIX_TAIL="${SGLANG_SWA_KEEP_PREFIX_TAIL:-1}"
    args+=(--kv-cache-dtype int2)
    qgs="${QUANT_GROUP_SIZE:-$qgs_default}"
    [[ "$qgs" != "0" ]] && args+=(--kv-cache-quant-group-size "$qgs")
    if [[ "$method" == nova ]]; then
      if [[ ! -f "$artifact_dir/codebook.pt" ]]; then
        echo "missing VQ codebook: $artifact_dir/codebook.pt" >&2
        exit 2
      fi
      export SGLANG_VQ_CODEBOOK_PATH="$artifact_dir/codebook.pt"
      export SGLANG_VQ_FP8_FMT="${SGLANG_VQ_FP8_FMT:-e5m2}"
      export SGLANG_VQ_OPT_QMAP="${SGLANG_VQ_OPT_QMAP:-1}"
      export SGLANG_VQ_OPT_FLUSH="${SGLANG_VQ_OPT_FLUSH:-1}"
      export SGLANG_VQ_OPT_PREFILL="${SGLANG_VQ_OPT_PREFILL:-1}"
      kv_splits_default=48
    fi
    ;;
  *)
    echo "unsupported method: $method" >&2
    exit 2
    ;;
esac

args+=(--triton-attention-num-kv-splits "${KV_SPLITS:-$kv_splits_default}")

exec "${NOVA_PYTHON:-python3}" -m sglang.launch_server "${args[@]}" "$@"
