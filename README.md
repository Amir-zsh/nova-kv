# nova-kv

Artifact for *Spend Bits Where Queries Look: Attention-Aware KV-Cache
Quantization*. The serving engine (`python/`, a research fork of SGLang
v0.5.10) holds full-attention layers' KV cache in a mixed pool — a shared BF16
prefix, a per-request BF16 recent window, and rotated low-bit pages for the
rest: group-VQ keys (fp8 codebook) and INT2 values. Both the accuracy and the
throughput numbers in the paper are produced by this repo.

Methods served by `scripts/serve_method.sh`: `bf16` (baseline), `nova` (the
paper method), `oscar` (nova without the VQ key tier), `quarot` / `turboquant`
(real-INT2 baselines, no calibration needed), `turboquant_k3v3` (simulated
3-bit baseline, accuracy only).

The gpt-oss-20b calibration bundle and its short-task datasets ship in-repo
(sha256 in `artifacts/MANIFEST.json`), so gpt-oss reproduces out of the box
after step 1. Everything else is built from scratch in steps 2 and 5: this
archive is self-contained and downloads nothing except public model weights
and, if you delete the cached corpus, the public `simonjegou/ruler` dataset.

## Step 1 — Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e "python[all]"
```

Or point `PYTHONPATH=python` at an environment that already has the SGLang
dependency stack. Verified with torch 2.9.1 + cu128 on H100. Model weights
download from Hugging Face on first serve.

## Step 2 — Build the datasets

Short-task rows for gpt-oss (GPQA / MATH-500 / AIME25) ship in `datasets/`.
The NIAH long-context rows are generated — haystacks come from the public
`simonjegou/ruler` corpus, needles are seeded, and the model's chat template
is applied to produce client-ready rows:

```bash
python datasets/build_niah_rows.py --model openai/gpt-oss-20b \
  --length 8192 --seed 0 --out datasets/niah_8192_gptoss_s200.jsonl
```

Repeat per length (8192 … 131072) and per model (`Qwen/Qwen3-8B`,
`meta-llama/Llama-3.1-8B-Instruct`). Generation is deterministic under the
seed and reproduces the rows behind the paper's tables: for Qwen the output is
byte-identical to ours, and for gpt-oss it differs only in the `Current date:`
line that model's chat template stamps at render time. The essay haystack
ships as `datasets/niah_corpus_essays.txt`; delete it and the script
re-extracts it from the public `simonjegou/ruler` dataset.

## Step 3 — Accuracy experiment (example: NIAH, gpt-oss-20b)

Terminal 1 — serve the method:

```bash
PREFILL_BACKEND=triton scripts/serve_method.sh \
  openai/gpt-oss-20b nova artifacts/gptoss_20b \
  --port 30000 --context-length 36864 --mem-fraction-static 0.85 \
  --chunked-prefill-size 8192 --moe-runner-backend triton_kernel
```

Terminal 2 — run the rows and score them:

```bash
python scripts/run_cell.py \
  --task ruler_niah --model openai/gpt-oss-20b --method nova --seed 0 \
  --input datasets/niah_8192_gptoss_s200.jsonl \
  --output results/niah8k_nova.jsonl --workers 8 \
  --base-url http://127.0.0.1:30000
python -m nova_kv.evaluation.evaluate_jsonl results/niah8k_nova.jsonl --metric contains
```

Expected: `nova` ≈ 0.80, `bf16` ≈ 0.79 (n=200, greedy). Swap the method
argument and artifact dir per arm; the full task × model × seed matrix is
`nova_kv/protocols/paper.json` (`scripts/run_matrix.py` prints or executes
it). Input rows carry `prompt`, `answer`/`answers`, optional `max_new_tokens`.
Serve knobs per model (backends, group size, Hadamard order) are set
automatically by `serve_method.sh`; its header documents every env override.

## Step 4 — Throughput experiment

Throughput needs no calibrated bundles — rotations/codebooks are synthesized
per model geometry (throughput-valid, accuracy-invalid):

```bash
python throughput/make_synthetic_artifacts.py --model openai/gpt-oss-20b
python -u throughput/run_throughput.py \
  --config throughput/configs/gptoss_20b.json --gpus 0,1,2,3
```

Protocol: decode-only throughput at bs = 1 / 4 / max per input length;
distinct random token streams (no cross-request prefix sharing); each
request's own prefix warmed before measuring; six per-cell validity gates.
Expected at 30k input (H100-80GB): bf16 ≈ 1,858 tok/s at bs=32 vs nova ≈
2,531 tok/s at bs=210 — the quantized pool holds 6.5M tokens where bf16 holds
1.26M from the same memory budget. Results land in
`results/throughput/<study>/` with per-cell provenance and gate status;
`throughput/configs/qwen3_8b.json` is the dense-model config.

## Step 5 — Build calibration bundles (gpt-oss ships pre-built)

A bundle directory holds `k_rotation_qqt_r_h_pbr.pt`,
`v_rotation_sst_r_h_pbr.pt`, and `codebook.pt`. Calibration runs offline under
HF transformers (for gpt-oss the MXFP4 checkpoint is dequantized to BF16 for
the forward passes):

```bash
# 1. capture Q/K/V from the 198 GPQA-Diamond calibration prompts
python calibration/dump_qkv.py --model Qwen/Qwen3-8B \
  --prompts datasets/gpqa_diamond.csv --num-prompts 198 --out dump/
# 2. rotations (head-dim 64 for gpt-oss, 128 for Qwen/Llama)
python calibration/compute_kv_rotation.py --dump-path dump/ \
  --output-dir artifacts/qwen3_8b --head-dim 128 \
  --method qqt_sst --composition r_h_pbr --chunk-id all
# 3. group-VQ key codebook (fp8 e5m2 as served)
python calibration/fit_vq_codebook.py --dump dump/ \
  --out artifacts/qwen3_8b/codebook.pt
# 4. simulated-TurboQuant baseline bundle (uses vendor/, no download)
python calibration/build_simquant_turboquant.py --head-dim 128 --layers 36 \
  --k-bits 3 --v-bits 3 --out artifacts/qwen3_8b/turboquant_k3v3.pt
```

## Layout

- `python/` — the serving engine (SGLang v0.5.10 fork; Apache-2.0).
- `vendor/` — vendored third-party TurboQuant implementation (MIT), used
  only to rebuild the simulated baseline bundle.
- `scripts/` — serve and evaluation entry points.
- `throughput/` — capacity model, synthetic artifacts, benchmark driver, configs.
- `calibration/` — offline HF-transformers calibration and baseline builders.
- `nova_kv/` — scoring package and the paper protocol.
- `datasets/` — small task rows in-repo; NIAH rows built by `build_niah_rows.py`.
- `artifacts/` — gpt-oss bundle + tuned kernel launch tables + shipped-file
  checksums (`MANIFEST.json`); other bundles are built into place.
