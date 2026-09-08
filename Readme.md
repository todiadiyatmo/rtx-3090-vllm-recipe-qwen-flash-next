# Qwen3.8-Flash-Next on 4× RTX 3090

A recipe for running **Qwen3.8-Flash-Next** (125B MoE, hybrid GDN + sparse attention, PLE n-gram table) on
Ampere-class GPUs such as the **NVIDIA RTX 3090**. The goal is a large usable KV cache on four 24 GiB cards,
practical single-request throughput, and the PLE lookup table kept in host RAM instead of VRAM.

Everything here is version-pinned: a specific vLLM nightly, a specific set of patches, a specific chat template.
It is not a claim that stock vLLM or every Ampere GPU can run this model.

## Prerequisites

- **4× RTX 3090 24 GiB**, or an equivalent Ampere-or-newer setup with 24 GiB per GPU. The reference rig runs a 200 W
  power cap per card, PCIe only (no NVLink), some cards on Gen1 x4 risers. Topology and power limits change the numbers below.
- **Host RAM for the PLE table plus runtime overhead.** The PLE lookup table is ~95 GiB in BF16 (Intel checkpoint) and
  ~48 GiB in FP8 (W4A16-Attn8-FP8PLE checkpoint). Measured host RAM in use: **~68 GiB** with FP8 PLE, **~120 GiB** with
  BF16 PLE. Plan for at least **96 GiB** (FP8 PLE) or **160 GiB-class** (BF16 PLE); the reference host has 157 GiB usable.
  Avoid swapping.
- **NVMe storage**: the checkpoint is ~127 GB (W4A16-Attn8-FP8PLE) or larger (Intel), plus Docker layers and compile caches.
  Budget a few hundred GB.
- **Docker Engine, Docker Compose v2 and the NVIDIA Container Toolkit**, with all four GPUs visible to containers.
  The whole recipe is a stock vLLM Docker image plus an overlay of patched source files; no bare-metal Python install.
- **An NVIDIA driver compatible with the CUDA 13.0 base image** (reference host: 610.57.04).
- **Optional: GPU P2P** via the patched driver from
  [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules) (branch matching your driver,
  e.g. `610.57.04-p2p-v3`; needs Above 4G Decoding, Resizable BAR and `iommu=pt`). Not installed by this recipe; see
  [P2P notes](#p2p-notes) below.
- A complete checkpoint (weights, index, config, tokenizer, processor files) and the **Sharp v22.4.0 chat template**
  (download step below). Accept the upstream license terms before downloading.

## Recipe

### Recipe 1 — TP2 × PP2

Compose file: [`recipe/yml-tp2-pp-2.yml`](recipe/yml-tp2-pp-2.yml). vLLM is pinned to nightly
**`e962733e08d10f7ca65dac4df99e116460b8b174`** (image `vllm/vllm-openai:nightly-e962733e08d1…`, digest
`sha256:89dd8f44…`) plus the patches in the next section, built locally as `local/qwen38-flash-next:e962733e-ampere-pp`.

**Why TP2 × PP2 instead of TP4?** The model has only **two KV heads**, so its KV cache can be split across
two GPUs, not four. With **TP4**, the heads have to be duplicated across GPUs rather than split further.

**TP2 × PP2** uses the same four GPUs as two pairs: each pair splits the two KV heads, and handles only half
of the model's layers. This halves the attention KV memory needed per token on each GPU, giving roughly
**2× the KV cache capacity compared with TP4**—more room for cached conversations.

The trade-off: pipeline stages sometimes wait for each other during prompt processing, and speculative decoding
(MTP/DFlash/DSpark) is unavailable with pipeline parallelism in this vLLM version.

## vLLM patches

The base image is stock; the overlay adds these changes (full detail, upstream links and rebase notes in
[`recipe/patch/README.md`](recipe/patch/README.md)):

| Change | Purpose |
|---|---|
| [#53899](https://github.com/vllm-project/vllm/pull/53899), manually ported | Host-resident PLE: offload worker, CUDA IPC, graph synchronization. |
| [#54846](https://github.com/vllm-project/vllm/pull/54846), manually ported | FP8 KV cache on the QSA (sparse attention) path. |
| **Local: Ampere `KV_FP8_BYTES` byte path** | RTX 3090 has no FP8 tensor cores and Triton has no fp8 type on sm86, so the cache is stored as raw E4M3 bytes and decoded to BF16 in kernel registers. This is the one patch that exists nowhere upstream. |
| Local fix for [#54709](https://github.com/vllm-project/vllm/issues/54709) | Placement-aware PLE checks so PP>1 is accepted. |
| [#54793](https://github.com/vllm-project/vllm/pull/54793) | Handle empty KV groups under PP. |
| [#54795](https://github.com/vllm-project/vllm/pull/54795) | Intersect compatible KV layouts across workers. |
| [#55375](https://github.com/vllm-project/vllm/pull/55375), backport | Fix fused-PLE convolution state-index stride (merged upstream after the base was cut). |
| Local INC → FP8 PLE selector | Lets `--quantization inc` load the checkpoint's FP8 PLE table. |

Already in the base (not patches): #54517 fused PLE, #54513 separate QSA indexer, #54915 indexer workspace,
#54873 skip sparse padding, #54722 FP8 PLE scale validation, #54251 GDN warmup.

The overlay is 22 Python files copied byte-for-byte from the tested deployment; `recipe/patch/SHA256SUMS` records their
hashes and `recipe/patch/nightly-e962733e-flashnext-pp.patch` is the same change as a unified diff for audit/rebase
(the Dockerfile copies the overlay and does **not** apply the diff; do not do both).

```bash
docker build -t local/qwen38-flash-next:e962733e-ampere-pp recipe/patch
( cd recipe/patch && sha256sum -c SHA256SUMS )
```

### Measured reference results

All throughput values are **tokens per second**. Prefill is **cold** (cache-busted) at ~10K and ~90K input tokens.
Decode is **C=1 prose**; the server permits up to three concurrent sequences (`--max-num-seqs 3`) but the numbers
were measured at C=1.

| Model weights | KV pool (tokens) | Cold prefill 10K / 90K (tok/s) | Decode C=1 (tok/s) | VRAM used/GPU under load | PLE lookup table |
|---|---:|---:|---:|---|---|
| [Intel W4A16 AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) | 947,222 | 4,383 / 4,508 | ~53 | ~23.4 GiB on the tightest stage-0 GPU | BF16, ~95 GiB host RAM |
| [W4A16-Attn8-FP8PLE](https://huggingface.co/todiadiyatmo/Qwen3.8-Flash-Next-W4A16-Attn8-FP8PLE) | **941,463** | **4,355 / 4,183** | **72.33** | **22.64–23.00 GiB** across ranks | FP8, ~48 GiB host RAM |

Notes on reading this table:

- The Intel row is an older run (2 Sep 2026, previous nightly, text-only profile, utilization 0.95); the second row is
  the current image with vision on (5–6 Sep). They are reference points, not a controlled A/B.
- KV pool is the engine's logical token capacity for the whole engine, not per GPU and not the context limit.
  Boot-to-boot allocation varies by about ±2%.
- The decode gain of the second row comes from the INT8 attention/GDN projections being served by the AllSpark W8A16
  kernel on Ampere (half the weight bytes at decode). Decode differences below ~11% are inside this rig's noise.

### vLLM config

**GPU-to-GPU transfers (P2P).** The benchmarks in this README used a patched NVIDIA driver that lets GPUs
exchange data directly, rather than through system RAM.

- **With a P2P-capable driver:** set `NCCL_P2P_DISABLE=0` and `NCCL_P2P_LEVEL=SYS` in `.env`.
  These settings let NCCL use P2P, but cannot add P2P support to a driver that lacks it.
- **Without driver-level P2P support:** set `NCCL_P2P_DISABLE=1` in `.env`.
  Transfers between GPUs will go through system RAM instead, so multi-GPU communication will be slower.

On this rig, enabling P2P reduced an 80 MB all-reduce (a data-sharing operation across GPUs) from **26 ms to 5 ms**.
It also made processing a 100K-token prompt (prefill) **1.9× faster**. Token generation (decode) changed very little,
because its main bottleneck is reading model weights from GPU memory, not transferring data between GPUs.

**Benchmark your own setup if you run without P2P**—do not assume the reference speeds here will carry over.

- Hardware: RTX 3090 ×4, 24 GiB per GPU, 200 W power cap per card; ~157 GiB usable host RAM.
- vLLM configuration (current W4A16-Attn8-FP8PLE profile; identical for Intel except the model path):

| Setting | Value | Notes |
|---|---|---|
| Engine | vLLM nightly `e962733e`, patched | Ampere FP8 KV, PLE offload, pipeline-parallel fixes; not stock vLLM |
| Parallelism | TP=2 × PP=2, expert parallel | `--enable-expert-parallel` (required: 640-wide MoE intermediate is not divisible by group 128 under TP4) |
| Weights / activations | `--quantization inc --dtype bfloat16` | BF16 activations; INT4 experts via Marlin, INT8 projections via AllSpark |
| PLE table | host RAM | `VLLM_PLE_CPU_OFFLOAD=1`, `VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600` (large table, slow first load) |
| KV cache | FP8 | `--kv-cache-dtype fp8_e4m3` (Ampere byte path from the patch set) |
| Context limit | 262,144 tokens | `--max-model-len 262144` (the checkpoint's native ceiling) |
| GPU memory utilization | 0.94 | `--gpu-memory-utilization 0.94`; 0.95 left <650 MiB free on stage 0 |
| Concurrency | 3 sequences | `--max-num-seqs 3`; 3 × 262K still fits the pool |
| Prefill batch budget | 1,024 tokens | `--max-num-batched-tokens 1024`, chunked prefill on |
| Prefix caching | on | `--enable-prefix-caching --mamba-cache-mode align`; hits are in 3,136-token blocks |
| Speculative decoding | off | no MTP / DFlash / DSpark (not supported under PP) |
| CUDA graphs | `FULL_AND_PIECEWISE` | `--compilation-config={"cudagraph_mode":"FULL_AND_PIECEWISE"}`; FULL roughly doubles decode vs PIECEWISE on this backend |
| Indexer workspace | 128 MiB | `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128` (upstream default 512 MiB does not fit the ~1 GiB margin) |
| All-reduce / NCCL | NCCL, P2P allowed | `--disable-custom-all-reduce` (no NVLink), `NCCL_P2P_DISABLE=0`, `NCCL_P2P_LEVEL=SYS`, `NCCL_CUMEM_ENABLE=0`, 4 channels |
| Allocator | expandable segments | both `PYTORCH_ALLOC_CONF` and `PYTORCH_CUDA_ALLOC_CONF` (torch 2.13 and vLLM read different names) |
| Chat template | Sharp v22.4.0 | `--chat-template /models/qwen38-sharp-v22.4.0.jinja` |
| Parsers | reasoning `qwen3`, tools `qwen3_coder` | `--enable-auto-tool-choice`; reasoning is returned in the `reasoning` field |
| Vision | on, bounded | `--limit-mm-per-prompt={"image":2}`, `--mm-processor-kwargs={"max_pixels":1048576}`; unbounded vision reserves encoder memory and shrinks the pool |
| Container | `ipc: host`, `SYS_PTRACE`, `seccomp=unconfined`, `shm_size 16gb` | needed by the PLE offload CUDA IPC path (`pidfd_getfd`); use on a trusted host |
| Logging / restart | json-file 50 MB × 5, `restart: "no"` | bounded logs; failures stay visible on first deployment |

## Setup and run

Run everything from this repository's root. Every Compose call sets `--project-directory` and `--env-file` so paths
resolve the same way regardless of where the yml lives.

1. **Settings and weights**

   ```bash
   cp .env.example .env      # set VLLM_API_KEY (long random), MODEL_DIR, CHAT_TEMPLATE, HOST_PORT
   mkdir -p models templates
   hf download todiadiyatmo/Qwen3.8-Flash-Next-W4A16-Attn8-FP8PLE --local-dir ./models/Qwen3.8-Flash-Next-W4A16-Attn8-FP8PLE
   # or the Intel checkpoint (needs BF16-PLE RAM budget):
   # hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound --local-dir ./models/Intel-Qwen3.8-Flash-Next-W4A16-AutoRound
   ```

   Run long downloads inside `screen`. The container runs offline (`HF_HUB_OFFLINE=1`), so an incomplete download is
   not repaired at boot. Do not use the NVFP4 checkpoints: Ampere is not a native NVFP4 target.

2. **Chat template** (pinned; the checkpoint's built-in template is not used)

   ```bash
   curl -fL --retry 3 \
     'https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates/resolve/5cb86e230acb03ffd992b841ecb12318a518e374/archive/v22.4.0-sharp/chat_template.jinja' \
     -o templates/qwen38-sharp-v22.4.0.jinja
   echo '180e7015759b2b6b57574d6c2ca5c2d19eb2b05a4aaffa80866f71eb1a1fad1a  templates/qwen38-sharp-v22.4.0.jinja' | sha256sum -c -
   ```

3. **Validate and build**

   ```bash
   docker compose --project-directory "$PWD" --env-file .env -f recipe/yml-tp2-pp-2.yml config --quiet
   docker compose --project-directory "$PWD" --env-file .env -f recipe/yml-tp2-pp-2.yml build vllm
   ```

4. **Start** (only on idle GPUs; nothing else may hold the four cards)

   ```bash
   docker compose --project-directory "$PWD" --env-file .env -f recipe/yml-tp2-pp-2.yml up -d vllm
   docker compose --project-directory "$PWD" --env-file .env -f recipe/yml-tp2-pp-2.yml logs -f vllm
   ```

   Boot to healthy: about 4 minutes with warm compile caches, about 6 minutes cold (first run on a machine). Check the
   log for the reported KV pool, `PleOffload` initialization, and the absence of OOM/traceback lines before benchmarking.

5. **Verify inference, not just health**

   ```bash
   curl -fsS http://127.0.0.1:8000/health
   curl -fsS --max-time 300 http://127.0.0.1:8000/v1/chat/completions \
     -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
     -d '{"model":"qwen38-flash-next","max_tokens":4000,"messages":[{"role":"user","content":"What is 7 multiplied by 8?"}]}'
   curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/v1/models   # must be 401 without a key
   ```

   Expect a coherent answer with `finish_reason: stop`. To verify prefix caching, send an identical prompt of at
   least ~3,200 tokens twice and require `usage.prompt_tokens_details.cached_tokens > 0` on the second call; short
   prompts never touch the cache path.

6. **Stop, keeping the logs**

   ```bash
   mkdir -p logs
   docker compose --project-directory "$PWD" --env-file .env -f recipe/yml-tp2-pp-2.yml logs --no-color > "logs/vllm-$(date +%Y%m%d-%H%M%S).log"
   docker compose --project-directory "$PWD" --env-file .env -f recipe/yml-tp2-pp-2.yml down
   ```

   This yml declares its own Compose project name (`rtx3090x4-qwen38-recipe`), so `down` here cannot remove another
   stack's containers. Keep `HOST_IP=127.0.0.1` unless a TLS/auth proxy is in front.

## Testing

We tested speed, stability and answer quality using the
[club-3090](https://github.com/noonghunna/club-3090) benchmark scripts. The paths below belong to that repository.

| Test | Command | What it checks |
|---|---|---|
| Speed | `scripts/bench.sh` | Prompt processing at 10K/90K tokens, text and code generation, and time to the first token. Generation tests use 3 warm-ups and 5 measured runs. |
| Stability | `scripts/soak-test.sh --continuous` and `--quick` | Checks for slowdowns and empty responses. `--continuous` grows conversations to about 22–25K tokens; `--quick` tests fresh conversations. |
| Tool calls | `scripts/stream-toolcall-probe.py --thinking on --tool-choice both --repeat 3` | 60 streaming tool-call tests. |
| Answer quality | `scripts/quality-test.sh --medium` | 75 scenarios across 5 test packs, scored by `benchlocal-cli`. |

**Authentication:** of these scripts, only `quality-test.sh` supports an API key (set `API_KEY` or pass `--api-key`).
For the other scripts, use a proxy that adds the
`Authorization` header, or a local test server with authentication disabled. Do not expose a keyless server publicly.


## Acknowledgements

- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — the original model.
- [Intel AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) — base checkpoint and INT4 expert quantization.
- [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) — original FP8 PLE lookup weights (only the PLE table is reused, not the NVFP4 backbone), obtained through [albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE).
- [vLLM](https://github.com/vllm-project/vllm) and the authors of the upstream PRs listed above, whose code the overlay carries.
- Alibaba [DashInfer](https://github.com/modelscope/dash-infer) team — the AllSpark W8A16 kernel (in vLLM) that serves the INT8 projections on Ampere.
- [peculiar-ragdoll / Sharp](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates), building on [froggeric](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) — chat template.
- [noonghunna/club-3090](https://github.com/noonghunna/club-3090) — community recipes and the benchmark harness used here.
- [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules), building on tinygrad's work — optional driver-level P2P.
- **todiadiyatmo / Tonjoo** — patch integration, the Ampere FP8 KV byte path, the W4A16-Attn8-FP8PLE checkpoint, and validation on four RTX 3090s.

## License

This repository is **MIT** (see [`LICENSE`](LICENSE)), **except** `recipe/patch/overlay/` and `recipe/patch/*.patch`,
which are modified vLLM source files and stay under **Apache License 2.0** with their upstream headers
([`recipe/patch/LICENSE.vllm`](recipe/patch/LICENSE.vllm)).

Model weights, the chat template and drivers are not distributed here. The Qwen-derived weights follow the
**Qwen Community License 1.0** (a separate license from Qwen is required for "Model as a Service" or "AI Work
Assistant" businesses; internal use is exempt), the Sharp template follows its own repository's terms, and the
optional P2P driver follows NVIDIA's and the patch authors' terms.
