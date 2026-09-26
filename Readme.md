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
  ~48 GiB in FP8 (W4A16-Attn8-FP8PLE checkpoint). Measured host RAM in use: **~85 GiB** with FP8 PLE, **~149 GiB** with
  BF16 PLE (both measured 20 Sep 2026 while serving; the BF16 figure supersedes an earlier "~120 GiB" estimate).
  Plan for at least **96 GiB** (FP8 PLE) or **160 GiB-class** (BF16 PLE); the reference host has 157 GiB usable and
  the BF16 table leaves only ~8 GiB of it free.
  Avoid swapping.
- **NVMe storage**: the checkpoint is ~124 GB (W4A16-Attn8-FP8PLE, revision of 13 Sep 2026 or later) or larger (Intel), plus Docker layers and compile caches.
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

### Recipe 1 — TP2 × PP2, with or without MTP

Two Compose files, same settings except one line:

Pick the file that matches your checkpoint:

| File | Checkpoint | Speculative decoding | Notes |
|---|---|---|---|
| [`recipe/yml-tp2-pp-2-mtp-attn8.yml`](recipe/yml-tp2-pp-2-mtp-attn8.yml) | Attn8-FP8PLE | MTP, 2 draft tokens | **The reference configuration** — what the rig serves. Vision on, `GPU_MEMORY_UTILIZATION` 0.94. |
| [`recipe/yml-tp2-pp-2-intel-autoround.yml`](recipe/yml-tp2-pp-2-intel-autoround.yml) | Intel AutoRound | off (not available) | BF16 PLE table: **~149 GiB host RAM**, and needs the PLE gate patch. `GPU_MEMORY_UTILIZATION` 0.88 — its weights are ~0.6 GiB/rank larger. |

The two generic files below take any checkpoint via `MODEL_DIR` and are what the named files above are built from:

| File | Speculative decoding | Use when |
|---|---|---|
| [`recipe/yml-tp2-pp-2-mtp.yml`](recipe/yml-tp2-pp-2-mtp.yml) | MTP, 2 draft tokens | You want faster generation. Decode is about 1.6× (prose) to 1.9× (code) faster; the KV pool is about 45% smaller. |
| [`recipe/yml-tp2-pp-2.yml`](recipe/yml-tp2-pp-2.yml) | off | You want the largest KV pool (about 995,000 tokens). |

Intel cannot use the MTP files: it ships `model_extra_tensors.safetensors` (a BF16 draft head), not the
`mtp-model-*.safetensors` this vLLM loads.

vLLM is pinned to nightly **`29468dde8b515031dc6d4d9d06bf0a2fa0442098`** (25 Sep 2026, image
`vllm/vllm-openai:nightly-29468dde8b51…`, digest `sha256:c718e2d7…`) plus the patches in the next section, built locally as
`local/qwen38-flash-next:29468dde-ampere-pp-mtp`. The image reports its own version as `0.30.1rc1.dev143+g29468dde8`; the
commit hash is the identity. MTP needs the checkpoint revision that ships `mtp-model-*.safetensors` (13 Sep 2026 or later).

**Why TP2 × PP2 instead of TP4?** The model has only **two KV heads**, so its KV cache can be split across
two GPUs, not four. With **TP4**, the heads have to be duplicated across GPUs rather than split further.

**TP2 × PP2** uses the same four GPUs as two pairs: each pair splits the two KV heads, and handles only half
of the model's layers. This halves the attention KV memory needed per token on each GPU, giving roughly
**2× the KV cache capacity compared with TP4**—more room for cached conversations.

The trade-off: pipeline stages sometimes wait for each other during prompt processing. Speculative decoding with MTP
works under pipeline parallelism since this vLLM base (upstream #46994 plus a local fix for this model's draft head).
DFlash and DSpark are not available for this model in vLLM.

## vLLM patches

The base image is stock; the overlay replaces 11 source files (full detail, upstream links and rebase notes in
[`recipe/patch/README.md`](recipe/patch/README.md)):

| Change | Purpose |
|---|---|
| **Local: Ampere byte path for upstream FP8 KV ([#55557](https://github.com/vllm-project/vllm/pull/55557))** | FP8 KV on the QSA path is upstream now, but its kernel loads the cache as `float8_e4m3fn`, which Triton rejects on sm86. On sm<89 the cache stays raw E4M3 bytes and is decoded in kernel registers (bit-exact for all finite codes). |
| **Local: Ampere FP8 PLE lookup** | Upstream's pinned-host PLE lookup kernel refuses fp8 on sm86. The kernel only copies bytes, so the gather runs as `uint8`; the result is bit-identical. |
| Local fix for [#54709](https://github.com/vllm-project/vllm/issues/54709) | Placement-aware PLE checks so PP>1 is accepted. |
| [#54793](https://github.com/vllm-project/vllm/pull/54793) | Handle empty KV groups under PP. |
| [#54795](https://github.com/vllm-project/vllm/pull/54795) | Intersect compatible KV layouts across workers. |
| **Local mirror of [#46994](https://github.com/vllm-project/vllm/pull/46994) for `qwen4_exp`** | Upstream enables MTP under PP but fixes the draft head only for `qwen3_5_mtp`. Without this 3-line fix the `qwen4_exp` drafter skips its `fc` projections on the last stage. |
| **Local: PLE gate for unquantized tables** | `from_quant_config` has no branch for "body quantized by INC, PLE table not quantized", so an INC/auto-round checkpoint with a BF16 PLE table (e.g. the Intel checkpoint) dies at boot with `NotImplementedError ... INCConfig`. Adds one branch that trusts the checkpoint's own INC rule (`.*ple.*` = 16-bit float); anything not provably unquantized still raises. Upstream-bound. |
| [#55506](https://github.com/vllm-project/vllm/pull/55506) (25 Sep revision), **still open upstream** | Index the mamba spec-decode block tables by request slot, not batch row. Without it, PP + MTP + prefix caching poisons recurrent state and a share of requests loop on one token forever. Added after the v1.1.0 validation and measured on its own — see [Known failure modes](#known-failure-modes). |

Already in the base (patches in the previous recipe, not needed any more): #53899 host-resident PLE table, FP8 PLE table
loading with `--quantization inc`, #55375 fused-PLE stride fix, #55745 draft-broadcast stream guard, #46994 MTP under PP.

`recipe/patch/SHA256SUMS` records the overlay hashes; `nightly-29468dde-flashnext-mtp.patch` (all 11 files) is the
same change as one unified diff, and `pr55506-mamba-spec-block-table-req-slot.patch` (3 files) and
`ple-inc-bf16-gate.patch` (1 file) isolate those two changes, for audit/rebase (the
Dockerfile copies the overlay and does **not** apply the diffs; do not do both).

```bash
docker build -t local/qwen38-flash-next:29468dde-ampere-pp-mtp recipe/patch
( cd recipe/patch && sha256sum -c SHA256SUMS )
```

### Measured reference results

All throughput values are **tokens per second**, one request at a time (C=1), measured with club-3090 `bench.sh`
(3 warm-ups, 5 measured runs). Prefill is **cold** (cache-busted) at ~10K and ~100K input tokens. Decode excludes time to
first token. Checkpoint: W4A16-Attn8-FP8PLE, vision on, `--max-num-seqs 3`, utilization 0.94.

| Recipe (vLLM `eed1f3d0`) | KV pool (tokens) | Cold prefill 10K / 100K | Decode prose / code | Draft acceptance | Free VRAM under load, tightest GPU |
|---|---:|---:|---:|---:|---:|
| TP2 × PP2, **MTP n=2** (`yml-tp2-pp-2-mtp.yml`) | 526,909 | 4,404 / 4,601 | **131.8 / 160.5** | 70% (position 1: 84%, position 2: 68%) | 904 MiB |
| TP2 × PP2, no speculation (`yml-tp2-pp-2.yml`) | **1,017,569** | 4,380 / 4,583 | 83.8 / 84.1 | — | 1,144 MiB |
| TP2 × PP2, MTP n=3 (not shipped) | 521,742 | 4,325 / 4,520 | 131.9 / 171.5 | 53% | 860 MiB |

The MTP n=2 row is the 13 Sep 2026 run of this repository as published: image built from `recipe/patch/Dockerfile`, booted
with `yml-tp2-pp-2-mtp.yml` from a clean compile cache (boot to healthy 422 s). Its KV pool and the no-speculation pool
(1,017,569, boot 302 s) come from that run; the no-speculation throughput and the n=3 row come from the same image and
checkpoint one day earlier (KV pool 995,019 / 540,016 in that session; the ±2% boot-to-boot spread applies).
MTP does not change prefill; it trades KV pool for decode speed. n=3 adds 6.6% on code and nothing on prose, so n=2 is
the shipped default.

Same run, MTP n=2: quality suite **68/75** pass@1 (Task 83 protocol, three earlier boots without MTP: 66–69), streaming
tool-call probe **60/60**, prefix cache hit on the second call of a 29K-token prompt (25,472 cached tokens), 0 OOM.

Previous recipe (`v1.0.0-e962733e`, 5–8 Sep 2026, same checkpoint, no speculation): KV pool 941,463, prefill 10K / 90K
4,355 / 4,183, decode 72.3. The new base alone gives about +16% decode and +5.7% KV pool.

Notes on reading this table:

- KV pool is the engine's logical token capacity for the whole engine, not per GPU and not the context limit.
  Boot-to-boot allocation varies by about ±2%.
- Decode differences below ~11% are inside this rig's noise.
- The MTP profiles leave less than 1 GiB free per GPU under load. Watch for OOM on other GPUs or with larger images.

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
| Engine | vLLM nightly `29468dde`, patched | Ampere FP8 KV and FP8 PLE lookup, pipeline-parallel fixes, MTP draft-head fix; not stock vLLM |
| Parallelism | TP=2 × PP=2, expert parallel | `--enable-expert-parallel` (required: 640-wide MoE intermediate is not divisible by group 128 under TP4) |
| Weights / activations | `--quantization inc --dtype bfloat16` | BF16 activations; INT4 experts via Marlin, INT8 projections via AllSpark |
| PLE table | host RAM | `--engram-config '{"cpu_offload": true}'` (the old `VLLM_PLE_CPU_OFFLOAD` env var is ignored by this vLLM without a warning) |
| KV cache | FP8 | `--kv-cache-dtype fp8_e4m3` (Ampere byte path from the patch set) |
| Context limit | 262,144 tokens | `--max-model-len 262144` (the checkpoint's native ceiling) |
| GPU memory utilization | 0.94 | `--gpu-memory-utilization 0.94`; 0.95 left <650 MiB free on stage 0 |
| Concurrency | 3 sequences | `--max-num-seqs 3`; 3 × 262K still fits the pool |
| Prefill batch budget | 1,024 tokens | `--max-num-batched-tokens 1024`, chunked prefill on |
| Prefix caching | on | `--enable-prefix-caching --mamba-cache-mode align`; hits are in 3,136-token blocks |
| Speculative decoding | MTP, 2 draft tokens (`yml-tp2-pp-2-mtp.yml`) or off (`yml-tp2-pp-2.yml`) | `--speculative-config={"method":"mtp","num_speculative_tokens":2}`; the drafter is the checkpoint's 4-bit/8-bit MTP head (1.33 GiB) |
| CUDA graphs | `FULL_AND_PIECEWISE` | `--compilation-config={"cudagraph_mode":"FULL_AND_PIECEWISE"}`; FULL roughly doubles decode vs PIECEWISE on this backend |
| Indexer workspace | 128 MiB | `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128` (upstream default 512 MiB does not fit the ~1 GiB margin) |
| All-reduce / NCCL | NCCL, P2P allowed | `--disable-custom-all-reduce` (no NVLink), `NCCL_P2P_DISABLE=0`, `NCCL_P2P_LEVEL=SYS`, `NCCL_CUMEM_ENABLE=0`, 4 channels |
| Allocator | expandable segments | both `PYTORCH_ALLOC_CONF` and `PYTORCH_CUDA_ALLOC_CONF` (torch 2.13 and vLLM read different names) |
| Chat template | Sharp v22.4.0 | `--chat-template /models/qwen38-sharp-v22.4.0.jinja`. This is what the reference results were measured with. The author's own deployment moved to [froggeric v22.5](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) on 15 Sep 2026; that template scored 66/75 on the same suite (inside the 66–69 spread of this one) but has not been re-validated across the rest of this recipe, so the recipe still pins Sharp. |
| Parsers | reasoning `qwen3`, tools `qwen3_coder` | `--enable-auto-tool-choice`; reasoning is returned in the `reasoning` field |
| Vision | on, bounded | `--limit-mm-per-prompt={"image":8}`, `--mm-processor-kwargs={"max_pixels":1048576}`; the limit counts images in the whole prompt; raising it from 2 to 8 did not change the KV pool |
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
   # or the Intel checkpoint (BF16 PLE table — needs ~149 GiB host RAM, and the PLE gate patch below):
   # hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound --local-dir ./models/Intel-Qwen3.8-Flash-Next-W4A16-AutoRound
   ```

   Run long downloads inside `screen`. The container runs offline (`HF_HUB_OFFLINE=1`), so an incomplete download is
   not repaired at boot. The MTP profile needs `mtp-model-00001-of-00002.safetensors` and `mtp-model-00002-of-00002.safetensors`
   in the checkpoint (revision of 13 Sep 2026 or later). Do not use the NVFP4 checkpoints: Ampere is not a native NVFP4 target.

2. **Chat template** (pinned; the checkpoint's built-in template is not used)

   ```bash
   curl -fL --retry 3 \
     'https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates/resolve/5cb86e230acb03ffd992b841ecb12318a518e374/archive/v22.4.0-sharp/chat_template.jinja' \
     -o templates/qwen38-sharp-v22.4.0.jinja
   echo '180e7015759b2b6b57574d6c2ca5c2d19eb2b05a4aaffa80866f71eb1a1fad1a  templates/qwen38-sharp-v22.4.0.jinja' | sha256sum -c -
   ```

3. **Validate and build**

   ```bash
   # pick one: recipe/yml-tp2-pp-2-mtp.yml (MTP) or recipe/yml-tp2-pp-2.yml (no speculation)
   docker compose --project-directory "$PWD" --env-file .env -f recipe/yml-tp2-pp-2-mtp.yml config --quiet
   docker compose --project-directory "$PWD" --env-file .env -f recipe/yml-tp2-pp-2-mtp.yml build vllm
   ```

4. **Start** (only on idle GPUs; nothing else may hold the four cards)

   ```bash
   docker compose --project-directory "$PWD" --env-file .env -f recipe/yml-tp2-pp-2-mtp.yml up -d vllm
   docker compose --project-directory "$PWD" --env-file .env -f recipe/yml-tp2-pp-2-mtp.yml logs -f vllm
   ```

   Boot to healthy: about 4 minutes with warm compile caches, about 6 minutes cold (first run on a machine). Check the
   log for the reported KV pool, `Resolved Engram configuration: EngramConfig(cpu_offload=True…)`, and for the MTP profile
   `speculative_config=SpeculativeConfig(method='mtp'…`, and the absence of OOM/traceback lines before benchmarking.

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
   docker compose --project-directory "$PWD" --env-file .env -f recipe/yml-tp2-pp-2-mtp.yml logs --no-color > "logs/vllm-$(date +%Y%m%d-%H%M%S).log"
   docker compose --project-directory "$PWD" --env-file .env -f recipe/yml-tp2-pp-2-mtp.yml down
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


## Changelog

Version names are internal to this repository and point at the vLLM image the recipe was built with.

- **v1.4.0-29468dde — 26 Sep 2026.** vLLM nightly `29468dde` (25 Sep 2026). Decode and quality on par with v1.3.0,
  KV pool +1%, cold prefill −3 to −4%. `VLLM_PLE_CPU_OFFLOAD` is silently ignored on this base: use
  `--engram-config '{"cpu_offload": true}'`. Upstream's FP8 QSA KV cache does not compile on sm_86 without this
  recipe's byte-decode patch.
- **v1.3.0-eed1f3d0 — 21 Sep 2026.** Same base image as v1.2.0, one more overlay change: the **PLE gate**
  now accepts an unquantized PLE table on an INC-quantized checkpoint. Without it,
  [Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) —
  the base most derivatives come from — aborts at boot with `NotImplementedError: Qwen4Exp PLE embedding does
  not support quantization config INCConfig`. Reported by a user following this recipe; the gate came from
  upstream, not from this overlay. It is deliberately **not** loosened into a general fallback: anything not
  provably unquantized still raises, because a quantized PLE table silently loaded as BF16 produces wrong
  answers with no error. Adds two checkpoint-specific Compose files
  (`yml-tp2-pp-2-intel-autoround.yml`, `yml-tp2-pp-2-mtp-attn8.yml`) so neither checkpoint needs hand-edited
  settings. Corrects the measured host RAM for the BF16 PLE table: **~149 GiB, not ~120 GiB** — a 128 GB host
  will swap. Built and verified from this repository on 21 Sep 2026, both checkpoints: Intel boot 393 s,
  KV pool 453,819 at utilization 0.88, no MTP (it ships a BF16 draft head, not `mtp-model-*.safetensors`);
  Attn8 boot 303 s, KV pool 540,016 — identical to v1.2.0, and the new branch never executes on that path.
- **v1.2.0-eed1f3d0 — 17 Sep 2026.** Same base image and same nine files as v1.1.0, plus
  [#55506](https://github.com/vllm-project/vllm/pull/55506) (2 files, still open upstream): it removes a
  constant-token loop that hit 12.5% of responses at three concurrent requests, at no measurable cost to the KV
  pool, free VRAM, decode or prefill. The reference results below were measured on v1.1.0 and re-measured on the
  patched build within run-to-run spread, so they stand. Validated 16 Sep 2026 across five boots and 204 responses.
- **v1.1.0-eed1f3d0 — 13 Sep 2026.** vLLM nightly `eed1f3d0` (12 Sep 2026). Overlay reduced from 22 to 9 files: host PLE
  offload, FP8 PLE loading and the MTP-under-PP fixes are now upstream. MTP speculative decoding works on TP2 × PP2
  (`yml-tp2-pp-2-mtp.yml`, 2 draft tokens): decode 132 / 160 tok/s prose / code, KV pool 527K, quality 68/75, tool calls
  60/60. Without MTP: decode 84, KV pool 1,018K. Image limit per prompt raised to 8. Requires the checkpoint revision with
  the 4-bit MTP head. Built and verified from this repository on 13 Sep 2026.
- **v1.0.0-e962733e — 8 Sep 2026.** First release. vLLM nightly `e962733e` (5 Sep 2026), 22 overlay files, TP2 × PP2,
  FP8 KV, host-resident PLE table, no speculative decoding. Decode 72, KV pool 941K.

## Acknowledgements

- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — the original model.
- [Intel AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) — base checkpoint and INT4 expert quantization.
- [klee100](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP) — the 4-bit/8-bit MTP draft head reused in the checkpoint (only its MTP shards).
- [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) — original FP8 PLE lookup weights (only the PLE table is reused, not the NVFP4 backbone), obtained through [albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE).
- [vLLM](https://github.com/vllm-project/vllm) and the authors of the upstream PRs listed above, whose code the overlay carries.
- Alibaba [DashInfer](https://github.com/modelscope/dash-infer) team — the AllSpark W8A16 kernel (in vLLM) that serves the INT8 projections on Ampere.
- [peculiar-ragdoll / Sharp](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates), building on [froggeric](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) — chat template.
- [noonghunna/club-3090](https://github.com/noonghunna/club-3090) — community recipes and the benchmark harness used here.
- [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules), building on tinygrad's work — optional driver-level P2P.
- **todiadiyatmo / Tonjoo** — patch integration, the Ampere FP8 KV byte path and FP8 PLE lookup, the `qwen4_exp` MTP draft-head fix, the W4A16-Attn8-FP8PLE checkpoint, and validation on four RTX 3090s.


## Known failure modes

Two faults are worth knowing before you serve this recipe, because both are quiet.

**A constant-token loop under concurrency.** With PP + MTP + prefix caching, a share of requests degenerate into
one token repeated until the budget runs out (`ductductduct…`, independent of the prompt), and draft acceptance
collapses with it. That token is what the sampler emits for an all-NaN logits row — the request's recurrent state
has been poisoned. On sm_86 the underlying misdirected access stays mapped, so **there is no crash, no CUDA error
and nothing in the log**; the only symptom is the answer. Measured here at **12.5% of responses** (9/72) before
[#55506](https://github.com/vllm-project/vllm/pull/55506) and **0%** (0/132) after, Fisher exact p = 0.00006, at
no measurable cost. The overlay includes the fix. It needs **three or more concurrent requests** to appear — a
single-stream smoke test will not find it.

**`--no-async-scheduling` does not serve on this recipe.** Upstream threads recommend that flag as a mitigation for
MTP × hybrid-model corruption, and on tensor-parallel-only rigs it is reported at roughly no cost. On this
pipeline-parallel configuration the second stage blocks in `irecv_tensor_dict`, gloo times out after 30 minutes and
the engine dies: 36 of 36 requests failed in our test. Boot, warmup and CUDA-graph capture all pass first, so it
looks healthy until the first real request. The shipped configuration sets `--async-scheduling` explicitly.

**Not a failure, but worth expecting:** if you read `cached_tokens` to check prefix caching, a prompt shorter than
one KV block (3,184 tokens here) can never register a hit. Use a prompt of at least two blocks before concluding
that caching is broken.

## License

This repository is **MIT** (see [`LICENSE`](LICENSE)), **except** `recipe/patch/overlay/` and `recipe/patch/*.patch`,
which are modified vLLM source files and stay under **Apache License 2.0** with their upstream headers
([`recipe/patch/LICENSE.vllm`](recipe/patch/LICENSE.vllm)).

Model weights, the chat template and drivers are not distributed here. The Qwen-derived weights follow the
**Qwen Community License 1.0** (a separate license from Qwen is required for "Model as a Service" or "AI Work
Assistant" businesses; internal use is exempt), the Sharp template follows its own repository's terms, and the
optional P2P driver follows NVIDIA's and the patch authors' terms.
