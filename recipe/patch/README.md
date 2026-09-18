# vLLM patches — nightly `eed1f3d0`

This directory packages the overlay tested on four RTX 3090s on September 12–13,
2026, plus one patch added after that validation (recipe `v1.2.0-eed1f3d0`). It is not a
patch set for arbitrary vLLM releases.

## Pinned base

- Image: `vllm/vllm-openai:nightly-eed1f3d0c6043bd494424a22443ee198dd56f657`
- Digest: `sha256:d0742e7e31b16c85c9e215960604452f5f172a522d35c94ca42b7030cb0afa36`
- vLLM commit: `eed1f3d0c6043bd494424a22443ee198dd56f657` (12 Sep 2026). The image reports
  `vllm.__version__` as `0.1.1.dev50+geed1f3d0c`; treat the commit hash as the identity.
- PyTorch `2.13.0+cu130`, CUDA 13.0.

## Included changes (11 files)

| Change | Files | Purpose |
|---|---|---|
| [#54846](https://github.com/vllm-project/vllm/pull/54846), manually ported | `models/qwen4_exp/nvidia/{qsa.py,ops/qsa.py}`, `platforms/interface.py`, `v1/attention/backends/utils.py` | FP8 KV cache on the QSA sparse-attention path. |
| Local Ampere `KV_FP8_BYTES` path | same files | RTX 3090 has no FP8 tensor cores and Triton has no fp8 type on sm86. The cache is stored as raw E4M3 bytes and decoded to BF16 in kernel registers. |
| Local Ampere FP8 PLE lookup | `models/qwen4_exp/nvidia/ngram_embedding.py` | Upstream's pinned-host PLE lookup kernel fails on sm86 with `type fp8e4nv not supported`. The kernel only moves bytes, so the gather runs as `uint8`; the result is bit-identical. |
| Local fix for [issue #54709](https://github.com/vllm-project/vllm/issues/54709) | `model_executor/models/config.py`, `models/qwen4_exp/nvidia/model_state.py` | Placement-aware PLE checks so pipeline parallelism is accepted. Upstream still rejects `pipeline_parallel_size > 1` for this model. |
| [#54793](https://github.com/vllm-project/vllm/pull/54793) | `v1/core/kv_cache_utils.py` | Handle empty KV groups under PP. |
| [#54795](https://github.com/vllm-project/vllm/pull/54795) | `v1/attention/backends/utils.py` | Intersect compatible KV layouts across workers. |
| Local mirror of [#46994](https://github.com/vllm-project/vllm/pull/46994) for `qwen4_exp` | `models/qwen4_exp/nvidia/mtp.py` | #46994 enables MTP under PP but fixes the draft head only in `qwen3_5_mtp.py`. The `qwen4_exp` draft head still branches on `is_first_rank`, which is false on the last stage where the drafter lives, so it skips the `fc` projections. Branch on `intermediate_tensors is None` instead. |
| [#55506](https://github.com/vllm-project/vllm/pull/55506), **still open upstream** | `v1/worker/mamba_utils.py`, `v1/worker/gpu/model_runner.py` | Index the mamba spec-decode block tables by request slot, not by batch row. Without it, PP + MTP + prefix caching poisons recurrent state and a share of requests degenerate into a constant-token loop. See the section below. |

Already in the base, **not extra patches** (they were patches in `v1.0.0-e962733e`):
[#53899](https://github.com/vllm-project/vllm/pull/53899) host-resident PLE (now `ngram_embedding.py` and
`config/engram.py`), FP8 PLE table loading with `--quantization inc`,
[#55375](https://github.com/vllm-project/vllm/pull/55375) fused-PLE stride fix,
[#55745](https://github.com/vllm-project/vllm/pull/55745) `record_stream` guard in draft broadcast, and
[#46994](https://github.com/vllm-project/vllm/pull/46994) MTP under pipeline parallelism.

## One patch has a different evidence basis

Nine of the eleven files were validated **as one package** on 12–13 September 2026: a boot, a bench and a
quality run of the whole stack. That is enough to say the package works; it is not enough to attribute any
result to an individual file in it.

`#55506` is different. It was added on 16 September and measured **on its own**, with an A/B that the other
nine never got, because it fixes a fault we hit in production rather than one we anticipated.

**The fault.** With pipeline parallelism + MTP + prefix caching, a share of requests degenerate into a
constant-token loop (`ductductduct…`, prompt-independent) until they exhaust the token budget. The loop token
is what the sampler deterministically emits for an **all-NaN logits row**: the request's recurrent state has
been poisoned and never recovers. On sm_80/86 the misdirected access stays mapped, so there is **no crash, no
CUDA error and nothing in the log** — on sm_121 the same bug is loud ([#54173](https://github.com/vllm-project/vllm/issues/54173)).
Do not read the absence of errors as absence of the fault.

**Why it happens.** `MambaSpecDecodeGPUContext` captures the block tables' raw data pointers exactly once, but
the V2 model runner bound that capture to the per-step *gathered* tables (batch-ordered, re-gathered every
step) while the copy kernels resolved rows as `batch_idx if HAS_IDX_MAPPING else req_idx`. Under async
scheduling with PP, a non-last rank runs its postprocess `pp_size` steps after its batch was gathered, so the
mapping it holds is stale and the copy walks another request's freed or reallocated block ids. In the unified
cache layout main KV, GDN conv/SSM and PLE conv state all alias the same page and are distinguished only by
block-id ownership, so those writes land inside a live request's state. The fix binds the context to the
source per-request-slot tables and indexes by `req_idx` — the contract the V1 path already relied on.

**Measured here** (17 arm-hours, 204 responses across five boots, TP2×PP2 + MTP n=2 on four RTX 3090s,
three requests in flight, agentic-shaped prompts of 8–20K tokens):

| | loop rate | KV pool | free VRAM | decode prose / code | prefill 10K / 100K |
|---|---|---|---|---|---|
| without the patch | **9/72 (12.5%)** | 540,016 | 2,544 / 848 MiB | 127.8 / 158.7 | 4,288 / 4,516 |
| with the patch | **0/132 (0.0%)** | 540,016 | 2,538 / 844 MiB | 128.0 / 158.3 | 4,330 / 4,445 |

Fisher exact, two-sided: **p = 0.00006**. Every performance number is inside the anchor's own run-to-run
spread, and the KV pool is identical — this patch costs nothing measurable.

Two conditions matter for reproducing the fault: **three or more concurrent requests** (the reporter's
threshold is exactly 2→3 slots, and ours reproduced at 3) and long generations. A single-stream smoke test
will not show it.

⛔ **Do not "fix" this with `--no-async-scheduling`.** Upstream threads recommend that flag for MTP × hybrid
corruption, and it is measured at roughly no cost on tensor-parallel-only rigs. On this pipeline-parallel
recipe it does not serve **at all**: the second pipeline stage blocks in `irecv_tensor_dict`, gloo times out
after 30 minutes and the engine dies — 36 of 36 requests failed in our arm. Boot, warmup and CUDA-graph
capture all succeed first, so the failure only appears on the first real request.

## Build and audit

From the repository root:

```bash
docker build -t local/qwen38-flash-next:eed1f3d0-ampere-pp-mtp recipe/patch
```

`overlay/` contains 11 Python source files copied byte-for-byte from the tested
deployment. `SHA256SUMS` records their hashes and the unified diff's hash:

```bash
( cd recipe/patch && sha256sum -c SHA256SUMS )
```

The Dockerfile copies the overlay; it does **not** run the unified patches.
`nightly-eed1f3d0-flashnext-mtp.patch` (9 files) and
`pr55506-mamba-spec-block-table-req-slot.patch` (2 files) are the audit/rebase alternative. Both apply to the
base commit with `patch -p1` (0 fuzz; #55506 lands with one hunk at offset 32 lines). The two diffs touch
disjoint files, so order does not matter. Do not apply either on top of the overlay.

## Rebase notes

- Preserve `@triton.jit` decorators. A missing decorator passes Python syntax and import checks and fails only when the kernel is compiled on the GPU.
- QSA warmup calls the kernel positionally. Update it together with the quantization arguments and Ampere dispatch, not just the steady-state kernel.
- The PP gate for this model moved between nightlies; it did not disappear. Check `model_executor/models/config.py` for the current rejection message.
- If a later nightly changes how the V2 runner binds block tables, re-read `#55506` before carrying it
  forward: check what `preprocess_state` receives in `v1/worker/gpu/model_runner.py`. Once the PR merges,
  inherit it from the base and drop these two files instead of rebasing them.
- `VLLM_PLE_CPU_OFFLOAD=1` is still honored but logged as legacy; the upstream replacement is `--engram-config.cpu_offload`. The recipe keeps the environment variable because that is what was tested. `VLLM_PLE_OFFLOAD_READY_TIMEOUT` no longer exists.

## Provenance and licensing

Source: development logbook `logbook/vllm/patch/nightly-eed1f3d0/` and Task 85/86.
The overlay consists of modified vLLM files; retain their SPDX/copyright headers.
vLLM's Apache-2.0 license is included as `LICENSE.vllm`. Upstream contributors retain
credit for their code; this recipe does not claim authorship of the upstream PRs.
