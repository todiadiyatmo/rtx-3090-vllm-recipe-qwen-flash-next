# vLLM patches — nightly `eed1f3d0`

This directory packages the overlay tested on four RTX 3090s on September 12–13,
2026 (recipe `v1.1.0-eed1f3d0`). It is not a patch set for arbitrary vLLM releases.

## Pinned base

- Image: `vllm/vllm-openai:nightly-eed1f3d0c6043bd494424a22443ee198dd56f657`
- Digest: `sha256:d0742e7e31b16c85c9e215960604452f5f172a522d35c94ca42b7030cb0afa36`
- vLLM commit: `eed1f3d0c6043bd494424a22443ee198dd56f657` (12 Sep 2026). The image reports
  `vllm.__version__` as `0.1.1.dev50+geed1f3d0c`; treat the commit hash as the identity.
- PyTorch `2.13.0+cu130`, CUDA 13.0.

## Included changes (9 files)

| Change | Files | Purpose |
|---|---|---|
| [#54846](https://github.com/vllm-project/vllm/pull/54846), manually ported | `models/qwen4_exp/nvidia/{qsa.py,ops/qsa.py}`, `platforms/interface.py`, `v1/attention/backends/utils.py` | FP8 KV cache on the QSA sparse-attention path. |
| Local Ampere `KV_FP8_BYTES` path | same files | RTX 3090 has no FP8 tensor cores and Triton has no fp8 type on sm86. The cache is stored as raw E4M3 bytes and decoded to BF16 in kernel registers. |
| Local Ampere FP8 PLE lookup | `models/qwen4_exp/nvidia/ngram_embedding.py` | Upstream's pinned-host PLE lookup kernel fails on sm86 with `type fp8e4nv not supported`. The kernel only moves bytes, so the gather runs as `uint8`; the result is bit-identical. |
| Local fix for [issue #54709](https://github.com/vllm-project/vllm/issues/54709) | `model_executor/models/config.py`, `models/qwen4_exp/nvidia/model_state.py` | Placement-aware PLE checks so pipeline parallelism is accepted. Upstream still rejects `pipeline_parallel_size > 1` for this model. |
| [#54793](https://github.com/vllm-project/vllm/pull/54793) | `v1/core/kv_cache_utils.py` | Handle empty KV groups under PP. |
| [#54795](https://github.com/vllm-project/vllm/pull/54795) | `v1/attention/backends/utils.py` | Intersect compatible KV layouts across workers. |
| Local mirror of [#46994](https://github.com/vllm-project/vllm/pull/46994) for `qwen4_exp` | `models/qwen4_exp/nvidia/mtp.py` | #46994 enables MTP under PP but fixes the draft head only in `qwen3_5_mtp.py`. The `qwen4_exp` draft head still branches on `is_first_rank`, which is false on the last stage where the drafter lives, so it skips the `fc` projections. Branch on `intermediate_tensors is None` instead. |

Already in the base, **not extra patches** (they were patches in `v1.0.0-e962733e`):
[#53899](https://github.com/vllm-project/vllm/pull/53899) host-resident PLE (now `ngram_embedding.py` and
`config/engram.py`), FP8 PLE table loading with `--quantization inc`,
[#55375](https://github.com/vllm-project/vllm/pull/55375) fused-PLE stride fix,
[#55745](https://github.com/vllm-project/vllm/pull/55745) `record_stream` guard in draft broadcast, and
[#46994](https://github.com/vllm-project/vllm/pull/46994) MTP under pipeline parallelism.

## Build and audit

From the repository root:

```bash
docker build -t local/qwen38-flash-next:eed1f3d0-ampere-pp-mtp recipe/patch
```

`overlay/` contains 9 Python source files copied byte-for-byte from the tested
deployment. `SHA256SUMS` records their hashes and the unified diff's hash:

```bash
( cd recipe/patch && sha256sum -c SHA256SUMS )
```

The Dockerfile copies the overlay; it does **not** run the unified patch.
`nightly-eed1f3d0-flashnext-mtp.patch` is the audit/rebase alternative. It applies to
the base commit with `patch -p1` (0 fuzz, 0 offset). Do not apply it on top of the overlay.

## Rebase notes

- Preserve `@triton.jit` decorators. A missing decorator passes Python syntax and import checks and fails only when the kernel is compiled on the GPU.
- QSA warmup calls the kernel positionally. Update it together with the quantization arguments and Ampere dispatch, not just the steady-state kernel.
- The PP gate for this model moved between nightlies; it did not disappear. Check `model_executor/models/config.py` for the current rejection message.
- `VLLM_PLE_CPU_OFFLOAD=1` is still honored but logged as legacy; the upstream replacement is `--engram-config.cpu_offload`. The recipe keeps the environment variable because that is what was tested. `VLLM_PLE_OFFLOAD_READY_TIMEOUT` no longer exists.

## Provenance and licensing

Source: development logbook `logbook/vllm/patch/nightly-eed1f3d0/` and Task 85/86.
The overlay consists of modified vLLM files; retain their SPDX/copyright headers.
vLLM's Apache-2.0 license is included as `LICENSE.vllm`. Upstream contributors retain
credit for their code; this recipe does not claim authorship of the upstream PRs.
