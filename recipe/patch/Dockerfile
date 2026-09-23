# Exact base used for the September 12-13, 2026 validation (Task 85/86).
FROM vllm/vllm-openai:nightly-eed1f3d0c6043bd494424a22443ee198dd56f657@sha256:d0742e7e31b16c85c9e215960604452f5f172a522d35c94ca42b7030cb0afa36

# Complete replacement files from the tested overlay (11 files). Do not also apply the diffs.
# 9 of them were validated together on 12-13 Sep 2026 (Task 85/86). The 2 files under
# v1/worker/ carry vllm#55506 and were validated separately on 16 Sep 2026 (Task 87) --
# see README.md "One patch has a different evidence basis". ngram_embedding.py additionally
# carries the PLE gate change, validated on its own on 20-21 Sep 2026 (both checkpoints).
COPY overlay/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY LICENSE.vllm /usr/local/share/licenses/qwen38-recipe/LICENSE.vllm
LABEL id.tonjoo.patches="54846,ampere-kv-fp8-bytes,ampere-ple-fp8-uint8,54709-pp-gate,54793,54795,mtp-pp-mirror-46994,55506-mamba-spec-block-table,ple-inc-bf16-gate" \
      id.tonjoo.base="eed1f3d0c6043bd494424a22443ee198dd56f657" \
      id.tonjoo.base_digest="sha256:d0742e7e31b16c85c9e215960604452f5f172a522d35c94ca42b7030cb0afa36" \
      id.tonjoo.recipe="v1.3.0-eed1f3d0"
