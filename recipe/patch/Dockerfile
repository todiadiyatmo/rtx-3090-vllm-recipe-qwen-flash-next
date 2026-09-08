# Exact base used for the September 5-6, 2026 validation.
FROM vllm/vllm-openai:nightly-e962733e08d10f7ca65dac4df99e116460b8b174@sha256:89dd8f442a3f4c08c6b3cd634c4f735cd709160651c296596673cf974ea6ee39

# Complete replacement files from the tested overlay. Do not also apply the diff.
COPY overlay/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY LICENSE.vllm /usr/local/share/licenses/qwen38-recipe/LICENSE.vllm
LABEL id.tonjoo.patches="55375,53899,54846,ampere-kv-fp8-bytes,54709-pp-gate,54793,54795,inc-fp8-ple-gate" \
      id.tonjoo.base="e962733e08d10f7ca65dac4df99e116460b8b174" \
      id.tonjoo.base_digest="sha256:89dd8f442a3f4c08c6b3cd634c4f735cd709160651c296596673cf974ea6ee39"
