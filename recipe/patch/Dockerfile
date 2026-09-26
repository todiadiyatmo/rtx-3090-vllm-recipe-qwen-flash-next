# Exact base used for the September 26, 2026 validation (Task 92).
FROM vllm/vllm-openai:nightly-29468dde8b515031dc6d4d9d06bf0a2fa0442098@sha256:c718e2d7402c7e0c3449e76b6c43a3c595406fbcba8da87de0fa17742b00aad5

# Complete replacement files from the tested overlay (11 files). Do not also apply the diffs.
# Built by a three-way merge of the v1.3.0 overlay onto nightly 29468dde. FP8 KV on the QSA path is now
# upstream vllm#55557 plus this recipe's Ampere byte-decode port; vllm#55506 is its 25 Sep revision, which
# also covers the new model_states/mamba_hybrid.py path. See README.md.
COPY overlay/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY LICENSE.vllm /usr/local/share/licenses/qwen38-recipe/LICENSE.vllm
LABEL id.tonjoo.patches="55557+ampere-kv-fp8-bytes,ampere-ple-fp8-uint8,54709-pp-gate,54793,54795,mtp-pp-mirror-46994,55506-mamba-spec-block-table,ple-inc-bf16-gate" \
      id.tonjoo.base="29468dde8b515031dc6d4d9d06bf0a2fa0442098" \
      id.tonjoo.base_digest="sha256:c718e2d7402c7e0c3449e76b6c43a3c595406fbcba8da87de0fa17742b00aad5" \
      id.tonjoo.recipe="v1.4.0-29468dde"
