# Phase 5 baseline config
**Git SHA:** 6d79f2af32580bf531cdbf76bce3697e87bb3f19
**Date:** 2026-06-16T21:29:26Z
**Hardware:** Nebius gpu-h100-sxm / 1gpu-16vcpu-200gb
**Image:** ubuntu24.04-cuda13.0 (default Nebius)
**vLLM version:** 0.10.2
**transformers version:** 4.57.6
## vLLM serving flags (from scripts/start_vllm.sh)
```
#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --max-num-seq 128 \
    --gpu-memory-utilization 0.90 \
    --guided-decoding-backend xgrammar \
    --enable-prefix-caching \
    --enable-chunked-prefill
```
## Agent config
- MODEL: Qwen/Qwen3-30B-A3B-Instruct-2507
- MAX_ITERATIONS: 3
## Headline eval result
- final_exec_match_rate: 43.3%
- mean_iterations: 1.57
- per-iteration: k1=36.7% → k2=43.3% → k3=43.3%
