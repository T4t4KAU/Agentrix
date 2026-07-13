# ForkAttention Tensor-Parallel Model Compatibility

## Validation Method

Each architecture is validated independently on the AutoDL CUDA 12.8 host
with two RTX 5090 GPUs. The server uses ForkAttention, bfloat16, tensor
parallelism 2, prefix caching, and no data parallelism or KV offloading. A
small API benchmark verifies model loading, CUDA Graph capture, tokenization,
generation, metrics collection, and clean server shutdown.

The common command shape is:

```bash
MODEL_PATH=/root/autodl-tmp/models/<model> \
SERVED_MODEL_NAME=<name> \
BACKENDS=FORK_ATTN \
DP_REPLICAS=1 \
TP_SIZE=2 \
GPU_IDS=0,1 \
DTYPE=bfloat16 \
PREFIX_TOKENS=2048 \
BRANCHES=2 \
OUTPUT_TOKENS=32 \
MAX_MODEL_LEN=4096 \
./benchmark/scripts/run_vllm_benchmark.sh
```

## Results

| Architecture | Model | Status | GPU KV capacity | Branch tok/s | E2E tok/s | Logical KV reduction |
|---|---|---|---:|---:|---:|---:|
| `Qwen3ForCausalLM` | Qwen3-8B | Passed | 220,592 tokens | 118.22 | 44.20 | 48.91% |

## Qwen3ForCausalLM

Qwen3-8B loaded natively on both TP ranks, selected the ForkAttention backend,
captured CUDA Graphs, and completed all API requests. The workload saved 2,100
logical KV token entries, equivalent to 0.288 GiB for this model. No vLLM
source adaptation was required.

Raw results are stored on the AutoDL host at:

```text
/root/autodl-tmp/Agentrix/benchmark/results/tp_model_compat/qwen3_8b
```
