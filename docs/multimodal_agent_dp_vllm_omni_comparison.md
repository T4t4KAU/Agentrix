# Multimodal DP Experiment: vLLM 0.25.0 versus vLLM-Omni on vLLM 0.28.0

## Executive Summary

The multimodal data-parallel experiment was repeated with the repository at
`/home/hwx/Documents/codes/vllm-omni`, commit
`e51fe6ec1b9a9a0e14bb1fdb296d61b6593b93c6`. This checkout identifies as
vLLM-Omni `0.28.1.dev14+ge51fe6ec1` and is designed to run with upstream vLLM
`0.28.0`.

The central scheduling result did not change:

- A cold vision encode delays a text prefill only when both requests execute on
  the same GPU replica. The different-replica control remained approximately
  1.0x.
- Cache affinity is valuable at low load, but queueing on the hot replica
  eventually dominates. The crossover load still depends strongly on vision
  token count.
- Copying a BF16 vision embedding between GPU 0 and GPU 1 remained orders of
  magnitude cheaper than recomputing it.

Relative to the vLLM 0.25.0 run, measured same-replica interference was lower
with the vLLM 0.28.0 stack: slowdown changed from
`1.13x / 1.35x / 3.10x / 6.49x` to
`1.05x / 1.27x / 2.50x / 5.99x` for
`196 / 1,024 / 4,096 / 9,216` vision tokens. The isolated vision-encoder kernel
time, however, was effectively unchanged for medium through XLarge images.

An additional vLLM 0.28.0 run with `VLLM_PLUGINS=""` disabled the Omni plugin.
It did not show a consistent plugin-specific latency improvement. Qwen3.5-9B
uses vLLM's upstream `Qwen3_5ForConditionalGeneration`, not an Omni model
implementation. The observed 0.25-to-0.28 differences should therefore be
attributed primarily to the newer upstream vLLM, PyTorch, and Transformers
stack, plus normal variance in this small preliminary sample, rather than to
the Omni staged runtime.

## 1. Compared Environments

| Component | Original baseline | vLLM-Omni run |
|---|---|---|
| Repository | upstream vLLM | vLLM-Omni extension plus upstream vLLM |
| Source revision | vLLM `702f4814fe54fabff350d43cb753ae3e47c0c276` | vLLM-Omni `e51fe6ec1b9a9a0e14bb1fdb296d61b6593b93c6` |
| Package version | vLLM 0.25.0 | vLLM-Omni 0.28.1.dev14 and vLLM 0.28.0 |
| PyTorch | 2.11.0+cu130 | 2.13.0+cu130 |
| Transformers | baseline 0.25 environment | 5.14.1, constrained by vLLM-Omni |
| Python | 3.12.14 | 3.12.14 |
| GPU | 4 x NVIDIA RTX 5090 | same server and GPUs |
| Model | Qwen3.5-9B, BF16 | same model directory and dtype |
| Attention | FlashAttention 2 | FlashAttention 2 |
| Execution | eager, no Torch compile, no CUDA Graphs | same |
| Prefix cache | disabled | disabled |
| MM processor cache | 4 GiB | 4 GiB |

The vLLM-Omni environment was created independently with `uv` at:

```text
/root/autodl-tmp/vllm-omni-e51fe6ec/.venv
```

`uv pip check` reported all 242 installed packages compatible. The environment
reported CUDA 13.0 and detected all four RTX 5090 GPUs.

The two online replicas used GPU 0 and GPU 1 on ports 8000 and 8001. GPU 2 was
used for isolated encoder timing. The serving arguments, 4K-token text victim,
deterministic image sizes, one-token response, and request timing method were
kept the same as in the original experiment. Each interference and cache/load
cell has two repetitions; each encoder cell has three cold and three hot
measurements; P2P copy has 30 repetitions.

## 2. What vLLM-Omni Changes for This Model

vLLM-Omni is an extension package, not a forked replacement for the complete
vLLM runtime. Its `vllm.general_plugins` entry point registers Omni-specific
architectures and imports a set of compatibility patches into vLLM processes.

For this experiment, the important scope distinction is:

- Qwen3.5-9B resolved to upstream vLLM's
  `Qwen3_5ForConditionalGeneration`. The Omni registry does not replace that
  architecture.
- The model was served with the standard `vllm serve` path, not the Omni
  multi-stage scheduler or a non-text output pipeline.
- Startup reported three installed global patches: an NVFP4 weight-scale NaN
  clamp, an Inductor symbolic-divisibility proof, and a CuMem shutdown callback
  guard. The first is irrelevant to this BF16 model, the second is inactive
  under `--enforce-eager`, and the third concerns allocator lifecycle rather
  than steady-state inference.

Consequently, this run is best interpreted as a vLLM 0.25 versus vLLM 0.28
stack comparison with the Omni plugin present, not as evidence that the Omni
scheduler accelerates Qwen3.5 vision inference.

## 3. Vision Encode versus Text Prefill Interference

The arrival offsets were held fixed at 10, 30, 150, and 400 ms for the four
image sizes. These offsets still place the 4K-token text victim inside the
vision workload's GPU interval under the new stack.

| Vision tokens | 0.25 same-rank TTFT | 0.25 slowdown | 0.28 + Omni same-rank TTFT | 0.28 + Omni slowdown | TTFT change | 0.28 different-rank slowdown |
|---:|---:|---:|---:|---:|---:|---:|
| 196 | 435.3 ms | 1.13x | 405.8 ms | 1.05x | -6.8% | 0.97x |
| 1,024 | 522.4 ms | 1.35x | 491.6 ms | 1.27x | -5.9% | 0.97x |
| 4,096 | 1,198.9 ms | 3.10x | 972.2 ms | 2.50x | -18.9% | 1.00x |
| 9,216 | 2,509.1 ms | 6.49x | 2,326.1 ms | 5.99x | -7.3% | 0.96x |

The unloaded 4K-prefill median was nearly identical: 386.75 ms on vLLM 0.25
and 388.29 ms on vLLM 0.28 with Omni enabled. This makes the slowdown ratios
directly comparable. The reduced same-rank slowdown is encouraging, especially
at 4,096 vision tokens, but two repetitions per cell are not sufficient to
assign the change to one implementation detail.

### Plugin ablation

| Vision tokens | 0.28 + Omni slowdown | 0.28 with Omni plugin disabled |
|---:|---:|---:|
| 196 | 1.05x | 1.10x |
| 1,024 | 1.27x | 1.31x |
| 4,096 | 2.50x | 2.28x |
| 9,216 | 5.99x | 6.05x |

The direction is inconsistent: enabling the plugin helped slightly in three
cells but hurt the 4,096-token cell. This is consistent with run-to-run overlap
and scheduling variance, and with the fact that no active Omni patch targets
the eager BF16 Qwen3.5 encoder/prefill path.

## 4. Encoder and Preprocessor Timing

| Vision tokens | 0.25 encoder | 0.28 + Omni encoder | Encoder change | 0.25 preprocessing | 0.28 + Omni preprocessing |
|---:|---:|---:|---:|---:|---:|
| 196 | 8.49 ms | 9.03 ms | +6.3% | 4.69 ms | 4.73 ms |
| 1,024 | 37.34 ms | 37.29 ms | -0.1% | 19.96 ms | 21.97 ms |
| 4,096 | 286.41 ms | 285.39 ms | -0.4% | 144.11 ms | 178.39 ms |
| 9,216 | 1,201.17 ms | 1,199.12 ms | -0.2% | 325.86 ms | 475.90 ms |

Medium through XLarge encoder forward time is unchanged within 0.4%. The newer
stack spends more CPU time preprocessing large inputs: +23.8% at 4,096 vision
tokens and +46.0% at 9,216. As a result, median cold end-to-end latency rose
from 767.7 to 808.6 ms for the 4,096-token input and from 2,324.8 to 2,463.9 ms
for the 9,216-token input. Hot-cache latency improved slightly for these sizes.

With the Omni plugin disabled, encoder medians were
`8.89 / 37.38 / 285.22 / 1,199.34 ms`. These are effectively the same as the
plugin-enabled values, confirming that the Omni plugin did not alter the vision
encoder kernel in this experiment. Preprocessor variation between the two 0.28
runs was larger than encoder variation, especially for large images.

## 5. Cache Locality versus Replica Load

The table reports:

```text
delta = TTFT(hot busy rank 0) - TTFT(cold idle rank 1)
```

Negative values favor cache locality; positive values favor moving the request
to the cold but idle replica.

| Vision tokens | Stack | Load 0 | Load 2 | Load 4 | Load 6 |
|---:|---|---:|---:|---:|---:|
| 196 | vLLM 0.25 | -16 ms | +560 ms | +1,222 ms | +1,856 ms |
| 196 | vLLM 0.28 + Omni | -7 ms | +569 ms | +1,234 ms | +1,875 ms |
| 1,024 | vLLM 0.25 | -116 ms | +505 ms | +1,176 ms | +1,864 ms |
| 1,024 | vLLM 0.28 + Omni | -80 ms | +526 ms | +1,176 ms | +1,843 ms |
| 4,096 | vLLM 0.25 | -507 ms | +87 ms | +746 ms | +1,421 ms |
| 4,096 | vLLM 0.28 + Omni | -466 ms | +158 ms | +816 ms | +1,481 ms |
| 9,216 | vLLM 0.25 | -1,652 ms | -1,111 ms | -411 ms | +317 ms |
| 9,216 | vLLM 0.28 + Omni | -1,430 ms | -896 ms | -213 ms | +441 ms |

The phase boundary is unchanged at the sampled load levels:

- 196- and 1,024-token images cross over between load 0 and load 2.
- The 4,096-token image also crosses over by load 2.
- The 9,216-token image remains locality-favored through load 4 and becomes
  load-balancing-favored at load 6.

The newer stack reduces cold-recompute TTFT for large and XLarge images in this
online experiment, narrowing the cache-locality advantage, but it does not
invalidate the need for a prefix-aware cost model.

## 6. GPU-to-GPU Copy Timing

| Vision tokens | 0.25 copy median | 0.28 copy median | Change |
|---:|---:|---:|---:|
| 196 | 0.06898 ms | 0.06969 ms | +1.0% |
| 1,024 | 0.22629 ms | 0.22555 ms | -0.3% |
| 4,096 | 0.81538 ms | 0.81740 ms | +0.2% |
| 9,216 | 1.79931 ms | 1.80161 ms | +0.1% |

P2P behavior is unchanged. GPU 0 and GPU 1 still report no direct peer access,
and the copies use the same host-mediated topology. This result is a data-plane
lower bound and excludes production connector and queueing overhead.

## 7. Implications for Agentrix

1. The original motivation for Prefix-aware DP survives the repository change.
   Cache affinity and rank load remain competing costs with an image-size-
   dependent crossover.
2. The new stack reduces measured same-rank interference but does not eliminate
   it. Large vision jobs can still multiply text TTFT by 2.5x to 6.0x.
3. The vision encoder itself is not materially faster. The interference change
   likely lies elsewhere in the newer upstream execution/scheduling stack or in
   measurement overlap, and should not be credited to vLLM-Omni without a
   higher-repetition profiler run.
4. Large-image CPU preprocessing regressed in this configuration. A production
   router should measure arrival at the GPU scheduler rather than relying only
   on HTTP request arrival when validating encode/prefill overlap.
5. vLLM-Omni can coexist with the planned operator and routing work, but this
   Qwen3.5 experiment does not provide a reason to couple ForkAttention or
   Prefix-aware DP to the Omni multi-stage scheduler.

## 8. Raw Data

- Baseline setup and interpretation:
  [multimodal_agent_dp_experiment_results.md](multimodal_agent_dp_experiment_results.md)
- vLLM 0.28 + Omni encoder:
  [mm_dp_vllm_omni_encoder_timing.csv](experiment_results/mm_dp_vllm_omni_encoder_timing.csv)
- vLLM 0.28 + Omni interference:
  [mm_dp_vllm_omni_interference.csv](experiment_results/mm_dp_vllm_omni_interference.csv)
- vLLM 0.28 + Omni cache/load sweep:
  [mm_dp_vllm_omni_cache_load_conflict.csv](experiment_results/mm_dp_vllm_omni_cache_load_conflict.csv)
- vLLM 0.28 + Omni P2P copy:
  [mm_dp_vllm_omni_p2p.csv](experiment_results/mm_dp_vllm_omni_p2p.csv)
- vLLM 0.28 plugin-disabled encoder ablation:
  [mm_dp_vllm_028_no_plugin_encoder_timing.csv](experiment_results/mm_dp_vllm_028_no_plugin_encoder_timing.csv)
- vLLM 0.28 plugin-disabled interference ablation:
  [mm_dp_vllm_028_no_plugin_interference.csv](experiment_results/mm_dp_vllm_028_no_plugin_interference.csv)

## 9. Limitations

- The sample is deliberately small: two repetitions per interference and
  cache/load cell and three cold encoder repetitions per image size.
- This comparison changes several components together: vLLM 0.25 to 0.28,
  PyTorch 2.11 to 2.13, Transformers, and the presence of vLLM-Omni.
- The plugin-disabled ablation isolates the active Omni entry point on vLLM
  0.28, but it does not isolate individual upstream commits between 0.25 and
  0.28.
- All performance runs used eager mode, so these results do not characterize
  the newer CUDA Graph or Inductor paths.
- Synthetic images control token geometry but are not a substitute for a
  production multimodal agent trace.
