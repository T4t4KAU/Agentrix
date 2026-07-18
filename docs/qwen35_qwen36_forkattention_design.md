# Qwen3.5/Qwen3.6 ForkAttention Implementation and Design

## 1. Goal and outcome

This document describes the Agentrix/vLLM ForkAttention adaptation for the
following locally deployed models:

- `Qwen3.5-27B`
- `Qwen3.6-27B`

Although the model directories use different external version names, both
configurations resolve to `Qwen3_5ForConditionalGeneration` with the internal
model type `qwen3_5`. Their language-model attention geometry is identical, so
they share one CUDA implementation and one backend-selection path. There are no
model-name-specific kernels.

The adapted runtime assigns work as follows:

| Model component | Implementation | Handled by ForkAttention |
|---|---|---|
| Vision Transformer | FlashAttention | No |
| 48 GDN/linear-attention layers | FlashInfer GDN | No |
| Prefill in 16 full-attention layers | FlashAttention fallback | No |
| Shared-prefix decode in 16 full-attention layers | ForkAttention | Yes |
| Multi-split output reduction | ForkAttention gather kernel | Yes |

The objective is broader than an isolated attention-operator speedup. In a
multimodal agent workload, multiple action branches often share the large visual
token prefix produced from the same webpage screenshot. Reusing that prefix can
reduce duplicate KV-page traffic and duplicate attention work across branches.

## 2. Model configuration and compatibility boundary

The two 27B checkpoints on the server have the same core geometry:

| Configuration | Qwen3.5-27B | Qwen3.6-27B |
|---|---:|---:|
| Architecture | `Qwen3_5ForConditionalGeneration` | `Qwen3_5ForConditionalGeneration` |
| Dtype | BF16 | BF16 |
| Hidden layers | 64 | 64 |
| GDN/linear layers | 48 | 48 |
| Full-attention layers | 16 | 16 |
| Query heads | 24 | 24 |
| KV heads | 4 | 4 |
| GQA ratio | 6 | 6 |
| Head dimension | 256 | 256 |
| Full-attention interval | 4 | 4 |
| Vision encoder | Yes | Yes |

The `config.json` differences are mainly newer explicit defaults and a newer
Transformers version. Examples include `language_model_only`, `bos_token_id`,
`output_gate_type`, and `partial_rotary_factor`. These differences do not change
the Q/K/V geometry consumed by ForkAttention.

### 2.1 Why this is not the `mm_prefix` backend mode

Qwen3.5/3.6 inserts visual embeddings into the decoder's causal token sequence.
ForkAttention therefore sees them as an ordinary causal shared prefix. The
vLLM backend flag `use_mm_prefix` instead denotes a multimodal prefix that needs
a bidirectional PrefixLM mask; it does not simply mean that visual tokens are
present.

Consequently:

- Qwen's causal visual prefix can use ForkAttention.
- Models that require a bidirectional `mm_prefix` mask must continue to fall
  back instead of being forced onto ForkAttention.

## 3. End-to-end data flow

```text
raw screenshot/video
        |
        v
multimodal hash + encoder cache
        |
        v
ViT / MM encoder (FlashAttention)
        |
        v
causal visual embeddings + text prompt
        |
        +------------------------------+
        | 48 GDN layers                | recurrent-state cache, align mode
        +------------------------------+
        | 16 full-attention layers     | paged KV cache
        +------------------------------+
                         |
             shared physical KV pages
                         |
          +--------------+--------------+
          |              |              |
       branch A       branch B       branch N
          \              |              /
           ForkAttention forest + gather
```

Three layers jointly enforce safe sharing:

1. The multimodal encoder cache uses `mm_hash` to identify identical image or
   video inputs.
2. The KV block hash includes multimodal identity and position in the prefix
   cache key, preventing different screenshots with identical placeholder
   tokens from being shared incorrectly.
3. The `PrefixAwareDPRouter` namespace also includes multimodal identity. This
   gives identical visual prefixes affinity to the same DP replica without
   creating false cache affinity between different screenshots.

## 4. ForkAttention implementation changes

### 4.1 Python backend capability declaration

File: `vllm/v1/attention/backends/fork_attn.py`

- Extend supported head dimensions from `64/128` to `64/128/256`.
- Make the metadata builder use the backend's canonical head-size check.
- Enable head dimension 256 only on compute capability 9.0 or newer.
- Keep explicit FlashAttention fallback reasons for all other unsupported
  combinations.

The largest head-256 tile needs approximately 160 KiB of dynamic shared memory.
The backend therefore does not advertise this path on Ampere or Ada. The server
uses H20 GPUs (SM90), which satisfy this constraint.

### 4.2 C++ entry point and template instantiations

Relevant files:

- `vllm/CMakeLists.txt`
- `vllm/csrc/libtorch_stable/attention/fork/fork_attention.cu`
- `vllm/csrc/libtorch_stable/attention/fork/fork_fwd_split_hdim256_fp16.cu`
- `vllm/csrc/libtorch_stable/attention/fork/fork_fwd_split_hdim256_bf16.cu`

The C++ custom operator now validates and dispatches head dimensions
`64/128/256`. CMake explicitly builds the FP16 and BF16 head-256 template
instantiations so the primary kernels do not depend on runtime JIT compilation
or fail with a missing symbol.

Qwen uses a GQA ratio of 6. Existing dispatch decomposes its query-head group
into `4 + 2`, so no ratio-6-specific MMA layout is required.

### 4.3 Head-256 gather correction

File: `vllm/csrc/libtorch_stable/attention/fork/fork_fwd_launch_template.h`

The first H20 numerical test found incorrect output beginning at head coordinate
128. The old gather launch used four warps for every non-64 head dimension, so a
head-256 launch had only 128 threads. The final store maps `tid` directly to a
head coordinate, leaving coordinates 128 through 255 unwritten.

The corrected launch mapping is:

| Head dimension | Gather warps | Threads |
|---:|---:|---:|
| 64 | 2 | 64 |
| 128 | 4 | 128 |
| 256 | 8 | 256 |

This change makes the head-256 path numerically complete, rather than merely
compilable: every output coordinate participates in split reduction and the
final store.

## 5. Hybrid KV/GDN cache design

Qwen3.5/3.6 is not a pure Transformer. Prefix caching must preserve both:

- paged K/V for full-attention layers;
- recurrent state at the prefix boundary for GDN layers.

When prefix caching is enabled, vLLM selects `mamba_cache_mode=align` for this
architecture. The Qwen3.5 model implementation supplies
`get_mamba_state_copy_func()` to copy recurrent state at a reusable boundary.
The runtime enlarges the scheduler/attention page until it can hold at least one
Mamba/GDN state page.

The H20 run selected an attention block size of 784 tokens:

- 784 is divisible by ForkAttention's required 16-token granularity.
- The full-attention KV page and GDN state page are exactly aligned.
- A prefix cache hit is accepted only at an aligned boundary, preventing K/V
  and recurrent state from referring to different history positions.

This makes the reusable-prefix granularity coarser than the 16-token page used
by a pure Transformer. WebLINX samples should therefore have a stable
screenshot/history prefix substantially longer than one 784-token block;
otherwise alignment loss can hide the benefit of sharing.

## 6. Sources of benefit in a multimodal agent

When several candidate actions branch from one webpage state, reuse occurs in
two places:

1. **Encoder reuse.** Identical screenshots hit the encoder cache through
   `mm_hash`, reducing repeated ViT computation.
2. **Decoder-prefix reuse.** Identical visual embeddings and text history map
   to the same physical KV blocks. The ForkAttention forest processes shared
   pages once and combines them with each branch's private suffix.

Experiments must record metrics from both sides. ForkAttention kernel time alone
misses visual-encoder cache gains, while aggregate end-to-end throughput alone
cannot distinguish scheduling, GDN, ViT, and full-attention contributions.

At minimum, record:

- end-to-end request latency and aggregate tokens/s;
- multimodal encoder-cache hit rate;
- prefix-cache hit rate;
- `eager_forest`/`cudagraph_forest` hit counts and fallback reasons;
- full-attention kernel time and memory traffic;
- per-replica request count, cache affinity, and load imbalance.

## 7. Launch configuration

Single-GPU functional validation example:

```bash
PROFILE_FORK=1 \
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/vllm serve /test__02/hwx/Qwen3.5-27B \
  --attention-backend FORK_ATTN \
  --enable-prefix-caching \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.65 \
  --enforce-eager
```

Replace the model path with `/test__02/hwx/Qwen3.6-27B` for Qwen3.6. Use the
`--attention-backend FORK_ATTN` CLI argument; the current code does not recognize
`VLLM_ATTENTION_BACKEND` as a valid vLLM environment variable.

`PROFILE_FORK=1` is diagnostic only. A successful shared-decode dispatch emits:

```text
ForkAttention profile: ... path=eager_forest:enabled ...
```

Production performance tests should cover eager and CUDA Graph configurations
separately. The eager smoke-test throughput is not a production benchmark.

## 8. Validation matrix

### 8.1 CUDA operator

The CUDA tests used FlashAttention as the reference on H20/SM90:

| Geometry | Dtype | Result |
|---|---|---|
| Head-128 regression | FP16 | Pass |
| 24 Q / 4 KV / head-256 | FP16 | Pass |
| 24 Q / 4 KV / head-256 | BF16 | Pass |

The tolerance was `atol=2e-2, rtol=2e-2`. These tests include a shared prefix,
private suffixes, split output, and gather; they do more than check custom-op
registration.

### 8.2 Qwen3.5/Qwen3.6 end-to-end smoke tests

Both models completed a single-GPU smoke test:

- The checkpoints loaded successfully (51.1 GiB reported for Qwen3.5 and
  51.75 GiB for Qwen3.6).
- Full-attention layers selected `FORK_ATTN`.
- ViT selected FlashAttention and GDN selected FlashInfer.
- Multimodal warmup completed.
- Eight concurrent requests sharing a 1,514-token prompt all succeeded.
- The log repeatedly reported `eager_forest:enabled`.
- Both runs reported a 45.3% prefix-cache hit rate.
- The request-group wall times were 3.039 seconds for Qwen3.5 and approximately
  3.04 seconds for Qwen3.6.

The Qwen3.5 run additionally confirmed automatic attention-page alignment to
784 tokens, eight HTTP 200 responses, and release of all eight H20 GPUs to 1 MiB
reported usage after shutdown. Both models resolved to the internal architecture
`Qwen3_5ForConditionalGeneration`, confirming that they use the shared
head-256/GQA6 backend path without model-name branching.

These are functional smoke tests, not a ForkAttention-versus-FlashAttention
performance conclusion.

### 8.3 Acceptance criteria

Both models should satisfy all of the following:

1. The config resolves to `Qwen3_5ForConditionalGeneration`.
2. The log selects `FORK_ATTN` for full-attention layers.
3. Text input and at least one real-image request complete generation.
4. Two or more shared-prefix branches trigger the forest path.
5. Output contains no CUDA error, NaN, or truncation at head coordinate 128.
6. GPU memory is released normally after service shutdown.

Criterion 3's real-image portion remains part of the WebLINX validation; the
completed end-to-end smoke tests used a synthetic shared text prefix.

## 9. WebLINX and 6/8-replica DP experiment design

WebLINX fits this workload because several candidate actions from the same
interaction turn can share:

- the current webpage screenshot;
- DOM and action-history text;
- the system prompt and tool definitions.

Hold total request count, output-token count, image resolution, and 32K context
scale constant, changing only the backend and DP replica count. The minimum
matrix should include:

| Variable | Values |
|---|---|
| Model | Qwen3.5-27B, Qwen3.6-27B |
| Backend | FlashAttention baseline, ForkAttention |
| DP | 6, 8 |
| Context | 32K |
| Branches per shared state | 2, 4, 8, 16 |
| Screenshot reuse | same-image, different-image control |

The `same-image` group measures real visual-prefix reuse. The
`different-image control` should keep visual-token counts similar while using a
different `mm_hash`; this separates genuine visual-prefix reuse from apparent
speedups caused only by prompt length or batching.

## 10. Current limitations

- ForkAttention currently handles causal decode only, with `q_len == 1`; prefill
  falls back.
- The head-256 path requires SM90 or newer.
- ForkAttention does not accelerate GDN layers; their reuse depends on the
  align-mode state cache.
- vLLM still marks hybrid prefix caching as experimental.
- The completed shared-prefix smoke tests used a synthetic text prefix. A final
  claim requires real WebLINX screenshots.
- Correctness and performance at 32K with 6/8 DP replicas require a separate
  experiment.
