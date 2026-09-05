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

The WebLINX validation below supplies the real-image coverage required by
criterion 3; the earlier single-GPU smoke tests used a synthetic shared text
prefix.

## 9. WebLINX 8-DP workload and validation

WebLINX fits this workload because several candidate actions from the same
interaction turn can share:

- the current webpage screenshot;
- DOM and action-history text;
- the system prompt and tool definitions.

### 9.1 Deterministic subset

`benchmark/src/weblinx_data.py` builds a small manifest from the WebLINX
validation split. It selects eight distinct demonstrations that have a
supported element action, at least eight ranked candidates, the ground-truth
UID in the top eight, and a replay turn with a good screenshot. It excludes
contexts containing password fields, downloads only the selected replay and
screenshot files, normalizes each image to 1280x720, and records the image
SHA-256 in `manifest.json`.

The current seed-2026 subset contains eight turns and eight candidates per
turn. The pressure workload expands every candidate into four independent
evaluation rollouts, producing 32 branches per root and 256 branch requests.

### 9.2 Multimodal request layout

`benchmark/src/weblinx_runner.py` sends native OpenAI multimodal chat messages.
Each root first issues one 64-token common-analysis request over its screenshot,
DOM, conversation, and action history. That generated assistant analysis is
then part of the shared causal context for all 32 branches. Every candidate
forms a four-rollout subgroup with a candidate-shared suffix and one of four
private evaluation strategies. Seeded lognormal suffix budgets are rescaled to
a mean of 256 tokens, matching the text Pressure32K/32 construction.

All 256 branches are deterministically shuffled and submitted together. The
client sends no `X-data-parallel-rank` header, so the internal-DP server owns
placement. Generation uses `ignore_eos` so all variants produce the same
number of output tokens. The runner retains a different-image cache-identity
mode for diagnostics, but it is not part of the primary DP matrix.

Unlike the text-only Pressure32K/32 workload, which constructs each shared
root by repeating or truncating textual Agent task content, the WebLINX
workload derives every root from a real browser interaction state. Its shared
causal prefix contains a webpage screenshot together with the DOM,
conversation, action history, and generated common analysis. Eight ranked
action candidates are selected from each WebLINX turn, and every candidate is
expanded into four evaluation rollouts to preserve the same 32-branch fanout
shape. Prefix identity and reuse consequently depend on both textual KV state
and visual-input identity, including multimodal processor caching. The text
portion is fitted to approximately 28K tokens rather than 32,768 tokens so
that image tokens, chat-template framing, branch suffixes, and generated output
remain within the 32K context limit.

### 9.3 Reproduction

```bash
cd benchmark
.venv/bin/python -m weblinx_data \
  --output-dir results/weblinx_subset \
  --split validation --case-count 8 --branch-count 8 --seed 2026

MODEL_PATH=/test__02/hwx/Qwen3.6-27B \
OUTPUT_TOKENS=256 \
NUM_GPU_BLOCKS_OVERRIDE=84 \
VARIANTS="flash_ordinary fork_ordinary fork_prefix_aware" \
./scripts/run_weblinx_8dp.sh
```

The script fixes DP to eight replicas, TP to one, one API frontend, 256
concurrent branch requests, `max_num_seqs=64`, prefix caching, Forest CUDA
Graphs, and a 32K model limit. Flash and ordinary Fork use native internal-DP
placement. Only `fork_prefix_aware` enables the prefix router, the 10 ms arrival
wave, and fanout scheduling. All arms receive identical roots, suffix budgets,
shuffle order, output limits, and physical KV capacity.

For the validated Qwen3.6/H20 build, 84 hybrid KV blocks expose 57,344 tokens
per rank. This deliberately admits one approximately 31K root plus its 32
branches while preventing two independent roots from residing together. The
script writes raw request traces, CSV/Markdown summaries, server logs, and
Prometheus metrics for each variant.

### 9.4 Corrected Pressure32K/32 result

The corrected experiment completed all eight multimodal bootstrap requests and
all 256 branch requests in every arm. Each arm generated exactly 512 bootstrap
tokens and 65,536 branch tokens. Branch prompt lengths were 29,391 to 32,124
tokens for FlashAttention and 29,391 to 32,076 tokens for both ForkAttention
arms, so every request remained below the 32K model limit. No arm reported an
OOM, request failure, or server error.

| Variant | Bootstrap | Branch wall | Total wall | Branch output tok/s | Total output tok/s | Mean branch latency | P50 TTFT | P95 TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FlashAttention, ordinary DP | 14.831 s | 528.169 s | 542.999 s | 124.08 | 120.69 | 247.141 s | 231.231 s | 457.414 s |
| ForkAttention, ordinary DP | 14.396 s | 546.274 s | 560.670 s | 119.97 | 116.89 | 256.673 s | 251.341 s | 476.821 s |
| ForkAttention, prefix-aware DP | 14.853 s | 148.652 s | 163.505 s | 440.87 | 400.82 | 49.989 s | 37.214 s | 105.461 s |

Relative to FlashAttention with ordinary DP placement, prefix-aware
ForkAttention increased branch throughput by 3.55x, shortened the branch phase
by 71.86%, and increased end-to-end throughput by 3.32x. Mean branch latency,
P50 TTFT, and P95 TTFT fell by 79.77%, 83.91%, and 76.94%, respectively.
Relative to ordinary ForkAttention placement, the branch-throughput increase
was 3.67x. Ordinary ForkAttention was 3.31% slower than FlashAttention in
branch throughput, confirming that the large result comes from combining
shared-prefix execution with DP placement and residency rather than from an
uncontrolled backend-only comparison.

The server-side counters support that interpretation:

| Counter | FlashAttention, ordinary DP | ForkAttention, ordinary DP | ForkAttention, prefix-aware DP |
|---|---:|---:|---:|
| Prompt tokens observed | 8,006,279 | 8,004,679 | 8,004,679 |
| Prompt tokens computed locally | 5,812,647 | 5,696,583 | 869,495 |
| Prefix-cache hit share | 27.40% | 28.83% | 89.14% |
| Cumulative prefill time | 3,042.403 s | 3,102.548 s | 797.192 s |
| Cumulative queue time | 55,977.823 s | 58,237.952 s | 7,530.646 s |
| Preemptions | 57 | 64 | 27 |

Against FlashAttention, the optimized arm therefore reduced locally computed
prompt tokens by 85.04%, cumulative prefill time by 73.80%, and cumulative
queue time by 86.55%. Its multimodal processor cache recorded 256 hits from
264 queries; the eight misses correspond to the first bootstrap for each
independent image. The prefix router recorded 256 affinity routes and eight
long-prefix bootstrap routes, with exactly 33 total requests routed to every
rank. The client supplied no rank headers.

This is a single-run systems result, not a claim about model quality or pure
attention-kernel speed. Its tight 84-block hybrid KV configuration still
caused 27 preemptions in the optimized arm, although that was fewer than both
ordinary-DP arms. Qwen3.6's 784-token hybrid cache pages make the capacity
boundary coarser than in the earlier text-only experiment. The optimized
arm's P50 time per output token also rose from 18.256 ms to 32.181 ms because
many more branches reached decode concurrently; the much lower queue and
prefill times dominate the end-to-end result. A capacity sweep and repeated
runs are required for confidence intervals and a zero-preemption operating
point.

The validated artifacts are stored on the experiment server under
`benchmark/results/weblinx_pressure32k_8dp_v1`; generated datasets and result
files remain Git-ignored.

### 9.5 Initial pinned-rank diagnostic

The formal eager-mode run used 64 requests, 256 forced output tokens per
request, and identical observed prompt lengths of 29,344 to 30,663 tokens in
all variants. All three variants completed 64 requests and exactly 16,384
output tokens.

| Variant | Shared-state warmup | Branch wall | Total wall | Total output tok/s | Mean request latency |
|---|---:|---:|---:|---:|---:|
| FlashAttention, same image | 13.916 s | 18.663 s | 32.578 s | 502.91 | 16.805 s |
| ForkAttention, same image | 13.923 s | 18.730 s | 32.652 s | 501.77 | 17.063 s |
| ForkAttention, different images | 0.000 s | 120.848 s | 120.848 s | 135.58 | 117.117 s |

The warmup is the explicit common-state phase of the agent workflow: one
request per DP rank ingests the webpage state before its eight action branches.
It is included in total wall time. The different-image control has no reusable
common visual state and therefore no warmup request.

Within ForkAttention, reusing the visual prefix reduced total wall time by
72.98% and raised end-to-end output throughput by 3.70x relative to the
different-image control. The branch phase alone was 84.50% shorter. Same-image
ranks reported 86.4%-88.3% prefix-cache hits and up to 100% multimodal-cache
hits; the different-image control reported 0% for both. The ForkAttention log
contained 51 profile records reporting `eager_forest:enabled` for the
shared-image workload and none for the control.

FlashAttention and ForkAttention were effectively tied on the same-image
workload; ForkAttention's total wall time was 0.23% higher in this single run.
This was a useful multimodal cache diagnostic, but not a reproduction of the
text DP result: the client pinned both backends to the correct rank, explicitly
disabled prefix-aware routing, warmed the exact final prefix, used only eight
branches per rank, limited each rank to 16 sequences, and ran eager mode. Those
controls removed the routing, residency, and admission effects that dominate
the Pressure32K/32 DP result. The corrected three-arm experiment above replaces
that setup for DP performance claims.

## 10. Current limitations

- ForkAttention currently handles causal decode only, with `q_len == 1`; prefill
  falls back.
- The head-256 path requires SM90 or newer.
- ForkAttention does not accelerate GDN layers; their reuse depends on the
  align-mode state cache.
- vLLM still marks hybrid prefix caching as experimental.
- The current WebLINX workload validates eight DP replicas only; six-DP scaling
  and larger branch-count sweeps remain separate experiments.
- The reported WebLINX comparison is one run per arm. Capacity sweeps and
  repeated trials are still needed to quantify variance and remove the
  remaining optimized-arm preemptions.
