# Qwen3-0.6B 100-Case ForkAttention and CPU-Offload Experiment

## Scope

This document contains only the current formal single-GPU result. Superseded
10-case, qualification, inactive-offload, and three-level-storage measurements
have been removed.

The experiment answers two separate questions:

1. With a large and identical GPU KV cache, how does ForkAttention compare
   with FlashAttention without offload?
2. Under a smaller and identical GPU KV cache, how does FlashAttention with
   original CPU offload compare with ForkAttention plus its fanout-optimized
   CPU-offload policy?

The two groups use different GPU KV capacities and must not be used to derive
a direct no-offload-versus-offload overhead. Filesystem storage is disabled.

## Corrected Closed-Loop Replay

The request trace is captured once and is not itself a performance sample.
Every formal variant replays the same request payloads and fixes each request
to its individually captured completion length.

Closed-loop replay keeps at most two complete Agent cases in flight. As soon
as one case finishes, the next captured case starts. It preserves case order,
planner-to-fanout-to-reducer dependencies, branch-specific tool delays,
request payloads, and output lengths without copying the capture backend's
absolute planner timestamps.

A captured forced function choice is normalized to `auto` while retaining the
same messages and tool schema. This is necessary because an ignore-EOS,
fixed-length decode cannot remain inside a finite forced-JSON grammar after a
valid JSON object terminates. The normalization is identical for all variants.

## Configuration

| Setting | Value |
|---|---|
| Validation date | 2026-07-17 |
| GPU | NVIDIA RTX 5070, 12 GiB |
| Model | Qwen3-0.6B, BF16 |
| Workload | Frozen HotpotQA long-prefix Agent manifest |
| Cases / sibling branches | 100 / 10 per case |
| LLM requests | 2,200 |
| Prompt / completion tokens | 26,463,283 / 233,089 in every variant |
| Maximum observed prompt | 18,206 tokens |
| Shared context in manifest | 7,871-14,311 tokens |
| Case / LLM concurrency | 2 / 20 |
| Context limit | 32,768 tokens |
| Maximum batched tokens / sequences | 16,384 / 32 |
| Prefix caching | Enabled |
| Async scheduling | Disabled |
| CUDA Graph | Enabled; Fork uses Prefix Forest CUDA Graphs |
| GPU-only physical KV | 7 GiB = 65,536 tokens for both backends |
| Offload physical GPU KV | 4 GiB = 37,440 tokens for both policies |
| CPU staging | 8 GiB pinned memory |
| Filesystem tier | Disabled |

All four rows completed 2,200 requests with identical token totals and no API,
OOM, CUDA Graph, or server failure.

## GPU-Only Backend Comparison

| Backend | Request wall | Request throughput | Prompt tok/s | Request latency P50 / P95 | GPU peak | Peak live KV | RSS peak | Preemptions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FlashAttention | 844.167 s | 2.606 req/s | 31,348.4 | 3,514.9 / 9,864.1 ms | 11,716 MiB | 65,136 | 3,838 MiB | 0 |
| ForkAttention | 594.853 s | 3.698 req/s | 44,487.1 | 2,219.7 / 5,664.4 ms | 11,657 MiB | 65,488 | 4,030 MiB | 3 |

ForkAttention is **1.419x faster** in request wall time and reduces wall time
by 29.53%. Both backends process exactly the same input and output token counts
and expose exactly 65,536 physical KV tokens.

### Prompt source and scheduler time

| Backend | Local prompt compute | Local cache hit | Cumulative prefill | Cumulative decode | Cumulative queue |
|---|---:|---:|---:|---:|---:|
| FlashAttention | 4,407,299 | 22,055,984 | 548.269 s | 8,501.529 s | 26.413 s |
| ForkAttention | 4,407,427 | 22,055,856 | 555.585 s | 4,692.988 s | 24.315 s |

The two backends differ by only 128 tokens between local compute and local
cache hit. Fork cumulative prefill is slightly slower, while cumulative decode
time falls by 44.80%. The result is therefore consistent with a decode-path
gain rather than a prefix-cache fairness difference.

After warm-up subtraction, ForkAttention was observed for 28,087 execution
steps and physically shared at least one CTA in 22,820 steps, an 81.25%
activation rate. It accumulated 178,322 shared and 598,181 singleton CTA-plan
entries. The shared execution path was active rather than silently falling
back to ordinary attention.

## Two-Level CPU-Offload System Comparison

Only two offload systems are included:

- FlashAttention with the original whole-batch CPU-offload policy;
- ForkAttention with the fanout-optimized chunked CPU-offload policy.

Both rows are `active`, completed all requests, drained after the final
response, and recorded zero `cannot store blocks` admission failures.

| System | Request wall | Wall plus active drain | Speedup | Prompt tok/s | GPU peak | Peak live KV | CPU cache peak | RSS peak | Preemptions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flash + original CPU offload | 858.133 s | 858.168 s | 1.000x | 30,838.2 | 9,842 MiB | 37,440 | 100% | 12,051 MiB | 19 |
| Fork + optimized CPU offload | 697.426 s | 697.457 s | **1.230x** | 37,944.2 | 9,947 MiB | 37,440 | 100% | 12,263 MiB | 24 |

Fork plus optimized offload reduces request wall time by 18.73%.

### CPU-GPU KV traffic

| System | GPU to CPU | Store ops / average | CPU to GPU | Load ops / average | Store / load time |
|---|---:|---:|---:|---:|---:|
| Flash + original | 469.588 GiB | 2,200 / 218.572 MiB | 52.444 GiB | 162 / 331.495 MiB | 9.832 / 1.437 s |
| Fork + optimized | 464.839 GiB | 5,920 / 80.405 MiB | 59.770 GiB | 198 / 309.114 MiB | 11.825 / 1.690 s |

The optimized Fork policy stores smaller chunks, but total GPU-to-CPU bytes
fall by only 1.01%; CPU-to-GPU bytes increase by 13.97%. The 1.230x system gain
is not evidence of a large PCIe-volume reduction. Because the attention
backend and offload policy change together, the comparison also cannot isolate
the offload-policy contribution.

### Prompt source and scheduler time

| System | Local prompt compute | Local cache hit | External KV | Cumulative prefill | Cumulative decode | Cumulative queue |
|---|---:|---:|---:|---:|---:|---:|
| Flash + original | 4,413,059 | 21,565,904 | 484,320 | 685.270 s | 7,778.142 s | 477.888 s |
| Fork + optimized | 4,403,891 | 21,552,432 | 506,960 | 649.472 s | 5,265.172 s | 329.338 s |

Fork reduces cumulative decode by 32.31%, cumulative queue by 31.08%, and
cumulative prefill by 5.22%. It physically activated on 24,112 of 29,596
measured steps (81.47%), with 178,496 shared and 601,231 singleton CTA-plan
entries.

## Interpretation and Limitations

- The 1.419x GPU-only result is a LangGraph-derived closed-loop inference
  replay on a deliberately favorable long-prefix, ten-way-fanout workload. It
  is neither a complete live LangGraph application result nor a universal
  operator speedup.
- The 1.230x offload result is a system comparison between two backend-policy
  combinations. It does not isolate the offload-policy effect.
- Connector counters quantify CPU-GPU KV traffic. No Nsight Compute HBM-read
  counter was collected, so this experiment makes no quantitative HBM KV-load
  claim.
- Each row currently has one formal repeat. Publication-quality reporting
  should randomize variant order and collect at least three repetitions.

Run JSON, server logs, Prometheus snapshots, memory samples, drain records,
and generated reports for these four current rows are retained. The document
contains no host-specific paths.
