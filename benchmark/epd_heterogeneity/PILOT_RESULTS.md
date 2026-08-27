# Two-H20 E/PD pilot results

Date: 2026-08-27

These results validate the experiment harness. They are not formal benchmark
numbers: each configuration ran once, shape-specific warm-up was incomplete,
and `nvidia-smi` polling can miss kernels from short requests.

## Environment

- Model: local Qwen2.5-VL-7B-Instruct, BF16
- vLLM source: exact `v0.26.0` tag
- GPU 0: encoder-only, eager mode, prefix cache disabled
- GPU 1: prefill+decode, eager mode, prefix cache disabled
- Encoder cache connector: `ECExampleConnector` on `/dev/shm`
- Controlled visual-token buckets: 256, 1024, 2304, and 9216

The encoder-only instance occupied about 2.6 GiB at startup. The PD instance
occupied about 117 GiB, most of which was its configured KV-cache reservation.

## A0 validation

Four simultaneous 2304-visual-token requests all completed successfully. The
proxy sent each request through the dedicated encoder before forwarding it to
PD. The encoder portion measured by the proxy was about 0.94 seconds for the
batch; client latency ranged from 3.72 to 4.13 seconds with up to 128 output
tokens.

This encoder interval includes queueing, compute, D2H, serialization, shared
memory storage, and publication. It is not pure GPU encoder time.

## Encoder pilot

Output length was fixed at one token.

| Visual tokens | Concurrency | Client mean (ms) | Encoder path mean (ms) |
|---:|---:|---:|---:|
| 256 | 1 | 129 | 59 |
| 256 | 4 | 410 | 257 |
| 1024 | 1 | 335 | 138 |
| 1024 | 4 | 742 | 336 |
| 2304 | 1 | 241 | 152 |
| 2304 | 4 | 1429 | 626 |
| 9216 | 1 | 2571 | 1120 |
| 9216 | 4 | 5903 | 3128 |

The overall sensitivity is clear, but the non-monotonic 1024/2304
single-request latency demonstrates why repetitions and shape-specific warm-up
are required.

## Text prefill/decode pilot

Text requests used prompt token IDs directly, so server-reported prompt length
matched the requested length exactly.

- Prefill throughput reached roughly 8.3k prompt tokens/s at 16k input tokens,
  for both concurrency 1 and 8.
- Decode with 128 context and concurrency 8 reached about 587 output tokens/s.
- Decode with 4096 context, 1024 output tokens, and concurrency 8 reached about
  474 output tokens/s.
- GPU 0 remained idle during all pure-text cases.

These are aggregate output-throughput measurements, not token-level TPOT.

## Temporal pilot

Each run contained 64 requests across eight sessions with up to 64 output
tokens per step.

| Target p(E) | Actual visual requests | Encoder avg util | PD avg util | E idle / PD busy |
|---:|---:|---:|---:|---:|
| 0.10 | 3/64 | 1.8% | 65.4% | 49.5% |
| 0.50 | 33/64 | 16.3% | 82.8% | 45.0% |

`E idle / PD busy` uses the provisional threshold `E < 20% and PD > 60%`.
The signal supports continuing the temporal-mismatch study, but formal results
need at least three repetitions and a profiler that can observe short kernels.

## Measurement gap

The host currently has no DCGM, Nsight Systems, or Nsight Compute executable.
At 100 ms client-side polling, `nvidia-smi` missed some 130--240 ms requests and
reported a false 0% utilization. Formal resource-signature figures must use
long steady-state windows or install a finer-grained profiler.
