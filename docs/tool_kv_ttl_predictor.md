# Tool KV TTL Predictor

## Purpose and Safety Boundary

The TTL predictor reduces unnecessary KV retention during long tool calls
without turning model output into an unconditional eviction command. It only
chooses when a session becomes eligible for the existing pressure-gated trim;
the live vLLM pressure check, serialized trim transaction, request lifecycle
validation, and fixed fallback remain authoritative.

The predictor is application-owned and introduces no additional vLLM types or
endpoints. It stores derived features only and does not retain raw commands,
paths, queries, or tool arguments.

## Model

`OnlineHorizonTTLPredictor` maintains six online logistic classifiers for:

```text
P(duration > 100 ms),  P(duration > 250 ms),
P(duration > 500 ms),  P(duration > 1,000 ms),
P(duration > 2,000 ms), P(duration > 5,000 ms)
```

Inputs are tool family, argument byte bucket, KV-token estimate, current KV
pressure, active tool-session count, shared-prefix ratio, and timeout. Feature
hashing bounds model state at 128 dimensions per horizon. The independent
outputs are projected to a non-increasing survival curve.

The model starts with a 500-ms fixed TTL. After 50 observations, a tool that is
confidently predicted to remain active becomes eligible sooner, bounded at 100
ms. Prediction can only shorten the fixed TTL; it cannot create an unbounded
retention period. Every completed tool call supplies an unbiased duration
label whether or not its KV was trimmed.

## Accuracy Experiment

The standalone evaluator uses a 70/30 held-out split. The controlled dataset
contains 2,000 deterministic samples across five long-tail regimes: read,
search, network, public test, and build. Durations are sampled from fixed
log-normal distributions with medians of 45, 220, 900, 4,000, and 8,000 ms.
The random seed is 2026; 1,400 samples train the online model and 600 are never
used for updates.

| Metric | Held-out result |
|---|---:|
| Macro accuracy across six horizons | 94.81% |
| Macro Brier score, lower is better | 0.0361 |
| Shorten/fallback decision accuracy | 91.50% |
| Exact TTL bucket accuracy | 88.67% |
| TTL within one adjacent bucket | 100.00% |

Five independent held-out seeds, 2026-2030, show that the result is stable:

| Metric | Five-seed mean | Min-max |
|---|---:|---:|
| Macro horizon accuracy | 95.25% | 94.67%-95.78% |
| Macro Brier score | 0.0349 | 0.0327-0.0370 |
| Shorten/fallback decision accuracy | 92.73% | 91.50%-93.67% |
| Exact TTL bucket accuracy | 90.33% | 88.67%-91.83% |
| TTL within one adjacent bucket | 99.90% | 99.67%-100.00% |

Per-horizon results are:

| Horizon | Positive rate | Accuracy | Precision | Recall | Brier |
|---:|---:|---:|---:|---:|---:|
| 100 ms | 78.67% | 99.67% | 99.58% | 100.00% | 0.0042 |
| 250 ms | 67.50% | 91.50% | 100.00% | 87.41% | 0.0503 |
| 500 ms | 56.17% | 97.17% | 95.20% | 100.00% | 0.0259 |
| 1,000 ms | 46.50% | 89.50% | 86.99% | 91.04% | 0.0563 |
| 2,000 ms | 35.83% | 98.33% | 96.38% | 99.07% | 0.0170 |
| 5,000 ms | 17.50% | 92.67% | 80.20% | 77.14% | 0.0631 |

The repository's existing real trace contains 1,100 timed tool events, but all
are `paragraph_search` calls shorter than 100 ms. The predictor correctly keeps
the 500-ms fallback for every held-out event, producing 100% accuracy, but this
single-class result is not evidence of generalization. The controlled
multi-regime result is the meaningful algorithm test; production activation
still requires shadow data from real read/search/test/build/network tools.

## Reproduction

Run all predictor and trimmer tests:

```bash
PYTHONPATH=application/src vllm/.venv/bin/python -m pytest application/tests -q
```

Run the controlled held-out evaluation:

```bash
vllm/.venv/bin/python \
  benchmark/scripts/evaluate_tool_ttl_predictor.py \
  --dataset synthetic \
  --output benchmark/results/tool_ttl_predictor/synthetic_accuracy.json
```

Evaluate timed tool events already present in benchmark `run.json` files:

```bash
vllm/.venv/bin/python \
  benchmark/scripts/evaluate_tool_ttl_predictor.py \
  --dataset trace \
  --trace-root benchmark/results \
  --output benchmark/results/tool_ttl_predictor/trace_accuracy.json
```

The recommended deployment sequence is fixed TTL, predictor shadow mode, then
active predicted TTL only after real horizon calibration and KV-peak/resumption
measurements confirm a net benefit.
