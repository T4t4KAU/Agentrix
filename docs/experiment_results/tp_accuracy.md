# Agentrix Main Experiment Results

## Performance and Resource Metrics

| Model | Dataset | Prefix | Branches | Variant | Output tok/s | TTFT P50/P95/P99 ms | TPOT P50/P95/P99 ms | Latency P50/P95/P99 ms | GPU compute | Memory BW | Peak GPU KV GiB (%) | Peak total KV GiB | Logical KV read reduction |
|---|---|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|
| qwen3-14b | agencybench | 16384 | 32 | flash_tp | 383.10 | 5380.65/10928.59/11380.67 | 45.34/49.25/69.16 | 8459.13/13173.98/13253.74 | 97.25% | 30.03% | 11.43 (52.20%) | 11.43 | 0.00% |
| qwen3-14b | agencybench | 16384 | 32 | fork_tp_run1 | 322.86 | 5993.79/13889.89/14323.28 | 58.70/64.80/76.62 | 9695.27/17103.07/17231.58 | 83.42% | 24.95% | 11.41 (52.11%) | 11.41 | 96.88% |
| qwen3-14b | agencybench | 16384 | 32 | fork_tp_run2 | 323.53 | 6012.49/13838.03/14025.32 | 58.86/62.75/76.41 | 9758.00/16575.79/16785.88 | 83.39% | 26.63% | 11.49 (52.43%) | 11.49 | 96.88% |
| qwen3-14b | agentboard | 16384 | 32 | flash_tp | 380.96 | 5433.35/11119.83/11456.38 | 44.27/49.09/70.66 | 8540.04/13132.63/13333.94 | 96.49% | 35.47% | 11.65 (56.43%) | 11.65 | 0.00% |
| qwen3-14b | agentboard | 16384 | 32 | flash_tp_repeat | 374.66 | 5500.67/11559.46/11618.57 | 46.35/50.48/70.56 | 8500.87/13499.88/13554.00 | 97.26% | 36.48% | 11.70 (53.39%) | 11.70 | 0.00% |
| qwen3-14b | agentboard | 16384 | 32 | fork_tp_run1 | 335.18 | 5956.62/13800.08/14160.89 | 50.48/63.56/77.12 | 9831.51/17002.39/17046.17 | 85.67% | 31.69% | 11.75 (56.91%) | 11.75 | 95.51% |
| qwen3-14b | agentboard | 16384 | 32 | fork_tp_run2 | 342.06 | 5880.26/13297.82/13801.83 | 49.30/62.02/77.20 | 9449.64/16425.77/16634.58 | 86.66% | 32.03% | 11.59 (52.90%) | 11.59 | 95.51% |
| qwen3-14b | appworld | 16384 | 32 | flash_tp | 375.27 | 5522.42/11959.95/12177.22 | 46.66/53.13/69.35 | 8439.04/13976.43/13989.45 | 97.61% | 31.58% | 11.46 (52.30%) | 11.46 | 0.00% |
| qwen3-14b | appworld | 16384 | 32 | fork_tp_run1 | 325.52 | 5779.24/13503.59/14139.67 | 57.40/64.03/76.10 | 9503.63/16591.63/16890.56 | 83.24% | 26.25% | 11.62 (53.03%) | 11.62 | 96.33% |
| qwen3-14b | appworld | 16384 | 32 | fork_tp_run2 | 327.79 | 5907.52/13693.88/14245.63 | 56.72/62.96/75.95 | 9610.06/16780.05/16974.08 | 83.46% | 27.30% | 11.63 (53.10%) | 11.63 | 96.33% |
| qwen3-14b | swebench | 16384 | 32 | flash_tp | 385.29 | 5331.66/10810.36/10999.38 | 42.92/47.40/71.35 | 8567.37/12803.66/12934.52 | 97.62% | 35.87% | 11.74 (53.59%) | 11.74 | 0.00% |
| qwen3-14b | swebench | 16384 | 32 | flash_tp_repeat | 389.26 | 5320.46/10754.07/10961.19 | 40.75/45.95/71.27 | 8564.98/12735.40/12902.16 | 95.62% | 37.28% | 11.77 (53.71%) | 11.77 | 0.00% |
| qwen3-14b | swebench | 16384 | 32 | fork_tp_run1 | 322.32 | 5860.78/13602.72/13765.28 | 57.50/62.05/78.31 | 9573.21/16445.29/16484.55 | 82.55% | 29.33% | 11.84 (54.05%) | 11.84 | 94.98% |
| qwen3-14b | swebench | 16384 | 32 | fork_tp_run2 | 321.76 | 5884.06/13625.92/13892.35 | 57.77/62.39/78.54 | 9394.79/16738.75/16893.39 | 82.07% | 29.89% | 11.82 (53.96%) | 11.82 | 94.98% |

## Baseline Deltas

| Model | Dataset | Prefix | Branches | Variant | Baseline | Output throughput change | Peak GPU KV reduction | Peak total KV reduction |
|---|---|---:|---:|---|---|---:|---:|---:|
| qwen3-14b | agencybench | 16384 | 32 | fork_tp_run1 | flash_tp | -15.72% | 0.17% | 0.17% |
| qwen3-14b | agencybench | 16384 | 32 | fork_tp_run2 | flash_tp | -15.55% | -0.45% | -0.45% |
| qwen3-14b | agentboard | 16384 | 32 | fork_tp_run1 | flash_tp | -12.02% | -0.84% | -0.84% |
| qwen3-14b | agentboard | 16384 | 32 | fork_tp_run2 | flash_tp | -10.21% | 0.55% | 0.55% |
| qwen3-14b | appworld | 16384 | 32 | fork_tp_run1 | flash_tp | -13.26% | -1.41% | -1.41% |
| qwen3-14b | appworld | 16384 | 32 | fork_tp_run2 | flash_tp | -12.65% | -1.53% | -1.53% |
| qwen3-14b | swebench | 16384 | 32 | fork_tp_run1 | flash_tp | -16.34% | -0.85% | -0.85% |
| qwen3-14b | swebench | 16384 | 32 | fork_tp_run2 | flash_tp | -16.49% | -0.69% | -0.69% |

## Accuracy Guardrail

| Model | Dataset | Prefix | Branches | Variant | Exact match | Token F1 | Text similarity | Repeat exact match |
|---|---|---:|---:|---|---:|---:|---:|---:|
| qwen3-14b | agencybench | 16384 | 32 | fork_tp_run1 | 93.56% | 97.84% | 96.63% | - |
| qwen3-14b | agencybench | 16384 | 32 | fork_tp_run2 | 93.56% | 98.39% | 97.14% | 93.94% |
| qwen3-14b | agentboard | 16384 | 32 | fork_tp_run1 | 93.18% | 98.43% | 97.56% | - |
| qwen3-14b | agentboard | 16384 | 32 | fork_tp_run2 | 93.18% | 98.81% | 98.05% | 91.29% |
| qwen3-14b | appworld | 16384 | 32 | fork_tp_run1 | 92.05% | 98.29% | 97.22% | - |
| qwen3-14b | appworld | 16384 | 32 | fork_tp_run2 | 94.70% | 98.74% | 98.17% | 92.05% |
| qwen3-14b | swebench | 16384 | 32 | fork_tp_run1 | 85.61% | 98.45% | 97.21% | - |
| qwen3-14b | swebench | 16384 | 32 | fork_tp_run2 | 85.61% | 98.45% | 97.06% | 98.86% |

## Provenance

| Agentrix commit | Dirty | vLLM commit | Dirty | GPU blocks override | FlashInfer sampler | Prefix-aware policy | Admission window | Record cap | Full dataset | Runs |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 26f63d93787f | no | 287304ad68ce | no | - | no | no | 0 | 8 | no | 14 |

## Metric Notes

- Memory BW is NVIDIA `utilization.memory`, a memory-controller activity proxy rather than measured HBM GB/s.
- Peak GPU KV is sampled from `vllm:kv_cache_usage_perc` during the request phase.
- Logical KV read reduction estimates repeated prefix KV read volume avoided by ForkAttention; it is not physical cache capacity.
- Peak GPU KV reduction uses sampled physical GPU KV occupancy relative to the baseline named in the delta table.
- Throughput and request latency include both common-analysis and branch requests; branch-only distributions remain in the CSV.
- Record cap is the deterministic maximum number of source records per dataset; `-` with Full dataset=yes means every available record was used.
- Accuracy is deterministic output agreement against FlashAttention, not environment-level task accuracy.
- Repeat exact match compares Fork TP run 2 directly with Fork TP run 1.
- Experimental KV reload rebalance is disabled for this experiment matrix.
