# LongBench Shared-Document QA Experiment

## Artifact index

All paths in this report are repository-relative.

| Artifact | Path |
|---|---|
| Exact 32-document, 68-request input manifest | `benchmark/data/longbench_qa_qwen3_32b.jsonl` |
| Dataset and tokenizer provenance | `benchmark/data/longbench_qa_qwen3_32b.provenance.json` |
| Manifest builder | `benchmark/scripts/build_longbench_agentrix_cases.py` |
| Paired A/B launcher | `benchmark/scripts/run_longbench_qa_matrix.sh` |
| Request runner and per-request scoring | `benchmark/src/longbench_qa_runner.py` |
| Paired report generator | `benchmark/src/longbench_qa_report.py` |
| Aggregate paired report | `benchmark/results/longbench_qa_qwen3_32b/quality_performance_ab.json` |

The input manifest contains the complete document text, question, gold
answers, LongBench source ID, document SHA-256, and Qwen3 token count for every
executed request. Its SHA-256 is
`2a25cf48284ff2e01f3a83033d4ad8353f28031615c80565c15defb6a674f709`.

## Workload

The workload is built from LongBench v1 `multifieldqa_en` and `qasper`.
Questions sharing byte-identical source documents are grouped into one case.
The selected manifest contains 32 documents and 68 gold-answer questions,
with two to three questions per document and an average fanout of 2.125.
Qwen3 tokenization gives a context range of 4,842 to 14,851 tokens.

Each request asks Qwen3-32B to answer one question using only its document.
Both arms use greedy decoding, disabled thinking, identical prompts, identical
gold answers, 64-request concurrency, and an output limit of 64 tokens.

## Paired configuration

| Setting | Baseline | Agentrix |
|---|---|---|
| Attention | FlashAttention | ForkAttention |
| DP routing | Ordinary internal DP | Prefix-aware internal DP |
| Prefix cache | Enabled | Enabled |
| GPU KV blocks per rank | 3,852 | 2,600 |
| Model / precision | Qwen3-32B / BF16 | Qwen3-32B / BF16 |
| Hardware | 4 x NVIDIA H20 | 4 x NVIDIA H20 |

## Result

| Metric | Baseline | Agentrix | Change |
|---|---:|---:|---:|
| Exact Match | 5.88% | 5.88% | unchanged |
| Token F1 | 40.53% | 40.42% | -0.11 percentage points |
| Wall time | 76.79 s | 54.92 s | 1.40x speedup |
| Mean TTFT | 44.01 s | 29.52 s | 1.49x speedup |
| Mean request latency | 60.23 s | 39.16 s | 35.0% lower |
| Peak single-process GPU memory | 84,926 MiB | 81,822 MiB | 3,104 MiB lower |
| Normalized prediction agreement | - | 91.18% | - |

The paired report is generated from the two `run.json` files. Each
per-request record includes the request identity, document hash, question,
gold answers, prediction, EM, F1, TTFT, request latency, and API token usage.

## Reproduction

Download LongBench v1 `data.zip`, extract `multifieldqa_en.jsonl` and
`qasper.jsonl`, and build the exact case format:

```bash
python benchmark/scripts/build_longbench_agentrix_cases.py \
  --data-dir /path/to/longbench/data \
  --model /path/to/Qwen3-32B \
  --output benchmark/data/longbench_qa_qwen3_32b.jsonl
```

Run the paired matrix from the repository root:

```bash
MODEL_PATH=/path/to/Qwen3-32B \
bash benchmark/scripts/run_longbench_qa_matrix.sh
```

The launcher writes all outputs under
`benchmark/results/longbench_qa_qwen3_32b/` by default. Machine-specific
locations are supplied through environment variables and are never embedded
in the result schema or documentation.
