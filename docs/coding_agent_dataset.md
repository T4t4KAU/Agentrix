# Agentrix Coding-Agent End-to-End Dataset

## Status and Scope

This document defines the repository coding-agent workload that will serve as
the primary end-to-end Agentrix experiment. The existing experiments in
`main_experiment_results.md` and `dp_experiment_results.md` remain vLLM service
and mechanism benchmarks. They explain ForkAttention, prefix-aware routing,
KV-cache capacity, and scheduling behavior, but do not by themselves measure a
complete Agentrix agent workflow.

The coding-agent workload has a validated multi-round trace workload and 12
executable functional tasks: four each for Django, SQLite, and FFmpeg.
The suite excludes memory-safety, vulnerability, CVE, crash-reproduction, and
malformed-input security tasks.
It exercises LangGraph, long repository context, heterogeneous subagents,
multi-round history, deterministic tool observations, and vLLM DP routing. It
does not yet execute model-produced patches in the multi-agent runner.
The formal experiment in this document intentionally measures the replayed
systems trajectory, not coding accuracy. The standalone oracle layer is
available for all three repositories but is not part of the reported runs.

## Repository Suite

| Repository | Pinned revision | Intended role |
|---|---|---|
| Django | `cae38ec9b9bd394f630bdcbffa013c2761e831e9` | Large Python framework; API, ORM, async, and documentation-rich tasks |
| SQLite | Git `c744314bca7858d131577e0dbf8bb21aa3e3cbf7`; manifest `9062c79fc273d9e59090ea475e7d2abaf33c7cfe9948cbfb92b979a6ed31a37f` | Fast deterministic build and test loop for regression and parameter sweeps |
| FFmpeg | `7f0b6476b6ef2d07d163a7d3229f8c9e250112b5` | Larger cross-module C workload covering ownership, scheduling, and media pipelines |

SQLite is the fast daily benchmark, Django supplies a Python framework
workload, and FFmpeg is the heavier cross-module validation. DP ranks remain
identical model replicas. Repositories or agent roles are never statically
assigned to particular ranks; prefix-aware routing dynamically places each
case cohort.

## Case Shape

Each repository currently contains four cases. A case has a frozen repository
parent context of approximately 30K harness-tokenizer tokens and 16 coding
subagents with different investigation roles. The four cases in one run are
submitted together, producing four independent long-prefix cohorts suitable
for DP=4 without manufacturing rank-specific roles.

The source specifications are stored in:

```text
benchmark/configs/django_agentrix_case_specs.json
benchmark/configs/sqlite_agentrix_case_specs.json
benchmark/configs/ffmpeg_agentrix_case_specs.json
```

Generated JSONL and redundancy reports are stored under:

```text
benchmark/data/{django,sqlite,ffmpeg}_agentrix/
```

Every generated case records the repository revision, parent hash, tokenizer,
source paths and hashes, parent-token count, branch instructions, stages, tool
observations, and termination depth. Repository source trees are linked locally
under `benchmark/repos/` and are not copied into Git.

## Heterogeneous Multi-Round Trajectories

A uniform fixed-depth conversation is not representative of current coding
agents. Each 16-subagent case therefore uses the following deterministic
trajectory population:

| Subagents per case | Model rounds | Stages |
|---:|---:|---|
| 4 | 1 | Triage, then terminate |
| 8 | 2 | Triage, repository-search observation, refinement |
| 4 | 3 | Triage, repository search, independent reviewer feedback, final recommendation |

Across four concurrent cases, the request waves are therefore `64 -> 48 ->
16`, or 128 branch model requests in total. A wave barrier applies only to
subagents that remain active in that stage. A branch retains its preceding
assistant responses, user instructions, and tool observations, so later turns
grow naturally instead of restarting from the frozen parent.

Two execution policies are required:

- `live`: feed each model response into the next turn. This is the primary
  end-to-end agent mode and may produce different later inputs across variants.
- `replay`: replace prior model responses with fixed trace text. This controls
  the exact request workload and is the systems-performance A/B mode.

Results from these modes must not be mixed. Replay establishes causal systems
performance; live mode measures the realized agent trajectory and must report
both quality and the actual token work performed.

The current tool observations are deterministic, bundled repository-search and
reviewer events. The formal task layer will replace or supplement these with
sandboxed `search`, `read`, `edit`, `build`, and `test` events while retaining a
replayable event log.

## Why This Shape Can Benefit Agentrix

The workload is selected from a plausible multi-subagent coding workflow:

1. a parent agent loads a large, pinned repository context;
2. specialized subagents share that context but receive private assignments;
3. inexpensive investigations terminate early;
4. uncertain investigations call tools and continue through additional stages;
5. histories and tool results accumulate, and some source segments occur in
   multiple cases; and
6. concurrent cohorts compete for finite per-rank KV capacity.

This naturally exposes Agentrix's target properties: shared long prefixes,
fanout, divergent suffixes, uneven branch lifetimes, multi-round KV growth, and
large tool-produced intermediate context. The benchmark must not enforce equal
branch depth, bind roles to DP ranks, or select variant-specific request order
to increase a reported speedup.

## Static Redundancy Baseline

The multi-round static audit counts the frozen parent and declared user/tool
messages materialized in every model request. Runtime-generated assistant
tokens are excluded because they are unknown before a live run. The metric is
logical prompt representation, not measured HBM traffic or directly removable
tokens.

| Repository | Cases / subagents | Declared model requests | Materialized static prompt tokens | Unique tokens with one parent per case | Logical redundancy |
|---|---:|---:|---:|---:|---:|
| Django | 4 / 64 | 128 | 3,857,087 | 125,469 | 96.75% |
| SQLite | 4 / 64 | 128 | 3,857,153 | 125,540 | 96.75% |
| FFmpeg | 4 / 64 | 128 | 3,857,677 | 125,837 | 96.74% |

SQLite also contains 14,397 statically duplicated source-segment tokens across
cases; FFmpeg contains 6,550. These values identify compression opportunities
but do not prove physical KV duplication. Runtime experiments must separately
record exact prefix depth, local prompt compute/cache-hit tokens, per-rank KV
residency, eviction/recomputation, and tool-result byte duplication.

## Formal Correctness Layer

The final dataset should use executable tasks rather than grading prose alone.
Each task will start from a clean pinned worktree, apply a deterministic seeded
regression, and expose public tests while retaining at least one hidden test.
The agent may inspect and edit only its sandbox. Its final patch is evaluated by:

1. patch application and repository cleanliness checks;
2. the task-specific fail-to-pass regression test;
3. selected neighboring pass-to-pass tests;
4. build or syntax validation;
5. a hidden edge-case test; and
6. patch-scope and forbidden-file checks.

Primary quality is resolved-task rate. Secondary quality includes public and
hidden test pass rates, invalid-patch rate, regression count, tool calls,
turns-to-resolution, and tokens-to-resolution. Evidence-citation or JSON-format
scores may be retained as diagnostics but cannot replace executable tests.

The initial task mix should contain small, reproducible defects with fast
oracles: four tasks per repository for development and at least 20 per
repository for a formal aggregate. Mutations must be reviewed to avoid trivial
textual reversal, and the gold fix must not be included in prompts or tool
observations.

### Executable Oracle Tasks

Preparation exports the pinned repository revision without its Git history,
injects the task mutation, initializes a new task-baseline repository, builds
the focused target, and verifies that the public test fails. The agent receives
the issue, task worktree, and public test but not the hidden oracle or original
clean history.

Evaluation rebuilds the candidate and checks modification scope, public-test
integrity, the focused regression, and hidden neighboring behavior. The 12
current tasks cover ordinary functional semantics: Django decorator and
request/session/module behavior, SQLite scalar-function results, and FFmpeg
string, dictionary, FIFO, and duration helpers. Security-oriented cases are
not part of this suite.

The repository tool layer currently exposes bounded `search`, line-range
`read`, scope-checked `apply_patch`, `diff`, and `public_test` operations. Every
tool event records arguments, full-content and returned-content hashes,
original and returned byte counts, truncation, and wall time. These event logs
are the input to subsequent exact deduplication and tool-output compression
experiments.

Relevant paths are:

```text
benchmark/coding_tasks/index.json
benchmark/coding_tasks/{django,sqlite,ffmpeg}_*/
benchmark/coding_oracles/hidden/*_regressions.py
benchmark/src/coding_task_oracle.py
benchmark/src/coding_agent_tools.py
```

## Performance and Compression Metrics

Every Flash ordinary-DP versus Fork prefix-aware-DP comparison uses a fresh
service, identical model and physical KV capacity, identical case order, and
all four GPUs. The report must include:

- end-to-end task throughput and resolved tasks per hour;
- per-stage wall time, TTFT, TPOT, and output throughput;
- actual input/output tokens by stage and branch depth;
- local prompt compute, local cache hits, and external KV transfer;
- KV occupancy over time, preemptions, evictions, and recomputed tokens;
- route ownership and per-rank allocation;
- context bytes/tokens before and after compression;
- exact duplicate tool-result bytes and retained dead-branch context; and
- correctness metrics from the executable task oracle.

Compression experiments must compare at least: no compression, exact segment
deduplication, tool-output compaction, and combined compression. A reduction in
tokens is not a success if resolved-task rate falls outside the predefined
quality tolerance. Results should report both raw systems throughput and
quality-adjusted throughput.

## H20 DP=4 Multi-Round Results

The formal systems run was recorded on 2026-07-17. It used four NVIDIA H20-3e
GPUs (SM90, 143,771 MiB each), Qwen3-32B in float16, internal DP=4 and TP=1.
Each rank used 3,852 KV blocks (61,632 tokens), `max_num_seqs=64`, a 40,960
model length, and no KV offload. Each fresh service processed four cases, 64
subagents, and the deterministic three-wave replay trajectory `64 -> 48 ->
16`, for 128 branch requests and 8,192 generated tokens. Application prompt
compaction was enabled identically in both arms.

| Repository / variant | Branch tok/s | Branch wall | Case wall | TTFT P50 | TPOT P50 | Input tokens | Peak GPU memory (4 GPUs) | Peak server RSS | Observed peak KV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Django, Flash ordinary DP | 22.43 | 365.22 s | 386.02 s | 45.01 s | 342.88 ms | 3,841,511 | 327.16 GiB | 12.70 GiB | 99.3% |
| Django, Fork prefix-aware DP | 445.92 | 18.37 s | 39.09 s | 1.73 s | 66.16 ms | 3,841,511 | 334.53 GiB | 15.97 GiB | 51.4% |
| SQLite, Flash ordinary DP | 27.21 | 301.07 s | 323.75 s | 48.20 s | 47.61 ms | 4,056,004 | 327.16 GiB | 12.60 GiB | 53.4% |
| SQLite, Fork prefix-aware DP | 449.46 | 18.23 s | 40.94 s | 1.60 s | 69.01 ms | 4,056,004 | 334.53 GiB | 15.94 GiB | 55.1% |
| FFmpeg, Flash ordinary DP | 29.55 | 277.21 s | 299.65 s | 47.44 s | 49.35 ms | 3,957,765 | 327.16 GiB | 12.63 GiB | 99.9% |
| FFmpeg, Fork prefix-aware DP | 457.32 | 17.91 s | 40.45 s | 1.67 s | 65.95 ms | 3,957,765 | 334.53 GiB | 15.98 GiB | 54.8% |

Fork prefix-aware DP improves branch throughput by 19.88x on Django, 16.52x
on SQLite, and 15.48x on FFmpeg. Case-wall improvements, which include the
four parent/bootstrap requests, are 9.88x, 7.91x, and 7.41x respectively.
These are single repeats on a deliberately favorable four-cohort/four-rank
capacity shape, so they establish mechanism benefit for this workload rather
than a general coding-agent average.

### Prompt Work, Queueing, and Storage

| Repository / variant | Local prompt compute | Local cache hit | Hit share | Cumulative prefill | Cumulative queue | Preemptions |
|---|---:|---:|---:|---:|---:|---:|
| Django, Flash | 1,608,613 | 2,352,624 | 59.39% | 1,943.42 s | 3,991.37 s | 0 |
| Django, Fork prefix-aware | 127,813 | 3,833,424 | 96.77% | 114.25 s | 0.0015 s | 0 |
| SQLite, Flash | 1,534,661 | 2,647,776 | 63.31% | 1,842.14 s | 4,780.22 s | 0 |
| SQLite, Fork prefix-aware | 134,677 | 4,047,760 | 96.78% | 131.73 s | 0.0017 s | 0 |
| FFmpeg, Flash | 1,486,720 | 2,594,400 | 63.57% | 2,021.73 s | 4,310.24 s | 1 |
| FFmpeg, Fork prefix-aware | 131,664 | 3,949,456 | 96.77% | 130.60 s | 0.0018 s | 0 |

Total prompt-source counters match exactly within every pair and external KV
transfer is zero. Prefix-aware placement reduces local prompt computation by
91.2-92.1% and collapses the ordinary-DP queue. Fork physical execution was
active: the three runs recorded 956-975 observed steps, 752-793 fork-active
steps, and 13,961-15,288 shared CTAs.

Storage does not improve in every sense. Fork's sampled peak aggregate GPU
memory is about 7.4 GiB higher and its process-tree RSS about 3.3 GiB higher,
reflecting backend workspace/metadata and more useful concurrently resident
work. Its residency is nevertheless more effective: it achieves a 96.8%
prompt-cache hit share and avoids the near-full KV/cache-thrash behavior seen
in the Django and FFmpeg Flash logs. “Storage benefit” should therefore be
reported as reduced duplicate logical context and reduced prompt
recomputation, not as lower peak allocated HBM.

The application module removed 48 byte-identical tool/source segments per
repository: 763,820 characters for Django, 547,975 for SQLite, and 414,950 for
FFmpeg. No raw/compact runtime arm was run; both primary variants used the
same compacted requests. Its design and limitations are documented in
`application_prompt_compaction.md`.

Raw server-side artifacts are retained under:

```text
/test__02/hwx/Agentrix/benchmark/results/
  h20_coding_application_20260717/{django,sqlite,ffmpeg}/
```

## Earlier Single-Round Pilot

The H20 DP=4 SQLite single-round pilot completed 4 bootstrap and 64 branch
requests in both arms. FlashAttention ordinary DP achieved 119.54 branch
tokens/s; ForkAttention with prefix-aware DP achieved 825.76 branch tokens/s,
a 6.91x branch-stage improvement. Complete case wall time improved by 3.85x.
Local prompt computation fell from 552,195 to 125,187 tokens, cumulative queue
time fell from 2,212.50 seconds to 0.0009 seconds, and preemptions fell from 46
to zero. All 64 generated branch responses were byte-identical across the two
arms.

This pilot validated the service mechanism before the multi-round replay above.
It has no executable coding-accuracy oracle and should not replace the formal
three-repository result.

## DP=8 Full-System Comparison

The next comparison extends the coding-agent workload to eight internal-DP
replicas and uses exactly eight independent approximately 30K parent contexts,
preserving the original one-case-per-replica pressure shape. Django, SQLite,
and FFmpeg remain separate experiments; repository cases are never mixed in
one run.

Each repository provides 24 deterministic cases divided into three fixed
eight-case batches at offsets 0, 8, and 16. Each batch contains 128 subagents.
Its three-round replay produces the request waves `128 -> 96 -> 32`, or 256
branch model requests. Thus the 24 cases provide three within-repository
repeats without changing the per-rank workload.

The additional cases come from ordinary functional commits reachable from the
pinned revision. The generator excludes commits whose subjects or paths match
security, credential, crash, malformed-input, fuzzing, bounds, or unsafe-memory
topics. Every selected case asks for read-only behavior and test analysis; it
does not request a code modification. Each generated record stores the source
commit and subject for audit.

Every eight-case batch contains 96 byte-identical tool/source segments that
are already present in its parent contexts. Across all three batches,
application compaction can omit 288 sections: 1,057,489 characters for Django,
1,003,179 for SQLite, and 861,577 for FFmpeg. The baseline retains those bytes;
the optimized arm removes them through stable application-owned segment IDs.

The default DP=8 matrix intentionally compares two complete configurations:

- `flash_uncompressed_dp`: vLLM with FlashAttention, ordinary internal-DP
  placement, and application prompt compaction disabled;
- `fork_prefix_aware_compact_dp`: ForkAttention with prefix-aware internal-DP
  placement and application prompt compaction enabled.

The baseline therefore contains neither application compression nor
prefix-aware placement. Tool-KV trimming and predicted TTL are explicitly
disabled for both variants and are outside this experiment. Both variants use
the same case files, replay history, branch order, output limit, model, and
physical KV capacity. The output records the attention backend, DP policy, and
compaction state to make accidental configuration mixing visible.

Memory sampling records per-GPU and aggregate HBM use, the delta above the
post-load warm minimum, memory-controller utilization, per-engine and aggregate
KV occupancy, estimated live KV tokens, vLLM process-tree RSS, and application
process-tree RSS. Raw 0.5-second samples remain available alongside the summary.

This two-arm result measures the complete optimized system against the stated
vLLM baseline. It does not by itself attribute an improvement separately to
prompt compaction, prefix-aware routing, or ForkAttention. A later attribution
matrix should add FlashAttention with compaction and ForkAttention with
prefix-aware DP but without compaction while holding all other controls fixed.

Run all three batches for one repository with:

```bash
MODEL_PATH=/test__02/hwx/Qwen3-32B \
REPOSITORY=sqlite \
OUTPUT_ROOT=benchmark/results/coding_agentrix_dp8/sqlite \
bash benchmark/scripts/run_django_agentrix_dp.sh
```

Repeat with `REPOSITORY=django` and `REPOSITORY=ffmpeg`. The launcher defaults
to GPUs `0,1,2,3,4,5,6,7`, DP=8, the three batch offsets `0 8 16`, three replay
rounds, and 64 generated tokens per branch request. `CASES_PATH` and
`BATCH_OFFSETS` can select a specific generated file or one batch. The
historical launcher name is retained for compatibility even though the runner
and schema are repository-generic.

## Reproduction

Regenerate one repository's deterministic commit specifications and cases with:

```bash
PYTHONPATH=benchmark/src benchmark/.venv/bin/python -m commit_case_specs \
  --repo benchmark/repos/sqlite --repository-slug sqlite/sqlite \
  --output benchmark/configs/sqlite_agentrix_commit24_specs.json \
  --count 24 --allowed-suffixes .c,.h --max-context-paths 32

benchmark/.venv/bin/python benchmark/scripts/build_django_agentrix_cases.py \
  --repo benchmark/repos/sqlite \
  --specs benchmark/configs/sqlite_agentrix_commit24_specs.json \
  --output benchmark/data/sqlite_agentrix/cases_30k_b16_commit24.jsonl \
  --target-tokens 30000 --repo-id sqlite/sqlite \
  --repository-name SQLite --manifest-file manifest.uuid
```

Use `--allowed-suffixes .py --max-context-paths 64` for Django and
`--allowed-suffixes .c,.h --max-context-paths 32` for FFmpeg. The baseline
keeps application compaction disabled; only the optimized variant enables it.
Use `live` only for a later quality-aware experiment.
