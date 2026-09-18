# Agentrix Coding-Agent End-to-End Dataset

## Status and Scope

This document defines the repository's separate coding-agent workload.
Serving-load replay and executable task quality are measured separately.

The coding-agent workload has a validated multi-round trace workload and 12
executable functional tasks: four each for Django, SQLite, and FFmpeg.
The suite excludes memory-safety, vulnerability, CVE, crash-reproduction, and
malformed-input security tasks.
It exercises LangGraph, long repository context, heterogeneous subagents,
multi-round history, deterministic tool observations, and vLLM DP routing. It
does not yet execute model-produced patches in the multi-agent runner.
The standalone oracle layer is available for all three repositories.

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
