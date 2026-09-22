# AgentX 在 vLLM-Ascend 上的路由与调度优化

## 目标和运行边界

以 [vLLM x AgentX 文章](https://vllm.ai/blog/2026-09-08-vllm-agentx)
为优化线索，以官方 [AgentX harness](https://github.com/SemiAnalysisAI/agentx-harness)
做端到端验证。第一阶段固定 Qwen3.5-9B、两张 Ascend 910B2 64GB、TP1/DP2、
BF16、eager、原生 262144 上下文、16 个 session trees，只改变路由或 prefill cap。

运行环境是 vLLM 0.22.1 + vLLM-Ascend v0.22.1rc1（子模块
`5f6faa0cb8830f667266f3b8121cd1383606f2a1` 加本地补丁）。本仓库 `vllm/`
是 0.28.0 CUDA 实验分支，不能直接安装到这个 Ascend 环境。
服务器使用独立 `/data/Agentrix/.venv`，vLLM 源码在 `/data/Agentrix/vllm-0.22.1`。
Ascend 插件当前装为 wheel；修改源码后须部署对应文件或重新构建，不能把源码同步误当成运行时已更新。

## 文章中的优化如何落地

| 优先级 | 优化 | 当前代码与差距 | 验证方法 |
| --- | --- | --- | --- |
| P0 | 会话粘性路由 | native DP 已接受 `X-data-parallel-rank`；实验代理把同一 `X-Session-ID` 固定到同一 rank，新会话轮转。CUDA 分支的 prefix/session-aware router 尚未迁入 0.22.1 | 对比 native、仅首轮固定和 sticky，统计缓存命中、TTFT、每用户输出速率和吞吐 |
| P0 | 长 prefill 分块，减轻队头阻塞 | 0.22.1 原生 scheduler 和 Ascend 自有 scheduler 已消费 `long_prefill_token_threshold`；无需复制整套调度器。新增 Ascend 混合块配置校验 | 保持 sticky、batch token budget=2048，比较 cap=0 与 1024 |
| P1 | 会话/分支边界的混合缓存保留 | Qwen3.5 已有 full-attention KV + GDN state 的 align 缓存。文章中的 interval/重复前缀选择性保留不能等同于开启 APC | 先统计丢失复用边界及重算量，再适配 scheduler、Mamba manager 和 GDN 状态保存；验证跨轮/分支状态及 logits 一致性 |
| P2 | 分层缓存与异步恢复 | Ascend 已有 `AscendStoreConnector`、`MooncakeHybridConnector` 和 CPU/NPU offload 路径；存在连接器不代表全部混合布局和状态恢复已验证 | 缓存压力足够时再测 CPU/NPU 搬运、恢复临时内存、抢占和重算，验证 full KV 与线性状态共同恢复 |
| 后续 | 图执行、GDN/attention 算子与 ForkAttention | 现有基线是 eager；Qwen3.5 的线性层与 full-attention 层需要分别定位瓶颈 | 先 profile，再做正确性和官方 AgentX 对照，单独记录图/算子收益 |

文章的 DEP prefill cadence 针对 MoE 跨 rank 同步；当前两个 dense 模型是独立 DP
副本，优先级较低。DCP/PCP、PP、P/D 拆分也需按模型与通信成本选择，不能从大规模
MLA/MoE 结果推断两卡 dense 模型收益。当前阶段维持两卡 DP。

### 首个 Ascend 补丁：避免 prefill 无法推进

Qwen3.5-9B TP1 在当前 Ascend 实现中，GDN state 与 attention page 对齐后，
`block_size=1024`，`mamba_cache_mode=align`。调度器会把中间 prefill chunk
向下取整为完整块。直接套用文章的 cap=512 会得到零个 token，使长请求无法推进。

在 [patch_mamba_config.py](../vllm-ascend/vllm_ascend/patch/platform/patch_mamba_config.py)
中，最终块大小和缓存模式确定后检查：当最大上下文至少能容纳一个块时，非零 cap 必须至少一个块，总 batch token budget
也必须至少一个块；非法配置启动时报错，不静默改变用户参数。cap=0 继续表示不限制
单个请求，1024 及更大 cap 保持现有语义。测试覆盖自动增大的块、自定义更大块、
由 Store 开启 align、非 align 模式，以及真实 scheduler 的零推进情况。

cap=2048 和 4096 在总 budget=2048 时不会比 cap=0 更严格，所以首轮只测 0/1024。
若要扩大分块扫描，须单独固定一个更大的 batch budget，重新跑对应 baseline。

### 对照矩阵

| 配置 | 新会话的首个请求 | 同会话后续请求 | Prefill cap |
| --- | --- | --- | ---: |
| native | native 负载调度 | native 负载调度 | 0 |
| first-turn-only | 轮转指定 rank | native 负载调度 | 0 |
| sticky | 轮转指定 rank | 固定原 rank | 0 |
| sticky + cap | 轮转指定 rank | 固定原 rank | 1024 |

native 与 sticky 比较的是整体路由策略，首轮分配和后续亲和性都不同。
first-turn-only 与 sticky 使用相同的首轮分配策略，更直接检查后续会话亲和性。
两者保持同一策略，但闭环负载中的实际请求到达顺序仍可能随运行变化。
最后两组只改变 prefill cap，总 token budget 始终为 2048。

## 可复现实验入口

代码在 [experiments/agentx-ascend](../experiments/agentx-ascend)。默认路径匹配当前服务器，
`AGENTRIX_ROOT`、`ASCEND_ENV`、`MODEL_PATH`、`VLLM_BIN`、`HARNESS_ROOT`、`AIPERF_BIN`
可以覆盖。推理和 benchmark 使用两个独立环境，harness 必须固定到
`56a0cf70f4c0359454ee4bd15a17770b541a3e3e`。脚本默认读取已准备的离线缓存。

```bash
# 在服务器的 /data/Agentrix 下运行；每轮只启动一个模型服务和一个 benchmark。
MODEL_NAME=Qwen3.5-9B LONG_PREFILL_TOKEN_THRESHOLD=1024 \
  bash experiments/agentx-ascend/serve.sh

# 另一个终端：可选 native、first-turn-only 或 sticky。
agentx/.venv/bin/python experiments/agentx-ascend/session_router.py \
  --policy sticky --port 8000 --upstream http://127.0.0.1:8001

# 再一个终端：输出目录必须不存在，防止覆盖前一轮结果。
ARTIFACT_DIR=/data/Agentrix/experiments/agentx-ascend/results/sticky-cap1024/artifacts \
  bash experiments/agentx-ascend/benchmark.sh
```

`DRY_RUN=1` 打印准确命令而不启动服务。`MODEL_NAME=Qwen3-8B` 选择 128K YaRN
模型配置，默认并发 4；比较模型必须显式统一并发、上下文过滤和其余实验条件。

代理只用于单进程实验：保留本轮所有会话映射，重启后清空；没有多副本共享状态和 TTL。
无 session ID 的请求交由 native DP。代理不解析或改写 prompt，保留 SSE 流式传输。
`first-turn-only` 只给每个新 session 的首次请求指定轮转 rank，后续不指定 rank，
与 sticky 共享首次分配逻辑。它保留已见 session 的记录，直到本轮实验结束。
`/routing-stats` 可查看映射数和请求分布。不同子代理的 session ID 独立分配，不能
声称已有父子分支共同路由。多 API frontend 的原生 engine 调度行为保持不变。

## 固定官方 benchmark 条件

- 数据集：`semianalysisai/cc-traces-weka-062126`，revision
  `23f152f6f0f9399a85901b89a6458def0ef16729`。
- `traces.jsonl` SHA-256：
  `29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`。
- 官方 `inferencex-agentx-mvp`，900 秒，seed=20260707，完整 dated corpus 按整条
  trajectory 的峰值上下文过滤；256K 下 220/393 条符合条件。无 entries cap。
- Streaming、server token count、`ignore_eos=true`、`first_turn_prefix` cache bust、
  idle gap cap=10 秒、trajectory start ratio=0..1；保留原始 trace 时序和输出长度。
- 不压缩、裁剪或改写官方 prompt，不把自定义 workload 当作官方 AgentX。
- 导出 `submission_valid`、错误、收尾取消、TTFT、每用户输出速率、输出吞吐、含缓存输入
  的 total throughput/NPU、cache-read 比例。`submission_valid` 是 harness 条件检查，
  不是官方榜单认证。时间窗内完成请求数可能随性能不同，须同时报告取消和覆盖率。

先以单次运行筛选方向；要宣称稳定提升，再做重复或交错 A/B，检查样本量与尾延迟。
已有两模型结果的并发和过滤池不同，不能用于归因模型快慢或缓存策略优劣。

## 已有基线与验证

原始启动脚本、manifest、summary 和 harness JSON 原样保存在
[baselines](../experiments/agentx-ascend/baselines)。这些脚本是历史记录，复跑请用上面的
参数化入口；完整逐请求记录和服务器日志仍保留在服务器原运行目录。

| 模型 | 上下文 / 并发 | 完成 / 错误 / 收尾取消 | 输出 tok/s | 平均 TTFT | cache read |
| --- | --- | --- | ---: | ---: | ---: |
| Qwen3-8B | 128K / 4 | 31 / 0 / 1 | 20.19 | 2.29 s | 94.01% |
| Qwen3.5-9B | 256K / 16 | 109 / 0 / 7 | 41.35 | 6.47 s | 78.78% |

两次均为 DP2、TP1、BF16、eager、sticky，官方报告 `submission_valid=true`。
Qwen3.5 原目录中的 `artifacts-final` 是中断的 C4 诊断；本地保存的有效报告来自 `artifacts-c16`。

### 2026-09-22 路由与调度对照结果

四组均为 Qwen3.5-9B、DP2/TP1、256K、并发 16、900 秒，BF16/eager，
总 batch token budget=2048。核对官方 JSON 后，除服务 URL 和输出目录外，
四组 `input_config` 完全一致。四组均 `submission_valid=true`、请求错误为 0。

| 路由 | Prefill cap | 完成 / 收尾取消 | 输出 tok/s | 平均 TTFT | P90 TTFT | cache read |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| native | 0 | 95 / 8 | 36.51 | 10.68 s | 31.51 s | 60.82% |
| first-turn-only | 0 | 98 / 8 | 37.35 | 9.36 s | 25.00 s | 64.63% |
| sticky | 0 | 109 / 7 | 41.35 | 6.47 s | 13.04 s | 78.78% |
| sticky | 1024 | 113 / 6 | 43.16 | 4.02 s | 7.92 s | 84.56% |

本轮最好的观测配置为 sticky + cap1024。相对 sticky cap0，输出吞吐高 4.4%，
平均 TTFT 低 37.8%；相同首轮策略下，sticky 也优于 first-turn-only。
这些结果支持继续保留会话亲和性，并把 1024 分块作为候选配置。
每组仅一次测量，未给出置信区间；闭环请求到达顺序和完成请求组合可能不同，
不能把这些幅度写成稳定性能保证。

机器可读汇总及相同负载校验指纹见
[comparison.json](../experiments/agentx-ascend/comparison.json)。新实验的原始官方 JSON、
manifest、路由计数和 summary 在
[comparisons](../experiments/agentx-ascend/comparisons)。完整日志和逐请求数据仍在服务器
`/data/Agentrix/experiments/agentx-ascend/results/`。

验证包括 15 项 Ascend 配置回归测试、7 项代理路由测试、真实 Qwen3.5 配置构建
（512 拒绝，0/1024 接受），以及两卡各两次 8219-token prompt 的生成和缓存复用检查。
每张卡的重复请求复用了 8192 token。配置验证日志在
[validation](../experiments/agentx-ascend/validation)，API 检查记录在 cap1024 的 `smoke.txt`。

服务器保留候选配置 sticky + cap1024，入口 `http://127.0.0.1:8000/v1`，模型名
`Qwen3.5-9B`。当前服务的 PID 和日志在 `results/active-service/`，独立于实验原始日志。
启动脚本仍允许用 `LONG_PREFILL_TOKEN_THRESHOLD=0` 复跑原配置；没有修改全局默认调度策略。
会话路由目前通过本仓库的实验代理实现，尚未接入原生 frontend。

验证命令：

```bash
# 普通 CPU 环境，需 aiohttp 和 pytest。
python -m pytest -q experiments/agentx-ascend/test_session_router.py

# 服务启动后：两张卡分别执行跨多个缓存块的长输入及重复请求。
python experiments/agentx-ascend/smoke.py

# 在已安装对应 Ascend 插件补丁的服务器环境中，避免 cwd 导入未构建的源码包。
source /data/Agentrix/activate-ascend.sh
cd /data
python -m pytest --import-mode=importlib -q \
  /data/Agentrix/vllm-ascend/tests/ut/patch/platform/test_mamba_prefill_config.py
```

在服务器完整结果目录运行 `summarize.py RUN_DIR`，从官方 JSON、逐请求记录和 `benchmark.log` 生成
`result-summary.json`。收尾取消数读取 profiling 完成日志，因为被取消的请求可能
不会进入成功请求的 JSONL，不能把 JSONL 中零条取消记录解释为没有收尾取消。
