# AgentX 在 vLLM-Ascend 上的优化与验证

## 目标和运行边界

以 [vLLM x AgentX 文章](https://vllm.ai/blog/2026-09-08-vllm-agentx)
为优化线索，以官方 [AgentX harness](https://github.com/SemiAnalysisAI/agentx-harness)
做端到端验证。第一阶段固定 Qwen3.5-9B、两张 Ascend 910B2 64GB、TP1/DP2、
BF16、eager、原生 262144 上下文、16 个 session trees，只改变路由或 prefill cap。
后续混合缓存对照固定 sticky + cap1024，只改变本地检查点保留策略。
图执行对照在选定的缓存策略上固定每卡 36 GiB 缓存预算，比较 eager 与 Decode ACLGraph。

运行环境是 vLLM 0.22.1 + vLLM-Ascend v0.22.1rc1（子模块
`da9b47a226f2d1b5428f4658af5c4bfe9813dbeb`，包含配置校验和混合缓存保留补丁）。本仓库 `vllm/`
是 0.28.0 CUDA 实验分支，不能直接安装到这个 Ascend 环境。
服务器使用独立 `${REPO_ROOT}/.venv`，vLLM 源码在 `${REPO_ROOT}/vllm-0.22.1`。
Ascend 插件当前装为 wheel；修改源码后须部署对应文件或重新构建，不能把源码同步误当成运行时已更新。

## 实验记录保存规则

本文路径遵循[文档占位符约定](README.md)，复现前通过私有环境设置变量。
本地只保留代码、复现方法和本文中的关键配置、结果、验证结论及局限。
后续直接在服务器运行和分析实验，不再将每轮日志、JSON 报告、manifest、诊断快照或测试输出下载到本地。
新实验的完整产物统一保存在服务器 `${RESULTS_DIR}/`。
下文的实验记录路径均指服务器；历史归档仍在服务器同项目的 `baselines/`、`comparisons/`、
`retention/`、`retention-v2/` 和 `validation/` 目录，均相对于 `${EXPERIMENT_DIR}/`。
2026-09-23 清理本地副本前，已逐文件核对 81 份服务器归档的 SHA-256。

## 文章中的优化如何落地

| 优先级 | 优化 | 当前代码与差距 | 验证方法 |
| --- | --- | --- | --- |
| P0 | 会话粘性路由 | native DP 已接受 `X-data-parallel-rank`；实验代理把同一 `X-Session-ID` 固定到同一 rank，新会话轮转。CUDA 分支的 prefix/session-aware router 尚未迁入 0.22.1 | 对比 native、仅首轮固定和 sticky，统计缓存命中、TTFT、每用户输出速率和吞吐 |
| P0 | 长 prefill 分块，减轻队头阻塞 | 0.22.1 原生 scheduler 和 Ascend 自有 scheduler 已消费 `long_prefill_token_threshold`；无需复制整套调度器。新增 Ascend 混合块配置校验 | 保持 sticky、batch token budget=2048，比较 cap=0 与 1024 |
| P1 | 会话/分支边界的混合缓存保留 | 新增默认关闭的稀疏检查点保留、共享前缀边界补存和未缓存块优先回收；沿用现有 GDN/卷积状态复制 | 缓存管理器测试、两卡冷计算/状态恢复一致性，以及固定 sticky/cap1024 的官方 AgentX 对照 |
| P2 | 分层缓存与异步恢复 | Ascend 已有 `AscendStoreConnector`、`MooncakeHybridConnector` 和 CPU/NPU offload 路径；存在连接器不代表全部混合布局和状态恢复已验证 | 缓存压力足够时再测 CPU/NPU 搬运、恢复临时内存、抢占和重算，验证 full KV 与线性状态共同恢复 |
| 已验证 | Decode ACLGraph | 复用现有 Ascend 图执行与 GDN 参数更新能力，新增显式启动配置；含必要的编译/融合路径 | 同容量 eager/graph 各两次官方 AgentX，另做两卡跨模式正确性与独立剖析 |
| 实验性接入 | NPU ForkAttention | 已接入 Qwen3.5 decode 的物理页准入、多共享组规划、计划/工作区复用和 ACLGraph 内原生回退；默认关闭 | 两卡算子与模型检查；独立算子收益与官方 AgentX 结果分别记录 |
| 后续 | GDN 算子与模型级集成 | Qwen3.5 的线性层与 full-attention 层仍需分别分析 | 基于图模式定位剩余瓶颈，再做正确性和官方 AgentX 对照 |

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

### 混合缓存的选择性保留

后续实现增加了默认关闭的 `additional_config.mamba_cache_retention`。当前针对
vLLM 0.22.1、Qwen3.5、`align` APC；拒绝 speculative decoding、KV connector 和
context parallelism 的组合。DP2/TP1 保持原配置。

源码检查发现，原 `align` 路径已经能保留每个已计算的块边界；cap1024 下，长 prefill
的每个 1024-token 分块末尾都会进入 GDN 状态缓存。改动重点是减少低价值的中间检查点，
让 attention KV 和有复用价值的 GDN 状态在共用块池中保留更久。
当前请求逐步推进所需的状态更新和复制仍会执行；减少的是供后续请求复用的缓存条目。

| 参数 | 语义 |
| --- | --- |
| `interval: null` 或省略 | 保持原有缓存和空闲块回收策略 |
| `interval: 0` | 仅保留已知复用边界及 decode 产生的完整块 |
| `interval: 8192` | 额外保留每 8192 token 的检查点；必须是最终块大小的整数倍 |
| `diagnostics: true` | 每 16 次缓存查询输出累计诊断快照，不记录 prompt 或 token 内容 |
| `prefer_reuse_boundaries: true` | 实验性回收策略：池压力下优先淘汰尚未实际复用的周期性检查点；必须显式设置 interval，默认关闭 |

保留位置包括输入末尾的完整块，以及完全相同输入重放时可达的
`floor((prompt_tokens - 1) / block_size) * block_size`。输入长度恰好是整块时，
重放与追加输入需要的检查点不同，两者都保留。decode 产生的完整块继续缓存。

对于轮次内部的分支，hybrid coordinator 先查询 full-attention KV，再查 GDN 状态。
如果 full-attention 已匹配到更远的位置，就把该位置记录为当前请求的共享前缀边界。
调度器确保 prefill 真正停在此边界，随后保存实际计算出的状态；后续同前缀请求才可命中。
这只使用本 rank 已有的哈希缓存，不推测父子 session 的关系，也不跨卡迁移状态。

仅跳过哈希登记并不能避免缓存被挤出：原 0.22.1 空闲队列会把未缓存的块也放到队尾。
开启策略后，引用计数归零且没有哈希的块优先回收；有哈希的块继续按原 LRU 顺序回收。
正在执行、被多个请求引用或作为状态复制源的块仍受原引用计数保护。
同一调度步刚登记、尚未由 worker 计算完的状态，继续使用原有屏障阻止其他请求提前读取。
预分配缓存池大小不变，优化的是池内内容的驻留和复用，不能用 `npu-smi` 总 HBM 占用是否下降来验收。

改动位于 `vllm_ascend/core/mamba_retention.py`、`patch_mamba_manager.py` 和
`patch_scheduler.py`，配置由 `ascend_config.py` 校验。EngineCore 从调度器显式传入配置，
不依赖 Worker 进程的配置单例。GDN/卷积状态计算和复制继续沿用现有实现。

```bash
# 候选配置；interval 暂为实验参数，并非已确定的最优值。
MODEL_NAME=Qwen3.5-9B LONG_PREFILL_TOKEN_THRESHOLD=1024 \
  MAMBA_RETENTION_INTERVAL=8192 MAMBA_RETENTION_DIAGNOSTICS=1 \
  bash experiments/agentx-ascend/serve.sh

# 对照：不设置 MAMBA_RETENTION_INTERVAL，只开同样的诊断。
MODEL_NAME=Qwen3.5-9B LONG_PREFILL_TOKEN_THRESHOLD=1024 \
  MAMBA_RETENTION_DIAGNOSTICS=1 bash experiments/agentx-ascend/serve.sh

# 直接连接后端，分别验证两个 rank，避免实验代理覆盖指定 rank。
# 该脚本是状态正确性检查，不计入官方 AgentX 分数。
python experiments/agentx-ascend/hybrid_cache_smoke.py \
  --expect-sparse --output /tmp/hybrid-cache-correctness.json
```

正确性脚本使用不同 cache salt 建立冷计算参照，再检查原输入重放、追加输入、轮次内部
分支与并发兄弟分支的生成 token、结束原因和生成 token 的 logprob。logprob 的绝对容差
为 0.02；这不是全词表 logits 比较。单元测试另外覆盖有限容量下的真实淘汰、同一步状态
屏障、引用计数、缓存重置与非法参数。

诊断中的 `checkpoint_gap_tokens` 是本 rank full-attention 可命中长度减去混合缓存
实际命中长度的累计值。它包含 GDN 检查点未保留或已淘汰导致的回退，不能单独区分
冷输入、跨 rank 路由和历史淘汰。每条诊断是当时的累计快照，不能把多条快照相加。
计数按缓存查询统计，可能包含等待调度时的重试，因此不等同于最终实际执行的重算 token 数。
`retained_blocks`/`skipped_blocks` 是各 Mamba cache group 的累计登记/跳过次数，
`resident_hashed_blocks` 才是采样时整个共享池中有哈希的物理块数。

### 上游来源和第二轮改进

这里的稀疏保留、共享前缀边界以及未缓存块优先复用都已有上游来源，不属于原创缓存算法：

- [vLLM #43447](https://github.com/vllm-project/vllm/pull/43447)：SWA 间隔保留，以及未缓存块优先复用。
- [vLLM #45845](https://github.com/vllm-project/vllm/pull/45845)：将间隔保留扩展到 Mamba/线性注意力。
- [vLLM #47782](https://github.com/vllm-project/vllm/pull/47782)：将 Marconi 共享前缀检查点与稀疏保留结合。
- [Ascend RFC #10517](https://github.com/vllm-project/vllm-ascend/issues/10517)：跟踪上述上游特性。

第二轮检查了 vLLM `v0.26.0` 和上游主线快照
[`74370a0e3044c16b6ba58f9c144b02ab733cd73a`](https://github.com/vllm-project/vllm/tree/74370a0e3044c16b6ba58f9c144b02ab733cd73a)。
运行环境仍为已验证的 0.22.1，未直接升级引擎。较新的子块检查点、写时复制、
context parallelism、投机推理等路径没有在本补丁中启用。

所参考主线的 `CacheConfig.prefix_cache_retention_interval` 已默认设为 0，
因此本轮也对照了仅保留复用边界的配置。本补丁仍额外保留 decode 产生的完整块，
并保留旧引擎整块输入重放与追加输入各自需要的边界；它是同类策略的适配，不等同于完整移植新版实现。

改进包括：

1. 保存请求已经观察到的共享前缀位置。请求因容量不足等待重试时，即使期间 full-attention
   KV 被其他请求淘汰，已知边界也不会丢失；到达该位置后仍保存真实重算的状态。
2. 增加可选的回收策略。未被实际复用过的周期性检查点进入待考察集合；轮次末尾、decode
   和已知共享前缀检查点不进入。已进入集合的检查点被请求真正引用后，退出集合，恢复普通 LRU。
   当下一次分配将淘汰有哈希的块时，优先选择集合中引用计数为零的块；空白或未缓存块仍先使用。
   这是一项需要实测的策略扩展，借鉴缓存的 probation/promotion 思路，不宣称算法首创。
3. 元数据按物理块 ID 保存，并用缓存哈希核对，容量受缓存池块数限制；淘汰及成功重置时清理。
   引用计数、状态复制源保护、同一步计算屏障和淘汰事件沿用原引擎，不永久固定任何缓存块。

新增诊断 `preferred_periodic_reclaims` 表示重新选择周期性块作为淘汰对象的次数，
`promoted_checkpoints` 表示周期性块因实际引用退出待考察集合的次数；
`periodic_blocks` 和 `free_periodic_blocks` 是当前集合大小。这些仍是调度器计数，不能直接换算为收益。

启动实验策略：

```bash
MODEL_NAME=Qwen3.5-9B LONG_PREFILL_TOKEN_THRESHOLD=1024 \
  MAMBA_RETENTION_INTERVAL=8192 MAMBA_RETENTION_DIAGNOSTICS=1 \
  MAMBA_PREFER_REUSE_BOUNDARIES=1 bash experiments/agentx-ascend/serve.sh
```

`run_retention.py` 可重启指定的上一轮实验服务、记录源代码与安装包哈希，并执行官方场景。
`--validate-only` 在独立服务周期执行正确性检查；正式测量另起服务，以免预热污染对照。
`compare_retention.py` 默认要求插件哈希相同；跨代码修订比较必须显式加 `--allow-code-change`，
报告会列出变化的文件。诊断在测量结束时冻结，后续健康检查或模型验证不会改写已有快照。
本轮相对第一轮的代码差异保存在服务器
`${EXPERIMENT_DIR}/retention-v2/validation/v1-to-v2.patch`，
所参考的上游提交及本地实现哈希见同目录的 `implementation-manifest.json`。

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
# 在服务器的 ${REPO_ROOT} 下运行；每轮只启动一个模型服务和一个 benchmark。
MODEL_NAME=Qwen3.5-9B LONG_PREFILL_TOKEN_THRESHOLD=1024 \
  bash experiments/agentx-ascend/serve.sh

# 另一个终端：可选 native、first-turn-only 或 sticky。
agentx/.venv/bin/python experiments/agentx-ascend/session_router.py \
  --policy sticky --port 8000 --upstream http://127.0.0.1:8001

# 再一个终端：输出目录必须不存在，防止覆盖前一轮结果。
ARTIFACT_DIR="${RESULTS_DIR}/sticky-cap1024/artifacts" \
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

原始启动脚本、manifest、summary 和 harness JSON 原样保存在服务器
`${EXPERIMENT_DIR}/baselines/`。这些脚本是历史记录，复跑请用上面的
参数化入口；完整逐请求记录和服务器日志仍保留在服务器原运行目录。

| 模型 | 上下文 / 并发 | 完成 / 错误 / 收尾取消 | 输出 tok/s | 平均 TTFT | cache read |
| --- | --- | --- | ---: | ---: | ---: |
| Qwen3-8B | 128K / 4 | 31 / 0 / 1 | 20.19 | 2.29 s | 94.01% |
| Qwen3.5-9B | 256K / 16 | 109 / 0 / 7 | 41.35 | 6.47 s | 78.78% |

两次均为 DP2、TP1、BF16、eager、sticky，官方报告 `submission_valid=true`。
Qwen3.5 原目录中的 `artifacts-final` 是中断的 C4 诊断；有效报告来自 `artifacts-c16`。

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

机器可读汇总及相同负载校验指纹在服务器
`${EXPERIMENT_DIR}/comparison.json`；原始官方 JSON、
manifest、路由计数和 summary 在同目录的 `comparisons/`。完整日志和逐请求数据在服务器
`${RESULTS_DIR}/`。

验证包括 15 项 Ascend 配置回归测试、7 项代理路由测试、真实 Qwen3.5 配置构建
（512 拒绝，0/1024 接受），以及两卡各两次 8219-token prompt 的生成和缓存复用检查。
每张卡的重复请求复用了 8192 token。配置验证日志在服务器
`${EXPERIMENT_DIR}/validation/`，API 检查记录在
`comparisons/sticky-c16-cap1024/smoke.txt`。

最近一次实验结束时，服务器选用 sticky + cap1024，并启用下节验证的 8192-token 混合缓存保留间隔，
入口 `http://127.0.0.1:8000/v1`，模型名 `Qwen3.5-9B`。当前服务的 PID 和部署记录在
`results/active-service/`，其中日志链接指向当前实验目录，之前的服务日志已另存。
启动脚本仍允许用 `LONG_PREFILL_TOKEN_THRESHOLD=0` 复跑原配置；没有修改全局默认调度策略。
会话路由目前通过本仓库的实验代理实现，尚未接入原生 frontend。

验证命令：

```bash
# 普通 CPU 环境，需 aiohttp 和 pytest。
python -m pytest -q experiments/agentx-ascend/test_session_router.py

# 服务启动后：两张卡分别执行跨多个缓存块的长输入及重复请求。
python experiments/agentx-ascend/smoke.py

# 在已安装对应 Ascend 插件补丁的服务器环境中，避免 cwd 导入未构建的源码包。
source "${REPO_ROOT}/activate-ascend.sh"
cd "${REPO_ROOT}/.."
python -m pytest --import-mode=importlib -q \
  "${REPO_ROOT}/vllm-ascend/tests/ut/patch/platform/test_mamba_prefill_config.py"
```

在服务器完整结果目录运行 `summarize.py RUN_DIR`，从官方 JSON、逐请求记录和 `benchmark.log` 生成
`result-summary.json`。收尾取消数读取 profiling 完成日志，因为被取消的请求可能
不会进入成功请求的 JSONL，不能把 JSONL 中零条取消记录解释为没有收尾取消。

### 2026-09-23 混合缓存保留对照结果

使用同一份插件代码，固定 Qwen3.5-9B、两张 910B2、TP1/DP2、BF16/eager、256K、
并发 16、sticky、cap1024 和总 batch budget=2048。两组都启用诊断，仅改变
`mamba_cache_retention.interval`。两次均从新启动的服务运行官方 900 秒场景；
正确性测试在单独启动的服务上及正式测量完成后执行，没有混入计时窗口。

| 配置 | 完成 / 错误 / 收尾取消 | 输出 tok/s | 平均 TTFT | P90 TTFT | cache read | total tok/NPU/s，含缓存输入 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 原保留策略，interval=null | 112 / 0 / 7 | 42.03 | 4.48 s | 11.66 s | 82.11% | 4743.35 |
| 稀疏保留，interval=8192 | 122 / 0 / 5 | 47.95 | 2.46 s | 4.26 s | 90.56% | 5195.53 |

本次观测：输出吞吐增加 14.1%，平均 TTFT 降低 45.1%，P90 TTFT 降低 63.5%，
缓存读取比例增加 8.45 个百分点。两组均 `submission_valid=true`，没有请求错误，
官方 `input_config` 除 URL/结果目录外一致，插件文件 SHA-256 也完全一致。
每组仍只有一次测量；闭环负载的请求到达顺序和完成请求组合不同，不能把这些幅度
当作稳定性能保证，也不能把含缓存输入的总吞吐解释为实际计算吞吐。

进程诊断快照中，原策略出现了 full-attention KV 可命中、GDN 状态却无法在相同位置
命中的回退；新策略的这一差距较小。该现象与 API 报告的缓存读取比例改善一致。
诊断包含 warmup 和调度重试，不能据此宣称节省了某个确切数量的重算 token。

21 项新增缓存测试与原有 15 项配置测试通过。两卡真实模型检查每轮共 22 个请求，
覆盖冷计算、原输入重放、完整用户续问、轮内分支及并发兄弟分支；生成 token 和结束原因一致，
所比较生成 token 的 logprob 最大绝对差为 0.001287（容差 0.02）。两张卡均观察到
首次 6144-token 分支没有检查点、后续兄弟分支命中 6144 token；追加输入命中 12288 token，
完全相同输入重放命中 11264 token。正式 benchmark 后复查也通过。

原始官方报告、manifest、诊断快照和验证记录在服务器
`${EXPERIMENT_DIR}/retention/`，机器可读对照为该目录的
`comparison.json`。完整逐请求数据和
服务日志保留在服务器 `${RESULTS_DIR}/mamba-retention/`。

```bash
# 在服务器 ${REPO_ROOT} 下重新检查实验条件并计算差异，不需要 NPU。
python experiments/agentx-ascend/compare_retention.py \
  experiments/agentx-ascend/retention/dense-c16-cap1024 \
  experiments/agentx-ascend/retention/sparse8192-c16-cap1024 \
  --output "${RESULTS_DIR}/mamba-retention/recomputed-comparison.json"

# 缓存生命周期与原配置保护回归测试，使用服务器的 Ascend 环境。
source "${REPO_ROOT}/activate-ascend.sh"
cd "${REPO_ROOT}/.."
python -m pytest --import-mode=importlib -q \
  "${REPO_ROOT}/vllm-ascend/tests/ut/patch/platform/test_mamba_retention.py" \
  "${REPO_ROOT}/vllm-ascend/tests/ut/patch/platform/test_mamba_prefill_config.py"
```

### 2026-09-23 第二轮：复用优先回收与仅保留复用边界

本轮先重启并复测第一轮 8192 配置，再测试新代码的两个配置。仍使用同一官方
AgentX harness、数据集、seed、900 秒场景、Qwen3.5-9B、TP1/DP2、C16、
sticky、cap1024 和 2048 总 token budget。三组每个 rank 的启动日志均报告
KV 容量为 1,195,656 token；不是通过扩大缓存池取得变化。

| 代码 / 配置 | 完成 / 错误 / 收尾取消 | 输出 tok/s | 平均 TTFT | P90 TTFT | cache read |
| --- | --- | ---: | ---: | ---: | ---: |
| 第一轮代码，8192，复测 | 122 / 0 / 5 | 47.95 | 2.34 s | 4.01 s | 90.59% |
| 新代码，8192，优先回收未复用的周期性检查点 | 123 / 0 / 6 | 48.42 | 2.16 s | 3.92 s | 91.59% |
| 新代码，0，仅保留复用边界及 decode 完整块 | 121 / 0 / 5 | 47.66 | 1.94 s | 3.64 s | 91.75% |

相对本轮第一行，8192 加新回收策略的输出吞吐增加 0.98%，平均 TTFT 降低 7.83%，
P90 TTFT 降低 2.36%；interval=0 的输出吞吐降低 0.61%，平均 TTFT 降低 17.04%，
P90 TTFT 降低 9.36%。每用户平均生成速度分别为 8.59、8.73、8.53 tok/s，
P90 分别为 9.20、9.46、9.30 tok/s。

这些差异仍不能视为稳定增益。两项新配置各只有一次测量，吞吐差异约 1%，
闭环请求顺序和完成请求组合也会变化。第一行与后两行是跨代码版本比较，包含
等待重试时保留共享边界的修正，不能把全部差异单独归因于回收策略。
后两行使用完全相同的插件哈希，但 interval 与优先回收开关都不同。
三组均 `submission_valid=true`、`was_cancelled=false`、请求错误为 0；
输入配置除 URL/结果路径外相同。所有比较已从官方报告重新计算并核对一致，完整报告保留在服务器。

新策略的最终诊断快照累计记录了 828 次优先选择周期性块淘汰，以及 3 次因实际引用而晋升。
它确实在缓存压力下触发，但后半程仍有状态检查点缺失引起的查询回退；保留更少中间状态
会付出额外重算的代价。诊断包含预热、重试且按 16 次查询采样，不是完整的执行 token 统计。

本轮选用 **8192 + prefer_reuse_boundaries=true** 作为服务器的实验配置：它在本次样本中
输出吞吐、每用户生成速度较高，TTFT 也低于复测基线。interval=0 是本次样本中首 token
延迟更低的选项。全局默认仍关闭该实验开关，当前结果不足以更改通用默认值。

测试总计 50 项通过（35 项缓存策略/生命周期测试及原有 15 项配置测试），插件 pre-commit
和实验脚本静态检查通过。两个候选配置均通过两卡 22 请求的冷计算/缓存恢复检查；
正确性检查位于正式测量之外。测试覆盖周期性块回收、被引用后的晋升、活跃状态保护、
批量分配和事件、淘汰/重置清理、块反复复用、关闭开关后的队列顺序，以及等待重试中共享边界被淘汰的情况。
选定配置重启后再次通过两卡 22 请求检查，前端健康检查为 HTTP 200，实际生成请求返回预期结果。
该次部署和校验记录位于服务器
`${EXPERIMENT_DIR}/retention-v2/validation/`。

官方报告、配置、代码哈希及比较结果在服务器
`${EXPERIMENT_DIR}/retention-v2/`。
完整日志保留于服务器 `${RESULTS_DIR}/retention-v2/`。
重跑单组实验（会停止指定的上一轮实验服务）：

```bash
source "${REPO_ROOT}/activate-ascend.sh"
python experiments/agentx-ascend/run_retention.py \
  --previous-run /path/to/previous-run \
  --run-dir /path/to/new-run \
  --interval 8192 --prefer-reuse-boundaries

# 在服务器 ${REPO_ROOT} 下跨第一轮代码和本轮代码比较，明确记录代码变化。
python experiments/agentx-ascend/compare_retention.py \
  experiments/agentx-ascend/retention-v2/reference8192 \
  experiments/agentx-ascend/retention-v2/priority8192 \
  --allow-code-change \
  --output "${RESULTS_DIR}/retention-v2/recomputed-comparison.json"
```

### 2026-09-23 Decode ACLGraph

本轮复用 Ascend 已有的 ACLGraph 和 GDN 图参数更新能力，启动器新增
`EXECUTION_MODE=decode-graph`；默认仍为 `eager`。图模式配置为
`mode=3`（VLLM_COMPILE）、`FULL_DECODE_ONLY`、捕获大小 `[1,2,4,8]`，
关闭可选的 `npugraph_ex`。该配置包含必要的编译及融合路径，收益不能全部归因于图回放本身。
prefill 和混合批次沿用该模式的非图执行路径。

当前 0.22.1 组合需要 `mode=3` 来初始化 Ascend 图参数；试用 `mode=0` 时，
GDN 捕获报 `NoneType ... conv1d_events`。该次启动失败记录保留在服务器，未计入性能结果。

两种模式固定 Qwen3.5-9B、TP1/DP2、BF16、256K、C16、sticky、cap1024、batch budget=2048，
以及 `interval=8192`、`prefer_reuse_boundaries=true`。
通过 `KV_CACHE_MEMORY_BYTES=38654705664` 固定每卡 36 GiB 的缓存预算，四组每个 rank
报告的实际容量均为 1,138,625 token；图模式实测每卡额外使用约 0.55 GiB 图内存。
此预算小于上一轮自动分配的缓存容量，须使用本轮的新 eager 基线比较。

正式测量前，在独立服务周期进行缓存命中后的短 decode 剖析。每卡 8 并发、每请求生成
32 token 的非剖析诊断耗时由约 4.00 秒降到 1.63 秒。剖析记录中，设备算子执行区间的
并集占比由约 16% 升到 50%–57%，每卡观测到 32 次 `aclmdlRIExecuteAsync`。
该占比是所记录 kernel 时间区间的统计，包含剖析开销，不是硬件利用率或官方 benchmark 分数。

两种模式分别通过两卡共 32 请求的状态正确性检查，覆盖冷计算、重放、续问、轮内分支和
7 个并发兄弟请求。图模式还直接对照 eager 的冷计算结果：生成 token、结束原因一致，
所比较生成 token 的 logprob 在 0.02 绝对容差内。

正式对照沿用前述官方 harness、数据集和 seed，按 eager → graph → graph → eager
顺序运行，每组都重启服务，执行完整的 `inferencex-agentx-mvp` 900 秒场景。
正式窗口没有启用 profiler，也没有混入正确性请求。

| 配置 | 完成 / 错误 / 收尾取消 | 输出 tok/s | 平均 TTFT | P90 TTFT | cache read |
| --- | --- | ---: | ---: | ---: | ---: |
| eager-1 | 123 / 0 / 4 | 48.42 | 2.237 s | 4.197 s | 91.87% |
| graph-1 | 205 / 0 / 5 | 88.26 | 1.957 s | 3.743 s | 92.87% |
| graph-2 | 208 / 0 / 2 | 89.68 | 1.925 s | 3.645 s | 92.93% |
| eager-2 | 122 / 0 / 4 | 48.14 | 2.268 s | 3.907 s | 91.79% |

对每种模式的两次官方报告指标取算术平均，输出吞吐从 **48.28 提升至 88.97 tok/s（+84.3%）**，
平均 TTFT 从 **2.252 降至 1.941 秒（-13.8%）**；两次 P90 TTFT 的均值从 4.052 降至
3.694 秒（-8.8%，不是合并所有请求后重新计算的 P90）。每用户平均生成速度从
8.67 提升至 33.88 tok/s；闭环总吞吐还受 prefill、工具等待和请求组合影响，增幅不同。

四组均 `submission_valid=true`、`was_cancelled=false`，请求错误为 0。
对照脚本核对了官方输入配置、启动脚本与插件 SHA、缓存策略、预算和实际容量。
两次配对输出吞吐增益分别为 82.3% 和 86.3%；每种模式仍只有两次运行、同一个 seed，
结果仅支持本次 256K 过滤负载下的观测，不代表 1M 上下文场景或其他负载的稳定增益。
本轮没有修改 vLLM-Ascend 插件源码，改动位于实验启动、剖析及验证脚本。

四组测量结束后，服务器重新启动选定的图模式服务，再次通过两卡 32 请求的跨模式
冷计算/缓存恢复检查，前后端健康检查均为 HTTP 200，前端生成请求返回预期结果。
部署记录指向 `results/decode-graph/selected-decode-graph/`，复测汇总为
`results/decode-graph/aggregate.json`；当前实例关闭 profiler，启动器的通用默认仍为 eager。

入口支持：

```bash
# 服务器 ${REPO_ROOT}；正式测量不启用 PROFILE_DIR。
EXECUTION_MODE=decode-graph KV_CACHE_MEMORY_BYTES=38654705664 \
  LONG_PREFILL_TOKEN_THRESHOLD=1024 MAMBA_RETENTION_INTERVAL=8192 \
  MAMBA_PREFER_REUSE_BOUNDARIES=1 MAMBA_RETENTION_DIAGNOSTICS=1 \
  bash experiments/agentx-ascend/serve.sh

# 用 run_retention.py 启动一轮完整官方场景，结果目录必须尚不存在。
python experiments/agentx-ascend/run_retention.py \
  --previous-run /path/to/previous-run --run-dir /path/to/new-run \
  --interval 8192 --prefer-reuse-boundaries \
  --execution-mode decode-graph --kv-cache-memory-bytes 38654705664

# 比较本轮同容量、同缓存策略的 eager 与图执行结果。
python experiments/agentx-ascend/compare_retention.py \
  experiments/agentx-ascend/results/decode-graph/eager-1 \
  experiments/agentx-ascend/results/decode-graph/graph-1 \
  --allow-execution-change \
  --output "${RESULTS_DIR}/decode-graph/recomputed-comparison.json"
```

`run_retention.py --serve-only --profile-dir ${RESULTS_DIR}/profile/traces` 单独启动剖析服务；
`profile_decode.py` 采集短诊断，`summarize_profile.py` 在服务器汇总解析后的 profiler CSV。
`hybrid_cache_smoke.py --reference-file ... --concurrent-siblings 7` 检查跨执行模式一致性。
执行模式对照须给 `compare_retention.py` 显式传入 `--allow-execution-change`；
它同时要求缓存保留策略、缓存预算、每个 rank 报告的实际容量和启动脚本哈希相同，并拒绝混入剖析运行。
完整记录位于服务器 `${RESULTS_DIR}/decode-graph/`。

### 2026-09-23 图模式基线核查与 CPU 编号映射修复

启动日志暴露了 CPU 绑核失败：`npu-smi info -t topo` 报告物理 NPU4/NPU5，
`npu-smi info -m` 则将其映射为逻辑设备 0/1。原代码直接用物理拓扑编号查询逻辑设备，
即使进程允许使用 CPU 0–191，也会误报 cpuset 与 NUMA 亲和冲突。
`vllm_ascend/cpu_binding.py` 现在将单芯片板卡的拓扑编号转换为 chip logic ID；
多芯片拓扑沿用原有编号解释。该改动是已有 CPU 亲和机制的缺陷修复，不是新的优化算法。

修复后真实 worker 的主线程及普通子线程分别绑定到 CPU 98–141 和 50–93，
ACL 线程分别使用 142 和 94，释放线程分别使用 143 和 95，两组互不重叠。
容器没有 `migratepages`，因此没有进行已有内存页的 NUMA 迁移；本轮绑核对照仅验证
进程和线程亲和性，不能把它解释为完整的 CPU/内存 NUMA 调优。
CPU 绑核测试共 75 项通过，包括容器重映射、编号重排与多芯片行为回归；pre-commit 通过。
三种候选配置分别通过两卡 32 请求的生成、缓存恢复和与 eager 冷计算对照检查。

短诊断固定相同的缓存前缀、每请求输出 32 token，每种并发数重复五次，报告耗时中位数。
这些结果不是官方 AgentX 分数，也不能代表长 prefill 或完整多轮流量。

| 每卡并发请求数 | 不绑核，npugraph_ex 关闭 | 绑核，npugraph_ex 关闭 | 绑核，npugraph_ex 开启 |
| --- | ---: | ---: | ---: |
| 1 | 0.885 s | 0.935 s | 0.863 s |
| 3 | 1.363 s | 1.461 s | 1.269 s |
| 5 | 1.454 s | 1.451 s | 1.353 s |
| 8 | 1.483 s | 1.501 s | 1.258 s |

单独绑核没有在本次短诊断中显示收益；是否采用由端到端测量决定。
异步调度已经默认启用，不计作新增优化。`npugraph_ex` 是已有的编译期优化能力，
当前只验证普通 FX 优化，没有开启 static kernel 或 super kernel。

完整官方场景固定同一 launcher 和插件 SHA、每卡 36 GiB、原生 256K、TP1/DP2、C16、
同一 seed 和缓存保留策略；每组重启服务，计时 900 秒，均开启相同的批次诊断日志。

| 配置 | 完成 / 错误 / 收尾取消 | 输出 tok/s | 平均 TTFT | P90 TTFT | cache read |
| --- | --- | ---: | ---: | ---: | ---: |
| 不绑核，npugraph_ex 关闭 | 207 / 0 / 2 | 89.28 | 2.007 s | 4.010 s | 92.80% |
| 绑核，npugraph_ex 关闭 | 206 / 0 / 3 | 88.87 | 2.005 s | 4.162 s | 92.91% |
| 绑核，npugraph_ex 开启 | 210 / 0 / 3 | 90.35 | 1.821 s | 3.518 s | 93.00% |

这三组报告均有效、无请求错误。单独绑核后的吞吐变化为 -0.46%，平均 TTFT 为 -0.13%，
P90 TTFT 为 +3.80%；各配置仅一次测量，不能据此确认稳定收益或退化。
在绑核配置上开启 `npugraph_ex`，输出吞吐增加 1.66%，平均 TTFT 降低 9.17%，
P90 TTFT 降低 15.47%。端到端吞吐提升小于短 decode 诊断中的改善；
这些也是一次测量的结果，不代表其他 seed、负载或硬件上的稳定收益。

随后固定绑核和 `npugraph_ex` 开启，单独比较原生缓存策略与当前的稀疏保留/回收组合。
两组同样使用每卡 36 GiB，实际容量均为 1,138,625 token；官方报告均有效、未被取消、
请求错误为 0。原生策略也先在独立服务周期通过两卡 32 请求正确性检查。

| 缓存策略 | 完成 / 错误 / 收尾取消 | 输出 tok/s | 平均 TTFT | P90 TTFT | cache read |
| --- | --- | ---: | ---: | ---: | ---: |
| 原生保留与回收 | 183 / 0 / 2 | 78.91 | 4.968 s | 15.694 s | 80.94% |
| 8192 + prefer_reuse_boundaries | 210 / 0 / 3 | 90.35 | 1.821 s | 3.518 s | 93.00% |

当前策略组合相对原生策略，输出吞吐增加 **14.49%**，平均 TTFT 降低 **63.34%**，
P90 TTFT 降低 **77.58%**，cache read 增加 12.06 个百分点。
第二行复用前表第三行，不是另一次重复运行。对照衡量的是稀疏保留、未缓存块优先复用、
复用边界保护等组合效果，不能单独归因于某一项策略。每用户平均生成速度则从
32.02 降到 30.55 tok/s；闭环下完成请求组合变化，不能将总吞吐提升解释成所有请求的 decode 都加速。
各配置仍只有一次测量、一个 seed，需要多次复测才能确认增益的稳定程度。

本轮服务器保留 **decode ACLGraph + npugraph_ex + CPU 绑核 + 8192 稀疏保留/回收组合**，
继续使用 TP1/DP2、36 GiB/卡、sticky、cap1024 和 batch budget=2048；保留批次诊断，关闭 profiler。
选择绑核是为了保留已修复的确定性线程分配，不代表本次实验确认了其性能收益。
最终实例重启后再次通过两卡 32 请求的跨模式正确性检查，前后端健康检查为 HTTP 200，
实际生成返回预期结果，五个插件文件的源码与运行时哈希一致。
当前部署记录已更新为 `results/active-service/deployment.json`，服务日志与正确性结果位于
`results/graph-baseline/selected-service/`。启动器的通用默认仍为 eager，`npugraph_ex` 默认关闭。

实验入口增加 `--npugraph-ex`、`--no-cpu-binding`、`--interval native` 和
`--batch-diagnostics`。最后一项使用 vLLM 原生批次日志；`summarize_batches.py` 汇总
prefill、decode 和混合批次。其耗时是引擎观察到的等待/处理区间，受异步重叠影响，
不能当作 NPU 执行时间，也不能把纯 decode 批次计数直接当作图回放计数。
汇总时传入 `--benchmark-log RUN_DIR/benchmark.log`，可按官方完成日志中的
profiling 起止时间剔除预热和收尾；服务日志只有秒级时间戳，窗口边界存在一秒内的精度限制。
在开启 `npugraph_ex` 的正式窗口中，两卡纯 decode 批次分别占约 96.2% 和 96.9%。
原生缓存组的 rank 0 混合批次占 15.34%，稀疏策略组则为 2.36%；rank 1 分别为
2.88% 和 2.68%。这与原生组缓存压力主要集中于一个副本的诊断相符，但不是逐请求配对的因果证明。
混合批次的主机侧耗时较长，但仅凭批次比例和这类耗时，无法证明延迟 prefill
能够改善总吞吐与首 token 延迟。本轮没有据此新增调度策略。

缓存对照使用相同插件代码和诊断开关；`--interval native` 使保留与回收走原生分支，
同时关闭优先回收周期性检查点，并非换用另一套未经修改的安装包。复现入口为：

```bash
# 在服务器 ${REPO_ROOT}，激活 Ascend 环境后执行；结果目录必须尚不存在。
python experiments/agentx-ascend/run_retention.py \
  --previous-run /path/to/previous-run --run-dir /path/to/native-run \
  --interval native --execution-mode decode-graph --npugraph-ex \
  --kv-cache-memory-bytes 38654705664 --batch-diagnostics

# 其他参数保持相同，使用 --interval 8192 --prefer-reuse-boundaries 跑另一组。
python experiments/agentx-ascend/compare_retention.py \
  /path/to/native-run /path/to/sparse-run \
  --output "${RESULTS_DIR}/cache-comparison.json"
```

完整诊断、代码快照、四组正式报告和三个对照汇总位于服务器
`${RESULTS_DIR}/graph-baseline/`。

## NPU ForkAttention 算子原型

2026-09-23 在两张 910B2 上实现并验证了独立的 ForkAttention 算子。
该阶段模型服务仍使用上一节的 attention 路径；以下均为合成算子负载，**不是新的官方 AgentX 成绩**。
Qwen3.5-9B 的实际 full-attention 几何为 16 个 query heads、4 个 KV heads、head dimension 256，
32 层中有 8 层 full attention。GDN 不由该算子处理。

### 实现

- [fork_plan.py](../vllm-ascend/vllm_ascend/attention/fork_plan.py)
  根据 CPU block table 的物理页身份验证一个候选组的公共前缀，按完整页切分共享段，
  每个 query 保留非空私有尾部。NumPy 路径批量比较页、校验有效范围，避免逐页 Python 标量循环。
- [fork_attention.py](../vllm-ascend/vllm_ascend/ops/fork_attention.py)
  将同组分支作为共享段的多个 query 行，用一次 paged FIA 调用处理所有共享分片和私有尾部。
  使用现有 Ascend C attention 内核；本轮没有新写 Cube attention 内核。
  不整理或复制 KV，只打包较小的 Q 和页描述符。
- [Triton 辅助内核](../vllm-ascend/vllm_ascend/ops/triton/fork_attention.py)
  分别完成 Q 打包、FP32 稳定 LSE 加权合并，结果写回 BF16/FP16。
  合并接受最多 33 个部分结果，避免逐段调用合并算子以及整张中间输出的额外 FP32 转换。
- `ForkAttentionGraph` 使用 ExternalEvent 和 graph-task update 更新 FIA 的长度参数及固定地址页表。
  已验证尾部增长、跨页、私有页重排和长度回退；超出捕获的组大小、分片数或页表容量时明确拒绝。
  普通 `NPUGraph` 捕获仍固定原 plan，不能把改变后的长度直接用于旧图。

混合缓存的 1024-token **管理块**会由 runner 拆为 128-token **内核页**。
本原型只接受后者；直接传入 1024-token 页会在 Python 层被拒绝。
当前范围是 2～8 个单 token decode 查询、BF16/FP16、普通全注意力、连续 Q/K/V。
head sizes 64/128/256 的数值测试使用 GQA4；性能矩阵固定模型的 16/4/256 配置。

### 算子性能与对照

每卡扫描 2/3/4/8 分支、8K/32K/64K 共享前缀、128/1024 token 私有尾部、
1/4/8/16/32 个前缀分片，共 24 个形状、每形状 5 个候选。
三条路径共用同一份碎片化物理 KV：服务当前使用的因果 paged FIA、
ForkAttention、以及不合并共享查询的分片对照。
每个候选采用三条路径的全部 6 种执行顺序，每次计时连续回放 50 次固定 plan 图，报告中位数。
NPU event 时间包括 Q 打包、attention、结果合并及图回放间隙；不包括 CPU 规划或逐步图任务更新。

下表取尾部 128 token；32K 固定 4 分片，64K 固定 16 分片。
卡 0 使用后续确认轮，卡 1 使用完整矩阵；并非挑每行最快分片后的结果。

| 分支 / 共享前缀 | 分片 | 卡 0 FIA → Fork（µs） | 卡 1 FIA → Fork（µs） | 卡 1 仅分片（µs） |
| --- | --- | --- | --- | --- |
| 2 / 32K | 4 | 278.46 → 113.91 | 277.48 → 116.79 | 187.45 |
| 4 / 32K | 4 | 286.01 → 115.81 | 289.55 → 126.12 | 261.69 |
| 8 / 32K | 4 | 553.45 → 124.06 | 561.32 → 131.47 | 458.50 |
| 2 / 64K | 16 | 827.85 → 240.22 | 832.53 → 247.45 | 388.55 |
| 4 / 64K | 16 | 855.55 → 272.32 | 837.67 → 264.45 | 559.63 |
| 8 / 64K | 16 | 1723.64 → 292.68 | 1706.28 → 290.88 | 973.31 |

这些结果支持继续做模型级集成。分片和共享查询分组都有贡献；不能把总加速全部归于减少 HBM 读取。
单个长段直接执行在小分支数下可能变慢，分片越多也不一定更快；目前没有把候选扫描结果固化为生产准入规则。
单实例显式缓冲在本轮形状中约 369～373 MiB，以 FIA 保守最大 workspace 为主。
接入模型时需要按执行流复用工作区，避免每层、每个图重复保留。

独立 profiling 的 4 分支/32K/4 分片用例中，每次 Fork 回放为 Q 打包、FIA、结果合并三项，
5 次回放及两条对照共核对 35 条设备记录。
CANN 的 AIC GM→L1 搬运计数由 543232 降至 142848 KB，约减少 74%；
仅分片对照为 543488 KB。这支持共享分组降低重复搬运，但该计数不是直接测得的 HBM DRAM 流量。
profiler 报告了 ACL→NPU flow 关联解析错误，因此不使用它推断 host/device 重叠或完整调用关联；
设备记录按有同步隔开的已知回放顺序分组，并逐项核对 kernel 名称。
独立、固定 CPU 的重复规划测试中，最终 NumPy 路径的 4 分支/32K 中位耗时约 114 µs；
这是额外的 CPU 成本，尚需在服务中复用计划，并计入端到端评估。

### 正确性和复现

CPU 规划测试覆盖完整页边界、无共享/共享后缀、页重排、长度回退、无效页及 NumPy 输入。
两卡各通过 22 项 NPU 测试，覆盖 FP32 dense reference 对照、不同尾长、碎片化页、
兄弟分支隔离、图回放中 Q/V 数值变化、大 LSE 的 33 段合并，以及不重新捕获图的元数据更新。
数值验证使用 BF16/FP16 对应容差，不声称逐位相等，也尚未验证完整模型生成 token 一致性。

```bash
# 在服务器激活 Ascend 环境，并将三个新算子文件部署到实际安装的插件。
python experiments/agentx-ascend/benchmark_fork_attention.py \
  --device 0 \
  --output "${RESULTS_DIR}/NEW_RUN/matrix.json"

# 独立 profiling，不能混入无 profiler 的性能报告。
python experiments/agentx-ascend/benchmark_fork_attention.py \
  --branches 4 --prefixes 32768 --tails 128 --splits 4 \
  --profile-dir "${RESULTS_DIR}/NEW_PROFILE/trace" \
  --output "${RESULTS_DIR}/NEW_PROFILE/timing.json"
```

完整代码快照、矩阵、逐轮计时、测试输出和 profiler 数据均在服务器
`${RESULTS_DIR}/fork-attention-v1/`。
关键记录为 `matrix-rank0.json`、`matrix-rank1.json`、`confirm-rank0.json`，以及
`npu-tests-dynamic-rank0.txt`、`npu-tests-dynamic-rank1.txt`。
最终代码另通过 22 项 CPU 规划测试和卡 0 的 22 项 NPU 复测；记录为
`planner-tests-final-v2.txt`、`npu-tests-final-v2.txt`，代码快照为 `final-code.zip`。
设备诊断摘要和原始数据分别为 `profile-key-summary.json`、`profile-memory/`。
后续的在线接入见下节。Agent Hints 尚未接入。

## NPU ForkAttention 在线 decode 接入

当前以 **FIA 版 ForkAttention 作为后续开发与实验基线**：共享前缀规划、Q 打包、CANN FIA
分段计算及结果合并。优化优先级为量化真实流量的共享机会、调整准入和分片策略、降低规划及
回退开销，并用官方 AgentX 验证端到端收益。直接计算实验的实现已移除，历史数据及源码快照仅在服务器归档。
正式服务仍保留原生图路径，ForkAttention 默认关闭；选择 FIA 版作为开发主线不代表已验证在线收益。

2026-09-24 新增默认关闭的 `additional_config.fork_attention`。当前验证目标为
Qwen3.5-9B、BF16、TP1/PP1、两卡 DP；仅处理普通 causal full attention 的单 token decode。
GDN、prefill、speculative decode、CP、DBO、滑窗、sinks、量化 KV 不属于这一优化路径。
不支持的全局并行配置在启用时拒绝；没有可用共享组或不支持的 attention 形状沿用原生路径。

### 接入和资源复用

- `model_runner_v1.py` 只向 metadata builder 提供已经存在的 CPU 内核页表视图。
  不增加 NPU→CPU 页表回读，也不按会话名推断共享。
- [fork_batch.py](../vllm-ascend/vllm_ascend/attention/fork_batch.py) 按共同的起始物理页分组，
  一批可以包含多个不相邻的共享组及独立请求。每个组按完整页切分共享前缀，当前 token 留在尾部。
  默认共享前缀门槛为 32768 token；不足 64K 使用 4 分片，达到 64K 使用 16 分片。
  这是基于已测形状的初始规则，不是全形状最优调度器。
- 每个 metadata builder 只保存一个最近的计划。每步重新比较 CPU 物理页和有效页数；
  页表未变时只更新尾长，并让同组各层共用结果。页重分配、重排、跨页或长度回退会重新规划。
- [fork_decode.py](../vllm-ascend/vllm_ascend/ops/fork_decode.py) 在图捕获前分配缓冲区。
  同一 KV 组的各层、各图尺寸复用一份 FIA 工作区、Q/部分结果和描述符缓冲；不复制 KV。
  只在页描述符、查询映射或启用状态变化时传输这些小型数据；逐 token 长度通过已有 FIA task update 更新。
- `AscendAttentionBackendImpl` 捕获固定的 pack→FIA→merge 结构。
  每次图回放可更新 FIA 的形状、长度、页表和输出地址；没有共享组时 FIA 直接写最终输出，
  pack/merge 内核执行空分支。ExternalEvent 等待位于 pack 之前，保证元数据更新先于 Q 打包。
  图补齐行不进入共享组；Fork 分支把补齐行输出置零。

启用后，无共享组的 2～8 token 图仍有两个空内核的开销，因此不能把“自动回退”解释成零开销。
禁用配置沿用原生图。池不支持并发模型执行，启用时拒绝 DBO；DP 副本各自持有缓冲区。
这版不会为了创造共享机会重排请求、延迟 decode 或改变 AgentX 输入。

### 接入验证

CPU 规划共 34 项测试通过；原有配置及 attention backend 的 35 项测试通过。
两张卡分别通过 26 项 NPU 测试，包括原型的 22 项和新增的 BF16/FP16 图内切换、
两层共用缓冲区、补齐行、多共享组、原生回退，以及 32K/64K 长前缀与原生 FIA 对照。

独立的主机规划诊断中，4 分支/32K 的单次原型重建中位数约 139 µs，批次计划重建约 179 µs，
复用约 86 µs；8 分支/64K 分别约 212、260、101 µs。
复用仍包含页表身份检查和新尾长生成，不是零成本；这些诊断不等于服务延迟改善。

最初的并发 HTTP 对照中，50 个请求的 2400 个生成 token 全部一致，
但逐 token logprob 的 0.03 绝对容差未通过。Fork 对原生参考的最大差异为 0.105709，
原生服务重跑对同一参考的最大差异为 0.073591；平均绝对差异分别为 0.000382、0.000254。
最大差异集中在混合批次同一请求的第 3 个输出 token，不能将原生重复运行的波动全部归因于 Fork。
原始失败结果保留为 `model-fork-complete.json`、`model-native-repeat.json`。
后续检查改为单个 HTTP 请求提交整个批次，并用不同前缀区分共享组，减少到达时序差异；
这仍不保证内部调度完全相同。最终检查显式记录使用的 logprob 容差，同时要求所有生成 token 一致。
整批提交的最终两卡检查通过：50 个请求、2400 个生成 token 一致，最大 logprob 绝对差异
0.087215，低于本轮显式指定的 0.1。它不满足最初的 0.03 门槛，也不是逐位一致性证明。
参考及最终结果分别为 `model-native-batched.json`、`model-fork-final.json`；
复现使用 `fork_decode_smoke.py --batch-request --reference ... --logprob-atol 0.1 --output ...`。

`fork_decode_smoke.py --batch-request` 记录批量接口报告的缓存命中数，不能把它当作逐请求命中数。
vLLM 0.22.1 的非流式批量 completion 返回最后一个 prompt 的 `num_cached_tokens`，
因此脚本不再将该数值除以请求数。共享是否实际参与执行以 CPU 物理页规划和后端执行日志核对。

### 官方 AgentX 对照

两组分别重启服务，使用相同的官方 harness、数据集、seed 和 900 秒场景；
固定 TP1/DP2、原生 256K、36 GiB/卡、sticky、cap1024、batch budget=2048、
8192 稀疏保留、CPU 绑核、decode ACLGraph 和 `npugraph_ex`。
启动脚本及 11 个插件文件的 SHA 一致；实际缓存容量均为每卡 1,138,625 token。
两组唯一的 Fork 配置差异是 `enabled`，共享门槛均记录为 32768，诊断开关一致。
正确性测试和 profiler 不在正式测量窗口内运行。

| 配置 | 完成 / 请求错误 | 输出 tok/s | 平均 TTFT | P90 TTFT | cache read |
| --- | --- | ---: | ---: | ---: | ---: |
| 原生图路径 | 205 / 0 | 88.79 | 1.869 s | 3.555 s | 92.80% |
| 启用 ForkAttention | 205 / 0 | 88.26 | 2.016 s | 3.716 s | 92.85% |

两组均 `submission_valid=true`、`was_cancelled=false`。
排空超时分别取消 3、5 个未完成请求；该计数独立于报告中的请求错误数（两组均为 0）。
启用后吞吐变化为 **-0.59%**，平均 TTFT 为 +7.83%，P90 TTFT 为 +4.54%。
两个 worker 的最后一条累计执行诊断分别为 14,848、14,080 批次，`selected` 均为 0，
且全程没有首次共享执行记录。这些计数包含预热、每 256 批次采样一次，
不能当作精确的正式窗口批次数；它们支持本轮未触发 32K 门槛下的共享路径这一判断。

本轮没有确认 ForkAttention 的 AgentX 端到端收益。较高的 prefix-cache 命中率
并不保证同卡同批次存在足够长的共享物理前缀；不能把独立算子的加速倍数套用到这里。
只有一次对照、一个 seed，闭环请求组合和到达时序可能不同，无法据此确认稳定的退化幅度，
也不能把 TTFT 变化单独归因于两个空内核或 CPU 规划。
后续需要先量化更短共享前缀及父子分支的同卡并发机会，再评估路由、调度或 Agent Hints；
本轮没有实现新的 Agent Hints 协议。

服务器最终保留原生图路径，保留此前的混合缓存、绑核、`npugraph_ex` 和 TP1/DP2 配置。
ForkAttention 默认关闭，可通过下述开关复现。正式报告为 `official-native/`、`official-fork/`，
对照为 `official-comparison.json`，诊断及验证摘要为 `final-summary.json`，冻结代码为 `benchmark-code.zip`。
最终服务记录在 `selected-native/`；活动指针为 `results/active-service/deployment.json`。

### 复现入口

```bash
# 服务器 ${REPO_ROOT}，激活 Ascend 环境后执行。
ENABLE_FORK_ATTENTION=1 FORK_DIAGNOSTICS=1 FORK_MIN_SHARED_TOKENS=32768 \
  EXECUTION_MODE=decode-graph ENABLE_NPUGRAPH_EX=1 \
  KV_CACHE_MEMORY_BYTES=38654705664 LONG_PREFILL_TOKEN_THRESHOLD=1024 \
  MAMBA_RETENTION_INTERVAL=8192 MAMBA_PREFER_REUSE_BOUNDARIES=1 \
  bash experiments/agentx-ascend/serve.sh

# 正式对照由现有入口重启服务并运行完整官方场景，仍为 TP1/DP2。
python experiments/agentx-ascend/run_retention.py \
  --previous-run /path/to/previous-run --run-dir /path/to/new-run \
  --interval 8192 --prefer-reuse-boundaries --execution-mode decode-graph \
  --npugraph-ex --kv-cache-memory-bytes 38654705664 --batch-diagnostics \
  --fork-attention --fork-diagnostics
```

对照组移除 `--fork-attention`，其余设置一致。`compare_retention.py --allow-execution-change`
记录 Fork 配置差异，同时核对官方输入、保留策略、插件 SHA、启动脚本和实际 KV 容量。
所有原始记录位于服务器 `${RESULTS_DIR}/fork-attention-v2/`。

## 已移除的直接计算实验

2026-09-24 的直接 Triton 原型通过数值验证，但性能低于 FIA 版，未接入在线服务。
现已从本地仓库、服务器源码和已安装插件中删除对应内核、封装、测试及独立 benchmark，
并清理专用 profiler 选项；后续以 FIA 版为开发与实验基线。

保留关键历史数据：Qwen3.5-9B 的 16 Q heads / 4 KV heads、D=256、BF16、私有尾长 128，
卡 1 固定请求 16 分片（1K 实际为 8），每个形状轮换四条路径测量 8 轮、每轮 50 次图回放。
下表为完整算子耗时中位数，单位 µs，不含 CPU 规划和描述符 H2D；卡 0 趋势一致。

| 分支 / 共享前缀 | 原生 FIA | FIA 版 ForkAttention | 已移除的直接原型 |
| --- | ---: | ---: | ---: |
| 2 / 1K | 34.95 | 55.91 | 63.79 |
| 4 / 8K | 100.83 | 78.53 | 234.05 |
| 8 / 32K | 555.93 | 146.50 | 939.91 |

这些是算子实验，不是官方 AgentX 成绩。剖析显示主要差距在 attention 主体；
直接原型减少 Q 打包不足以抵消计算成本。原始数据及当时源码快照保存在服务器
`${RESULTS_DIR}/fork-attention-v3/`。
审核后的源码另归档于
`${RESULTS_DIR}/fork-attention-review-20260924/reviewed-source.zip`。
历史归档不属于当前可用实现，不再保留已删除脚本的启动命令。

## 代码审核与边界修复（2026-09-24）

审核覆盖 CPU 规划、算子封装、图更新、多组 serving 缓冲及实验入口。FIA 主线保留以下修复：

- 缓存命中的批规划器原先会把小数序列长度静默截断。现在在缓存查询前统一校验整数长度，
  并先扩为 int64 再计算页数，避免 int32 加法溢出；设备描述符拒绝超出 int32 范围的值。
- 单组 plan 原先只依赖工厂函数校验，直接构造或 `dataclasses.replace` 可注入负页号、
  缺失分段或错误 query 映射。现在不可变 plan 在构造时验证页数、分段覆盖、前缀长度与整数类型。
- FIA 原型及其图回放补齐当前 NPU device 检查；无效图更新在修改缓冲前拒绝。
- 旧算子 benchmark 提前检查分支、分片及前缀对齐，支持显式测试 128-token 小前缀；
  模型 smoke 脚本创建输出父目录并检查正数长度。运行 manifest 纳入共享规划模块的 SHA。

审核时 CPU 规划 **56 项**、原有配置及 attention backend **35 项**通过；两张 NPU 分别通过
**53 项**算子/图测试，其中 **26 项**属于随后移除的直接计算原型，FIA 版保留 **27 项**。
覆盖共享/私有尾段、长前缀、图切换、跨页更新及非法输入拒绝。
插件适用的 pre-commit 检查、实验脚本 Ruff 与 shell 语法检查通过。
本轮未修改计算内核，也没有新的性能或官方 AgentX 结论。
原始复现、测试日志及服务健康记录保存在服务器
`${RESULTS_DIR}/fork-attention-review-20260924/`。

移除直接原型后，两张 NPU 的 FIA 算子及图切换测试各 **27 项通过**；已删除模块无法再导入，
活动服务的 12 个插件文件哈希保持一致，前后端健康检查均为 HTTP 200。
移除及验证记录位于上述服务器目录的 `direct-removal/`，服务配置未调整。
