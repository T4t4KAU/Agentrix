# AgentX 在 vLLM-Ascend 上的路由、调度与混合缓存优化

## 目标和运行边界

以 [vLLM x AgentX 文章](https://vllm.ai/blog/2026-09-08-vllm-agentx)
为优化线索，以官方 [AgentX harness](https://github.com/SemiAnalysisAI/agentx-harness)
做端到端验证。第一阶段固定 Qwen3.5-9B、两张 Ascend 910B2 64GB、TP1/DP2、
BF16、eager、原生 262144 上下文、16 个 session trees，只改变路由或 prefill cap。
后续混合缓存对照固定 sticky + cap1024，只改变本地检查点保留策略。

运行环境是 vLLM 0.22.1 + vLLM-Ascend v0.22.1rc1（子模块
`da9b47a226f2d1b5428f4658af5c4bfe9813dbeb`，包含配置校验和混合缓存保留补丁）。本仓库 `vllm/`
是 0.28.0 CUDA 实验分支，不能直接安装到这个 Ascend 环境。
服务器使用独立 `/data/Agentrix/.venv`，vLLM 源码在 `/data/Agentrix/vllm-0.22.1`。
Ascend 插件当前装为 wheel；修改源码后须部署对应文件或重新构建，不能把源码同步误当成运行时已更新。

## 实验记录保存规则

本地只保留代码、复现方法和本文中的关键配置、结果、验证结论及局限。
后续直接在服务器运行和分析实验，不再将每轮日志、JSON 报告、manifest、诊断快照或测试输出下载到本地。
新实验的完整产物统一保存在服务器 `/data/Agentrix/experiments/agentx-ascend/results/`。
下文的实验记录路径均指服务器；历史归档仍在服务器同项目的 `baselines/`、`comparisons/`、
`retention/`、`retention-v2/` 和 `validation/` 目录，均相对于 `/data/Agentrix/experiments/agentx-ascend/`。
2026-09-23 清理本地副本前，已逐文件核对 81 份服务器归档的 SHA-256。

## 文章中的优化如何落地

| 优先级 | 优化 | 当前代码与差距 | 验证方法 |
| --- | --- | --- | --- |
| P0 | 会话粘性路由 | native DP 已接受 `X-data-parallel-rank`；实验代理把同一 `X-Session-ID` 固定到同一 rank，新会话轮转。CUDA 分支的 prefix/session-aware router 尚未迁入 0.22.1 | 对比 native、仅首轮固定和 sticky，统计缓存命中、TTFT、每用户输出速率和吞吐 |
| P0 | 长 prefill 分块，减轻队头阻塞 | 0.22.1 原生 scheduler 和 Ascend 自有 scheduler 已消费 `long_prefill_token_threshold`；无需复制整套调度器。新增 Ascend 混合块配置校验 | 保持 sticky、batch token budget=2048，比较 cap=0 与 1024 |
| P1 | 会话/分支边界的混合缓存保留 | 新增默认关闭的稀疏检查点保留、共享前缀边界补存和未缓存块优先回收；沿用现有 GDN/卷积状态复制 | 缓存管理器测试、两卡冷计算/状态恢复一致性，以及固定 sticky/cap1024 的官方 AgentX 对照 |
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
`/data/Agentrix/experiments/agentx-ascend/retention-v2/validation/v1-to-v2.patch`，
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

原始启动脚本、manifest、summary 和 harness JSON 原样保存在服务器
`/data/Agentrix/experiments/agentx-ascend/baselines/`。这些脚本是历史记录，复跑请用上面的
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
`/data/Agentrix/experiments/agentx-ascend/comparison.json`；原始官方 JSON、
manifest、路由计数和 summary 在同目录的 `comparisons/`。完整日志和逐请求数据在服务器
`/data/Agentrix/experiments/agentx-ascend/results/`。

验证包括 15 项 Ascend 配置回归测试、7 项代理路由测试、真实 Qwen3.5 配置构建
（512 拒绝，0/1024 接受），以及两卡各两次 8219-token prompt 的生成和缓存复用检查。
每张卡的重复请求复用了 8192 token。配置验证日志在服务器
`/data/Agentrix/experiments/agentx-ascend/validation/`，API 检查记录在
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
source /data/Agentrix/activate-ascend.sh
cd /data
python -m pytest --import-mode=importlib -q \
  /data/Agentrix/vllm-ascend/tests/ut/patch/platform/test_mamba_prefill_config.py
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
`/data/Agentrix/experiments/agentx-ascend/retention/`，机器可读对照为该目录的
`comparison.json`。完整逐请求数据和
服务日志保留在服务器 `/data/Agentrix/experiments/agentx-ascend/results/mamba-retention/`。

```bash
# 在服务器 /data/Agentrix 下重新检查实验条件并计算差异，不需要 NPU。
python experiments/agentx-ascend/compare_retention.py \
  experiments/agentx-ascend/retention/dense-c16-cap1024 \
  experiments/agentx-ascend/retention/sparse8192-c16-cap1024 \
  --output /data/Agentrix/experiments/agentx-ascend/results/mamba-retention/recomputed-comparison.json

# 缓存生命周期与原配置保护回归测试，使用服务器的 Ascend 环境。
source /data/Agentrix/activate-ascend.sh
cd /data
python -m pytest --import-mode=importlib -q \
  /data/Agentrix/vllm-ascend/tests/ut/patch/platform/test_mamba_retention.py \
  /data/Agentrix/vllm-ascend/tests/ut/patch/platform/test_mamba_prefill_config.py
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
`/data/Agentrix/experiments/agentx-ascend/retention-v2/validation/`。

官方报告、配置、代码哈希及比较结果在服务器
`/data/Agentrix/experiments/agentx-ascend/retention-v2/`。
完整日志保留于服务器 `/data/Agentrix/experiments/agentx-ascend/results/retention-v2/`。
重跑单组实验（会停止指定的上一轮实验服务）：

```bash
source /data/Agentrix/activate-ascend.sh
python experiments/agentx-ascend/run_retention.py \
  --previous-run /path/to/previous-run \
  --run-dir /path/to/new-run \
  --interval 8192 --prefer-reuse-boundaries

# 在服务器 /data/Agentrix 下跨第一轮代码和本轮代码比较，明确记录代码变化。
python experiments/agentx-ascend/compare_retention.py \
  experiments/agentx-ascend/retention-v2/reference8192 \
  experiments/agentx-ascend/retention-v2/priority8192 \
  --allow-code-change \
  --output /data/Agentrix/experiments/agentx-ascend/results/retention-v2/recomputed-comparison.json
```
