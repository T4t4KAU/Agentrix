# 优化现状与文档导航

本文是当前优化结论的总入口；详细配置、原始记录索引和局限在各主题主文档。
核对日期为 **2026-10-01**。下表区分已有功能、实测收益与待实现方案，
不同模型、硬件、负载和阶段的加速比不相乘，也不拼成一个总提升比例。

优化不以吞吐提升为唯一保留条件：同等服务质量下的 GPU 容量节省、可承载会话数、
CPU 缓存占用或搬运量改善也可独立成立，并同时报告成本。后续优先核对官方选择性
卸载与更小 KV 池的容量收益，见 [内存验收与下一步](kv_memory_optimization_status.md#内存优化的验收与下一步2026-09-28)。

## 官方重复实现核对

以本地 CUDA vLLM 的官方基线 `2cf0a6915c`、Ascend 插件基线 `5f6faa0` 和
官方 router 发布包 `0.1.15` 核对实际实现；发布版以外的能力单独注明，
不把博客中的未来工作或未合入提案当作可用接口。

| 项目 | 核对结果与处理 |
| --- | --- |
| Prefix/session-aware DP | 官方 [router](https://github.com/vllm-project/router) 已有 `consistent_hash`、`cache_aware` 及 DP rank 转发。本轮删除自定义 router、frontend hook 和 Ascend Python 代理，AgentX 默认改为官方 `consistent_hash`；旧策略的数据保留为历史结果 |
| APC、物理 KV 页共享、混合缓存池 | CUDA 官方基线已有；继续使用原生管理器，不归为 ForkAttention 新增的内存优化 |
| CUDA 稀疏保留与共享前缀检查点 | 基线已有 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`、共享前缀边界发现和未缓存块优先回收。直接沿用官方实现，不另写管理器；较新上游改用 cache config 字段，不能直接套到旧版本 |
| Ascend Qwen3.5 混合保留 | 与官方方向重合；当前是针对 vLLM 0.22.1 的兼容补丁，显式限制版本。新版官方 [Mamba 适配](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/patch/platform/patch_mamba_manager.py) 调用上游管理器。现有环境尚未升级验证，因此暂保留旧版可选适配；迁移匹配的官方 vLLM/Ascend 版本后应移除补丁，不与官方策略叠加 |
| 长 prefill 分块、ACLGraph、`npugraph_ex` | 均调用官方配置与执行能力，保留配置和正确性检查；不宣称新增调度器、图运行时或原创算法 |
| KV 容量准入绕行 | 没有收益证据，本轮删除实现及专用测试，恢复官方 FCFS 准入 |
| Agent Hints 分叉 checkpoint | 官方已有检查点机制；本地新增的外部边界提示入口在两 seed、四轮 H100 官方 AgentX 中未建立整体收益，已撤回。新版官方 [内部 prefill checkpoint](https://github.com/vllm-project/vllm/pull/52789) 与外部提示协议不同；官方机制继续沿用 |
| CUDA ForkAttention / NPU FIA ForkAttention | CUDA 已有官方 attention/Cascade 路径，当前路由对照改用官方 FlashAttention；自定义 CUDA 后端和 FIA 共享前缀调度仍限于显式实验。FIA 计算本身为 Ascend 原生能力，已有结果不支持把自定义后端设为整体性能默认项 |
| 工具等待卸载、CPU/磁盘 KV 层级 | 优先接官方 connector。完整工具生命周期策略尚未实现，不把基础 offload 等同于 Agent Hints 驱动的迁移 |
| 工具结果外置与应用页面管理 | 操作的是工具正文和文件页面，与 vLLM KV block 管理不是同一层；保留已测应用实现和收益边界 |

路由替换与无收益准入清理没有升级服务器整套推理栈。官方 Rust router 的五项
接口测试通过；原生 DP 十项单元检查通过，使用固定
输入 token IDs 避开 gated tokenizer，不涉及模型执行；缓存/checkpoint 的 31 项回归
及原生准入的六项调度检查通过。新 router 的硬件 AgentX
吞吐和延迟尚待测量。随后单卡 checkpoint 完成四轮官方对照并撤回增量，撤回后的
九项原生 Request/混合缓存回归通过。启动、复现与旧路由数据见 [DP 路由](dp_routing.md)。

Ascend 混合 KV/状态 CPU 卸载已修复恢复阻塞与状态边界问题；4 会话 × 3 轮
强制 CPU 恢复输出全部一致。整块输入缺少更早状态快照时仍会回退重算；
这属于隔离配置下的模型级正确性结果，默认不启用，不代表官方 AgentX 收益。
后续两组 seed、各三轮缓存压力对照中，选择性备份将恢复平均 TTFT 从
**794.16 降至 192.22 ms（−75.80%）**，终止请求 CPU 写入为零；192 个请求
跨配置输出一致。缓存池预算相同，不宣称板卡显存下降。
详见 [Ascend 验证状态](kv_memory_optimization_status.md#ascend-可先执行的验证2026-09-30)。

## 当前有收益的部分

| 优化 | 已测收益 | 证据范围与当前处理 |
| --- | --- | --- |
| Ascend Decode ACLGraph 配置 | 输出吞吐 **48.28 → 88.97 tok/s，+84.3%**；平均 TTFT **2.252 → 1.941 s** | Qwen3.5-9B、双 910B2、官方 AgentX；eager/graph 各两次交错运行、同为 36 GiB/卡。复用已有图执行，包含编译/融合路径；这是当前最明确的官方端到端改善，限于该负载和一个 seed |
| Ascend 混合缓存选择性保留与回收 | 输出吞吐 **78.91 → 90.35 tok/s，+14.49%**；平均 TTFT **4.968 → 1.821 s** | 同图模式、同容量的官方 AgentX 单次对照；组合包含稀疏 GDN 状态保留、未缓存块优先回收和边界保护。保留为可选配置，尚未建立重复稳定性或单项归因；不减少预分配 HBM |
| Ascend 会话粘性路由与长 prefill 分块 | native → sticky：**36.51 → 41.35 tok/s**；sticky 内 cap0 → cap1024：**41.35 → 43.16 tok/s**，平均 TTFT **6.47 → 4.02 s** | 官方 AgentX 各一次。该数字来自已删除的旧实验代理；当前已替换为官方 router，待重测。分块复用原生调度能力，新增非法块配置检查。两项收益分开解读 |
| Ascend `npugraph_ex` | 在相同绑核配置上输出吞吐 **88.87 → 90.35 tok/s，+1.66%**，平均 TTFT **2.005 → 1.821 s** | 官方 AgentX 单次小幅改善，属于启用已有能力；需要复测。90.35 与上一缓存对照复用同一运行，不是独立重复 |
| 工具结果外置、按需读取 | 活动 KV 采样峰值降低 **53.36% / 70.74%**；同批任务完成速率 **3.40 / 3.78 倍** | H100、Qwen3-8B、2 GiB KV 池，两个 seed 各两次 A/B；64/64 工作流、192/192 分支正确。仅为合成工具工作流，改变模型输入；板卡显存峰值未降低，不属于官方 AgentX 或内核加速 |
| 工具数据共享、阶段回收与流式文件读取 | 唯一页面载荷 **72 → 8.03 MiB**；阶段存储峰值 **32 → 4 MiB**；读取进程 RSS **176.23 → 20.32 MiB** | 三项独立 CPU/存储实验；数据或输出校验通过。页面共享的更新耗时增加约 **67.8%**，不能说所有指标都改善 |
| 选中大结果流式入库 | 64 MiB 完整分页结果的进程 RSS 峰值 **285.32 → 29.63 MiB，−89.62%**；耗时 **+7.97%** | 本轮服务器 CPU/存储对照，64 MiB 三对、8 MiB 两对及小结果控制；完整恢复一致，48 项回归通过。额外临时文件 I/O 换取主存下降，不涉及模型或设备 KV 收益 |
| Ascend 终止请求选择性 KV 备份 | 恢复平均 TTFT **794.16 → 192.22 ms，−75.80%**；P95 **813.13 → 206.56 ms**；终止请求 CPU 写入为零 | Qwen3.5-9B、单 910B2、两 seed 各三轮、192 个请求跨配置输出一致；同为 2 GiB 设备 + 2 GiB CPU。官方机制适配及请求上限应用；顺序生命周期实验，不是官方 AgentX，不证明显存下降 |
| 终止请求选择性 KV 备份 | 相对官方默认卸载，恢复平均 TTFT **460.96 → 136.90 ms**，重算 token **−96.68%**；终止请求写入 **2.317 → 0 GiB/轮** | H100、Qwen3.5-9B、同为 2 GiB GPU + 2 GiB CPU，两个 seed 各三轮；输出一致。使用官方请求字段的受控顺序负载，不是官方 AgentX；同容量下板卡显存不变，容量扫描另列 |

Ascend 图执行、缓存保留及历史路由的完整结果与运行边界见 [AgentX Ascend](agentx_ascend.md)；
应用层前两项以 [NVIDIA 内存与工具数据结果第 3.3 节](nvidia_memory_results_and_ascend_plan.md#33-h100-当前应用代码复测2026-09-27)
为准；最新流式入库见同文第 4.4 节。这些方法分别涉及上游能力启用、旧版适配和
应用层实现，不作为原创算法清单。
1 GiB 选择性备份复验也已完成，P95 恢复 TTFT 为 **164.20 ms**，板卡显存
采样峰值比 2 GiB 配置低 **960 MiB**；两组均额外预留 2 GiB CPU 缓存。
恢复连接后已核验完整矩阵：**0.75 GiB GPU + 2 GiB CPU** 选择性备份的恢复
P95 为 **167.27 ms**，4 GiB 无卸载 APC 为 **163.24 ms**，均满足 300 ms
门槛；前者板卡显存采样峰值低 **3,062 MiB**，平均恢复 TTFT 增加约 24 ms。
这是已测达标配置的比较，不是绝对最小设备池或并发容量结论。部分尾块硬件检查
仍待空闲设备。选择性备份的基线、CPU 成本与容量验证见
[H100 混合 KV 实测](kv_memory_optimization_status.md#h100-混合-kv-选择性备份实测2026-09-29)。
Ascend 已完成两张卡共 32 项底层 DMA 检查，并通过官方实现兼容适配及调度边界
修复完成模型级混合卸载；选择性备份的重复收益及整块边界回退限制见
[Ascend 验证边界](kv_memory_optimization_status.md#ascend-可先执行的验证2026-09-30)。
会话路由与 CUDA prefix-aware 的自定义实现已由官方 router 替代；历史数据没有
证明超越官方方案，详见 [路由来源与边界](dp_routing.md#与官方文章的关系)。

## 有局部收益，但不能写成整体提升

| 项目 | 已确认的结果 | 不成立的结论 |
| --- | --- | --- |
| 已删除的 CUDA prefix-aware DP | H100/Qwen3-8B 合成前缀重访中，相对 native 命中率 **49.23% → 98.46%**，旧版批次耗时 **1576.18 → 1087.41 ms** | 不是官方 AgentX；新版热点分配从 9/7 修正为 8/8，但追加复测重访耗时增加 **2.52%**、P95 TTFT 增加 **21.55%**，新版不记为总体性能提升 |
| 显式 Agent Hints 分叉 checkpoint（已撤回） | H100/Qwen3.5-9B 合成父子请求六组对照，总耗时中位数降低 **7.84%**、子请求 TTFT 中位数降低 **49.02%** | 局部收益来自减少首个子请求重算；完整官方回放未改善，新增入口和专用测试已撤回，源码及复现脚本仅存服务器归档 |
| 修正后的 checkpoint 选择器 | trace 串行片段四组配对耗时改善中位数 **2.26%**；十请求预热批次 **0.59%** | 提示预计算的局部实验；后续两 seed 完整回放没有新增正式缓存命中，未纳入生产配置 |
| FIA 版 NPU ForkAttention | 双 910B2 的长共享前缀算子形状有明显加速，例如卡 1 的 8 分支/32K：**561.32 → 131.47 µs** | 不含 CPU 规划/逐步更新；官方 AgentX 未触发共享路径，吞吐 **88.79 → 88.26 tok/s**，没有端到端收益。默认关闭 |

详情分别见 [DP 路由](dp_routing.md)、[Agent Hints 与缓存](kv_memory_optimization_status.md)
和 [NPU ForkAttention](agentx_ascend.md#npu-forkattention-在线-decode-接入)。

## 未验证、负结果与已撤回项目

| 项目 | 当前结论 |
| --- | --- |
| checkpoint 完整 H100 复测 | Qwen3.5-9B、两个 seed、四轮官方 AgentX 均有效；开启后输出吞吐 **−3.95% / −0.50%**，共同请求平均完成延迟 **+1.20% / +0.11%**，正式缓存命中增量 **−528 / 0 token**。未建立整体收益，撤回新增提示入口；第一组请求数 68/67，吞吐变化不能直接解释为满载性能变化 |
| 旧 checkpoint 自动提示策略 | 官方 AgentX 开关对照的 67 个正式请求缓存命中量相同，平均 TTFT 增加 **24.34%**；不纳入正式自动策略，不能据此否定所有 Agent Hints |
| Agent Hints 父/根会话路由 API | 功能测试通过，但未建立性能收益；新增 API 和继承逻辑已撤回 |
| 工具等待期间卸载与恢复前预取 | **尚未接通、没有收益数据**。基础 connector 存在；应用侧 `ToolKVTrimmer`/TTL 代码仍在，但当前 vLLM 缺少其 trim API，不能把旧设计当作已实现的工具等待卸载 |
| CUDA ForkAttention 相对 FlashInfer Cascade | RTX 5070 算子矩阵中 Cascade 在 31/32 个形状更快；不能声称当前 ForkAttention 优于 Cascade，也不能外推 H100 |
| CPU 绑核修复 | 已修复编号映射和验证线程亲和性；官方单次吞吐变化 **−0.46%**，未建立独立性能收益 |
| NPU 直接计算内核 | 慢于 FIA 版，实现和测试已删除；原始数据、源码快照仅在服务器历史归档 |
| KV 容量绕行 | 未建立收益，已删除；采用官方 FCFS |
| 历史 CUDA session retention / adaptive prefill | 实验已撤回，不计为保留的有效优化 |
| 历史 residency/placement、LMCache `FORK_AWARE` | 当前固定子模块缺少相关实现或接入，不计为当前可用优化 |
| 累计工具上下文预算 | 补查历史四轮较大窗口结果：基线每轮 2/2 工作流正确，候选每轮 1/2；虽减少输入和活动 KV，质量门槛未通过，不启用为默认优化 |

[算子对照](fork_attention/forkattention_operator_profile.md)保留 Cascade 的测量范围；
[KV 状态](kv_memory_optimization_status.md)和[工具 trim 设计](tool_kv_trimmer.md)说明缺失的接入。
历史方案逐项取舍见 [历史设计有效性](kv_memory_optimization_status.md#哪些历史设计有效哪些还没有证据2026-09-29)；
累计预算遗漏结果的更正见 [质量门槛与验证进度](nvidia_memory_results_and_ascend_plan.md#5-测试口径与验证进度)。

## 本轮数据核对与归档

- 本轮在 Ascend 服务器核对 **18 份运行摘要与官方原始报告**，复算图执行、混合缓存、
  `npugraph_ex` 和 FIA 在线对照；路由总表另与其四份官方报告核对。
- 服务器 `${RESULTS_DIR}/documentation-audit-20260928/evidence-index.json` 建立
  **94 个来源文件**的路径、大小和 SHA-256 索引，包含官方报告、manifest、算子结果及既有汇总。
  原始目录未搬动、未覆盖、未删除；本地没有新增原始数据副本。
- 上述文档审计时 H100 连接超时，审计索引明确保留这一限制。随后连接恢复，完成
  两 seed、四轮 checkpoint 官方回放，271 个正式请求均无错误或取消；详细结果见
  [H100 完整回放复测](kv_memory_optimization_status.md#h100-完整回放复测与撤回2026-09-28)。
- 新测试的完整数据、失败尝试及源码只在服务器
  `${RESULTS_DIR}/agentx-checkpoint-h100-20260928/`；未追加本地原始报告。测试服务已停止，
  历史部署记录不代表服务当前仍在运行。

## 文档导航

部署路径统一使用占位符：`${REPO_ROOT}` 为仓库根目录，`${MODEL_DIR}` 为模型目录，
`${RESULTS_DIR}` 为服务器结果目录，`${EXPERIMENT_DIR}` 为服务器实验归档目录。
`${DEPS_DIR}`、`${CACHE_DIR}` 分别表示依赖和安装缓存目录；
`${CUDA_VENV_DIR}`、`${CUDA_INSTALL_DIR}` 表示独立 CUDA 环境和安装目录。
复现前在自己的环境中设置这些变量；实际值和连接参数仅保存在私有配置中。
命令中的 localhost 和示例端口用于说明调用方式，不代表实际部署入口。

| 主题 | 入口 |
| --- | --- |
| 官方 AgentX、Ascend 路由、调度、混合缓存与图执行优化 | [AgentX Ascend](agentx_ascend.md) |
| NVIDIA 工具数据与活动 KV 的已测结果 | [内存与工具数据结果](nvidia_memory_results_and_ascend_plan.md) |
| KV 生命周期、淘汰、offload/restore 与当前限制 | [KV 内存管理](kv_memory_optimization_status.md) |
| 官方 DP 路由与历史亲和性实验 | [DP 路由](dp_routing.md) |
| 服务器环境、增量构建与清理边界 | [AutoDL 运维](autodl_build_and_benchmark.md) |
| 系统分层和历史应用路径 | [系统概览](agentrix_system_overview.md) |
| 算子、Cascade 和系统级 profiling 方法 | [ForkAttention profiling](fork_attention/forkattention_operator_profile.md) |

## 专题指南

- [ForkAttention](fork_attention/README.md)：算子 profiling、SGLang/llama.cpp 后端适配、模型兼容性与设计。
- [Coding Agent](coding_agent/README.md)：数据集、实时演示与任务质量验证。

## 其他专项指南

- 应用：[精确 prompt compaction](application_prompt_compaction.md)、
  [工具等待期间 KV trim 与 TTL 历史设计](tool_kv_trimmer.md)（当前引擎接入缺失）。
- [SGLang LMCache](sglang_lmcache_usage.md)。

这些专项指南描述各自路径，不意味着都已在当前服务器、模型或子模块版本上重新验证。

## 维护规则

- 同一主题只维护一份主文档，不再为每轮调试新增报告。
- 本地只保留实验代码、复现方法和主文档中的关键实验数据、验证结论与局限。
- 完整日志、原始报告、逐请求数据和验证输出只保存在实验服务器；文档用占位符说明记录位置，不再按轮次下载到本地。
- 文档不记录实际地址、账号、个人信息、主机名、设备唯一标识、连接参数、凭据或部署绝对路径。
- 删除报告或图表时同步清理文档入口和失效链接。

## 提交前验证（2026-10-01）

主仓库 benchmark 测试 **196 项通过**；官方 router 独立环境接口测试 **5 项通过**；
vLLM 卸载调度 **126 项**、配置及相关调度 **48 项**通过。
本机 engine-client 测试模块因缺少 CUDA 平台整体跳过，不计入通过数。
Ascend 硬件验证沿用上一节已完成的模型往返及 192 请求对照，本轮没有重跑模型实验。
更改文件通过部署信息检查；原始实验产物保留在服务器，个人 IDE 配置不纳入提交。
