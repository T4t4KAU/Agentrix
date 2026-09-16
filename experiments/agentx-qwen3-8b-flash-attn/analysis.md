# AgentX attention 后端对比诊断（2026-09-16）

## 核心结论

此前 FORK_ATTN 与 FLASH_ATTN 对比不能作为 ForkAttention CUDA 内核收益评估。
服务器上用原启动参数重新执行 `EngineArgs.create_engine_config()`，解析结果为
`async_scheduling=True`、`fork_min_queries=8`、`fork_min_shared_tokens=16384`，executor 为 `uni`。
该验证只解析配置，没有启动第二个模型服务。

本分支代码在异步调度开启时禁止构建 Fork 计划，因此原配置走 FlashAttention 回退。
这是从原参数重建配置和代码控制流得到的结论；历史运行没有内核计数器或 GPU trace，
不能把它表述为已经测得的内核调用比例。

## 代码证据

- `vllm/vllm/config/vllm.py:1185`：兼容配置默认开启异步调度。
- `vllm/vllm/v1/attention/backends/fork_attn.py:1378` 附近：异步调度下 CUDA graph Fork capture limits 为零。
- 同文件 `prepare_cudagraph_plan`（约 1517 行）：异步调度直接返回 None。
- 同文件 `_can_build_fork`（约 1904 行）：异步调度直接返回 False。
- `_build_fork_kwargs` 返回空参数，`_can_run_fork` 检查 fork_enabled，最终 `forward` 调用父类 FlashAttention。
- `vllm/vllm/config/attention.py:27`：默认 16384 token、8 个 query 的门槛。
- Fork 计划不是检查请求各自长度，而是检查至少 8 个 query 的前 16K token 对应同一组物理 KV 页。

即使关闭异步调度，也必须满足共享前缀及同时运行请求数等条件。
服务端采样的运行请求数峰值，Fork 三档分别为 5、4、7；采样无法排除短暂峰值，
但说明本负载不容易满足默认门槛。AgentX 并发参数也不等于每个 decode batch 的 query 数。
跨轮次前缀缓存命中率高，不代表多个同时解码的请求共享这些前缀。

## 高并发容量证据

两个后端日志均记录 GPU KV cache 为 **399856 token**，对 131072 token 请求的理论并发为 **3.05**。
这不意味着所有请求只能并发 3 个：实际长度、共享页和生命周期会改变占用。

下表来自 `server_metrics_export.json`。虽然文件标记 metrics_phase=profiling，
服务端直方图 count 包含预热（例如 Fork c8 为 416，而正式请求为 326）。
因此这些汇总只能用于整体诊断，不能直接作为正式测量窗口内 TTFT 的精确分解。

| 后端 / 并发 | KV 使用率 P90 / 峰值 | 平均排队时间 s | 平均 prefill 时间 s | 抢占计数 |
|---|---|---:|---:|---:|
| Fork / 1 | 28.08% / 56.25% | 0.000035 | 0.453 | 0 |
| Fork / 4 | 56.72% / 72.56% | 0.153 | 0.804 | 0 |
| Fork / 8 | 94.17% / 100.00% | 16.067 | 4.702 | 1 |
| Flash / 1 | 28.94% / 57.97% | 0.000035 | 0.452 | 0 |
| Flash / 4 | 70.42% / 90.59% | 0.158 | 0.839 | 0 |
| Flash / 8 | 93.23% / 99.98% | 16.228 | 4.629 | 0 |

并发 8 的 waiting_by_reason 为 capacity；当前配置 scheduler_reserve_full_isl=True，
调度分配会检查完整输入所需容量（scheduler.py 的 full_sequence_must_fit）。
正式测量的缓存命中率从并发 4 的约 94% 降至并发 8 的约 66%，输出吞吐却基本不再增长。
这些证据支持 KV 容量压力、容量等待与更多 prefill 工作是优先排查方向。
不能把缓存淘汰、不同轨迹构成、排队和计算耗时各自的因果贡献完全分开；
也不能归因于大量抢占，因为 Flash c8 的抢占计数为零。

## 后续方向：调度与缓存（停止算子优化）

依据：[vLLM x AgentX 原文](https://vllm.ai/blog/2026-09-08-vllm-agentx)。
文章强调长 prefill 对短续轮的阻塞、分层 KV 缓存，以及会话局部性。
以下是针对本项目单 H100、Qwen3-8B 的代码检查和实验建议，不沿用文中其他模型的收益数字。

### 0. 先建立官方已有优化的基线

固定 FLASH_ATTN、异步调度、BF16 KV、同一模型和 128K 轨迹筛选。
当前 long_prefill_token_threshold=0；scheduler.py 已经在 running 和 waiting 两条路径实现该限制。
优先比较 0、512、2048，每次只改变此参数，重点并发 8，再复核并发 4。
短跑只用于排错和筛选，入选配置恢复一小时测量，至少重复三轮。
这是启用已有功能，不算自研优化；也不能保证能消除容量不足造成的阻塞。

CPU KV offload 也已存在：config/cache.py 的 kv_offloading_size / kv_offloading_backend，
以及 v1/kv_offload/cpu/。后续可先比较关闭与 32 GiB CPU 池，确认剩余主存后再扩大。
它保存被挤出 GPU 的可复用前缀，减少重算；不会扩大同时处于 GPU 上的活跃 KV 容量。
PCIe 搬运可能抵消收益，必须同时测 H2D/D2H 字节数、等待时间和命中率。

### 1. 首选自研：容量感知的有限队列绕行

代码位置：vllm/vllm/v1/core/sched/scheduler.py，等待队列分配逻辑约 1033–1054 行。
allocate_slots 返回 None 后直接 break。因此即使队列后方某个暖请求能够放入，
当前调度也不会继续查找。这是代码层面的机制确认，不等于已测得其性能影响。

建议设计：

- 仅在容量分配失败时，查看后续有限数量请求（先取 8）；不全队列排序。
- 依据实时前缀命中、实际新增 KV 页和剩余 token 预算选择可放入的请求。
- 用只读的容量估计筛选，再对选中请求执行一次有副作用的分配；不能反复试分配造成缓存状态变化。
- 保留长请求的等待年龄与跳过次数；先用最多绕行 4 次的上限实验，防止无限饥饿。
- 第一版限定本次文本、无外部 KV connector 的 FCFS 路径；显式 priority 策略和异步 KV 加载需另行设计。
- 观测 blocked-head 次数、后方可容纳请求比例、暖请求等待、冷请求最长等待、抢占与吞吐。

它与 long-prefill-token-threshold 互补：一个处理“进不来”，另一个处理“进来后占满计算预算”。
优先在现有 scheduler.py/kv_cache_manager.py 及附近测试中实现，避免新增并行调度框架。

### 2. 第二方向：会话感知的软缓存保留

代码位置：vllm/vllm/v1/core/block_pool.py 的 free_blocks/get_new_blocks，
以及已有 CPU offload policy。当前 GPU free queue 主要按 LRU 顺序复用缓存页。

用已观察到的会话重访、父子共享次数和历史轮间间隔估计近期复用价值，
给高价值前缀有限的淘汰优先级保护。必须按可复用的前缀链保留，
不能只保留长前缀尾部而淘汰其前置块。保护应有容量上限和过期时间，
仅影响空闲可淘汰页，不能永久 pin 或挤占活跃请求。
先用在线可见信息；不能读取 trace 未来请求/工具时长作为运行时先验。

关键权衡：命中率升高可能同时降低可接纳并发，因此同时看每 GPU 输出吞吐、
排队、缓存命中和尾延迟。若使用 CPU 池，再对照纯 LRU，检验是否值得引入复杂策略。

### 3. 多卡阶段再研究路由

已有 v1/engine/prefix_router.py 支持 prefix_aware、session_aware、session_sticky。
单卡测试无法体现 DP 路由收益。以后用两卡独立副本建立 sticky 基线，
再研究“预计排队节省时间 > KV 搬运/重算成本”时才迁移。
只均衡队列长度不足以保证收益；历史 prefix hint 也不等于实际 GPU 驻留。

### 验证与实施顺序

1. FLASH_ATTN 原配置与分块阈值基线；修正指标采集，在正式窗口边界取 counter/histogram 差值。
2. 统计容量队首阻塞时可绕行请求比例，再决定是否实现有限绕行。
3. 在最佳已有基线上单独比较有限绕行、CPU offload、会话保留，逐项消融。
4. 统一保留原 82/393 条完整轨迹、种子、cache bust、预热和每档一小时；
   报告请求构成差异。吞吐要同时列出输出 token/s 和含缓存输入的总 token/s。
5. 初步验收目标：并发 8 P90 TTFT 降低至少 20%，输出吞吐下降不超过 3%，
   同时无错误、无持续饥饿，冷请求尾延迟和抢占没有明显恶化。以上为目标，不是预测收益。

此次研究未启动新性能实验、未修改调度或缓存实现。历史运行版本仍为原 manifest 所记录的版本；
后续构建提交不会追溯改变已完成实验的版本记录。
