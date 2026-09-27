# KV 内存管理：设计、状态与验证边界

## 当前代码核查（2026-09-27）

当前 CUDA `vllm/` 固定在 `437cf6727e79`，Ascend 插件固定在 `7ba43c552045`。
内存容量、缓存复用和算子读取流量是不同指标，不能合并计算 ForkAttention 收益。

| 项目 | 当前实现与证据 |
| --- | --- |
| 共享前缀物理 KV 页 | vLLM 原生 APC 已通过哈希查找和引用计数共享完整页，FlashAttention 也能使用；不是 ForkAttention 独有的容量优化 |
| Ascend Qwen3.5 混合缓存 | 已实现稀疏 GDN 状态保留、未缓存块优先回收、复用边界保护及实际命中后的晋升；服务器源码、安装包和原测量的三个核心文件哈希一致 |
| Ascend 缓存收益 | 相同 36 GiB/卡，官方 AgentX 单次对照输出吞吐 78.91 → 90.35 tok/s，平均 TTFT 4.968 → 1.821 s；这是策略组合的观测，尚无稳定性或单项归因结论 |
| CUDA 稀疏状态保留 | 当前 `single_type_kv_cache_manager.py` 已有 `retention_interval` 和共享边界保留；`block_pool.py` 已有未缓存块优先回收，不能把向旧 Ascend 组合移植的部分算作相对当前 CUDA 的新增算法 |
| 驻留索引 / placement | 本文下方描述的 `kv_residency.py`、`kv_placement.py` 及对应环境开关不在当前 CUDA 子模块中；相关 profile 脚本仍保留旧导入 |
| Native fanout offload | 当前 vLLM 没有消费 `fanout_offload`、`fanout_allow_hot_prefix_backup` 的代码；旧脚本传入这些字段不代表策略生效 |
| LMCache `FORK_AWARE` | 当前 policy registry 仅有 LRU、LFU、FIFO、MRU；旧配方的 `FORK_AWARE` 会被拒绝 |
| Proactive backup | LMCache 侧协调代码仍在，但其依赖的 vLLM 开关、observer 和挂载调用缺失；当前组合没有接通 |
| 工具数据与上下文管理 | 当前应用层仍有外置工具结果、按需读取、不可变页面共享、阶段回收和文件流式扫描；独立于 CUDA/NPU attention 内核 |

历史提交说明上述旧策略确实实现过：vLLM 的 `agentrix` 分支保留驻留/placement
（`0d0ce78224`）及恢复成本排序（`f1ecb2e8a3`）；LMCache 的 `fork-attn` 分支
保留 `FORK_AWARE`（`b84945ca`、`768bd187`）。这些提交不属于当前固定版本的
祖先，不能把“历史分支已有实现”写成“当前运行时已启用”。

另一轮 CUDA GPU session retention、CPU session LRU 和 adaptive prefill
实验曾在 `a7b928cd76` 加入，随后在 `fa702ea44c` 明确撤回。提交记录说明筛选
没有建立一致收益，包括 GPU retention 的平均 TTFT 变差 51.3%；本轮未重跑这些实验。
当前仍保留 `8040e1b792` 的可选 KV 容量准入绕行：FCFS 队首放不下时有界查找
后面的可容纳请求，默认关闭，并限制于 DP1 等受支持配置。它改变准入顺序，不压缩 KV。

应用层源码在 `application/src/agentrix_application/prompt_compactor.py` 和
`benchmark/src/coding_agent_tools.py`。2026-09-27 已在原 H100 服务器 GPU 1
用当前应用代码复测：Qwen3-8B BF16、FlashAttention 3、eager、原生 APC、固定
2 GiB KV 池，两个 seed 各进行两次 inline/paged 对照。采样活动 KV 峰值分别
降低 **53.36% / 70.74%**，同批任务完成速率为 **3.40 / 3.78 倍**；合计
64/64 工作流、192/192 分支正确，零请求错误。两组板卡显存峰值均为 18,628 MiB。
主机实验也复现了页面共享、阶段回收和文件流式读取的存储/RSS 收益，页面共享的
写入耗时同时增加约 67.8%。相关 45 项测试通过，全部原始产物位于 H100 服务器
`${RESULTS_DIR}/memory-audit-h100-20260927/`。
首轮 53.2% / 3.59 倍的原始数据也已核验。配置、采样局限及复现入口见
[NVIDIA 内存实验第 3.3 节](nvidia_memory_results_and_ascend_plan.md#33-h100-当前应用代码复测2026-09-27)。
这些是合成工具工作流的应用层结果，不是官方 AgentX、CUDA 内核或历史
residency/placement/offload 策略的性能结论。

Ascend 优化改变固定池内保留的内容，不降低已分配 HBM，也没有取消每次实际
prefill 所需的 GDN 计算和状态更新。原生对照使用相同补丁包的 `interval=null`，
关闭稀疏保留及优先回收；不是另装一份未修改的上游环境。两组使用相同图模式、
绑核、路由、容量及官方输入配置，未启用后续的 ForkAttention 算子。

复算核对了官方报告、manifest 和诊断快照。优化组最后一次累计采样记录了
951 次优先周期块回收、3 次命中晋升、8368 次未缓存块回收，证明策略被实际触发；
这些是含预热及重试的过程计数，不是独立 KV 页数、节省字节数或已执行重算 token。
两组两卡的 `num_preemptions` 均为 0，不能据此声称减少了抢占或解决了 OOM。
`kv_cache_usage_perc` 按未进入 free queue 的块计算；可淘汰但仍有缓存内容的块在
free queue 中，因此该指标不能直接当作有效缓存驻留率或 HBM 分配率。
完整配置和局限见 [AgentX on Ascend](agentx_ascend.md)。核查产物仅保存在服务器
`${RESULTS_DIR}/memory-audit-20260927/`。

本次重跑 50 项缓存/配置测试通过。修复了测试提前导入 `Scheduler` 而持有插件
替换前类引用的问题，测试现在解析插件加载后的运行时类；缓存与调度运行时代码未改动。
这是既有官方结果的复算和功能回归，没有重新运行一轮官方性能 benchmark。

旧报告的 `branches * prefix + sum(suffix)` 对比 `prefix + sum(suffix)` 只是
“每分支复制”与“共享存储”的逻辑模型，不是启用 APC 的 FlashAttention 与
ForkAttention 的实际物理页对照，也不是测得的 HBM 读取字节。后续容量结论应使用
同一 APC 配置下的物理页数、有效混合命中长度、淘汰/重算及工作区峰值。

## 历史设计（当前 CUDA 子模块未接通）

以下保留驻留索引、压力准入、淘汰、backup/restore 和 DP 反馈的历史设计。
其中缺失模块的路径和开关不构成当前可用功能，不能直接按下面的配置复现实验。

## 结构与安全约束

`BlockPool` 仅依赖 observer 协议；`KVCacheManager` 挂载可选 residency index。
不向通用 KV block 增添策略字段。关闭时保留空 observer 检查。

历史 `vllm/vllm/v1/core/kv_residency.py` 跟踪
FREE、UNHASHED、ACTIVE、WARM、COOLING、COLD 及共享变体：

- 请求 adoption 与内部 pin/unpin 分离；offload 临时引用不增加共享度。
- 并发 request fanout 达 2，或复用次数达到阈值，进入该 generation 的共享分类。
- block 复用或 key 失效更新 generation，共享统计重置。
- Backup 使用 `(block_id, generation, tier, operation_id)`，拒绝迟到、重复或错误 ACK。
- 老化每步限制 transition 数，默认 256；O(1) 状态计数与快照，完整一致性检查仅用于测试。
- Pin 恢复有效的原队列位置；锚点已失效时刷新为 warm 尾项，不把旧时间戳插到新项后面。

## GPU 淘汰与准入

历史 `vllm/vllm/v1/core/kv_placement.py` 与 observer 分离。
Shadow 只观察；active 在需要回收缓存块时调整 free queue：

1. 排除共享块、正在 backup 的块和本次请求将采用的命中块。
2. 优先有下层副本的块，再按未复用/已复用和 cold/cooling/warm 排序。
3. Active 允许丢弃非共享、未备份块并接受未来重算；shadow 将这些候选标为待备份。
4. 应用前重新验证 generation，原子调整完整候选集，不回退淘汰共享前缀。

扫描预算跨同一 scheduler step 共享。Active 至少允许扫描本次分配所需候选数，
因此不是严格固定 64 项，而是随本次 allocation 有界增长，不扫描整个 cache。

**仍需排查的交互：** 候选不足或失效会让 allocation 返回失败，沿现有
admission/preemption 路径处理。因此“保护共享缓存”可能挤压运行请求；
应在压力场景下验证该交互。

成本目前是 tier、reuse、age 的固定整数排序，不是校准过的重算/传输延迟模型。
`KVCacheManager.get_placement_stats()` 提供诊断计数；不能把 cached-token 增长
直接解释成跨请求收益，抢占恢复也可能命中自身已计算的缓存。

## Proactive backup

协调逻辑属于 LMCache，vLLM 仅保留 observer 和薄 connector 接口：

- 请求完成后按完整 chunk 登记候选；按水位以上占用量接纳，扣除已 retained 的量。
- Scan、单批 transfer 和总 retained blocks 分别有上限，压力回落时有界释放。
- 再验共享属性、generation 和 CPU 副本后 pin GPU block，再开始 D2H。
- 使用 delayed-free 保持请求和物理 block 存活至 ACK；失败释放 pin，不紧循环重试。
- 同步 D2H 为默认；async 在 connector store stream 上复制，用 CUDA event 保证
  完成后才发布 CPU cache、ACK 和释放 GPU ownership。
- MLA 的 `save_only_first_rank` passive ranks 返回 no-op success，实际存储由 leader 完成。
- CPU eviction 以 chunk hash 映射回 block/generation，调用 `drop_backup()` 清除驻留位。
- Proactive 模式关闭普通请求驱动的重复 save，让 chunk 只走一条存储路径。

## Restore 与 CPU/Mooncake

Restore 反馈携带目标 block ID、generation、chunk hash。GPU load 成功且各实际存储
worker 确认 CPU key 仍存在后，才标记 CPU residency；eviction 反馈随后应用。
Partial/malformed load 应进入 `kv_load_failure_policy: recompute`。

`lmcache.max_tokens_per_load` 在 lookup/prefetch 前限制外部前缀，按 chunk 对齐，
其余上下文重算。沿用 LMCache async loading，不再新建一套 transfer executor。

| 配置 | 含义 |
| --- | --- |
| `max_local_cpu_size` | CPU 热缓存/传输 allocator 容量，GiB |
| `global_segment_size` | 本客户端贡献的 Mooncake DRAM |
| `local_buffer_size` | Native scratch buffer，额外占用 |
| `remote_max_inflight_bytes` | 远端写保留的 CPU buffer ownership 上限，不是另一内存池 |

复用完成的 D2H buffer 发布 CPU/Mooncake，使用 `save_chunk_meta: false` 和
`remote_serde: naive` 走现有注册内存路径。CPU 与 Mooncake 仍有独立容量，
“统一层”不意味着已实现统一 allocator 或全局容量调度。

`BoundedRemoteWriter` 整批接纳或拒绝，native transfer 完成并清理后才归还预算。
Python timeout 无法终止 C++ 线程；取消后必须等 native call 结束才能释放 buffer。
永久阻塞仍可能耗尽有界预算、拖延 shutdown，不能声称 timeout 已解决全部活性问题。

远端存在性在读取时校验；一次成功 put 不产生永久有效的 REMOTE 位。
单机 TCP/RDMA smoke 仅证明初始化和兼容性，不证明跨机带宽收益。
配置样例见 [lmcache_mooncake_tiered.yaml](../benchmark/configs/lmcache_mooncake_tiered.yaml)，
当前不要把它作为默认生产配置启用。

## 配置入口

默认关闭的功能按需选择，不要整段同时开启：

| 开关 | 默认/用途 |
| --- | --- |
| `VLLM_AGENTRIX_KV_RESIDENCY_SHADOW` | 0；仅观察 |
| `VLLM_AGENTRIX_KV_AGING_BUDGET` | 256 |
| `VLLM_AGENTRIX_KV_WARM_SECONDS` / `VLLM_AGENTRIX_KV_COOLING_SECONDS` | 10/100 s |
| `VLLM_AGENTRIX_KV_SHARED_REUSE_THRESHOLD` | 2 |
| `VLLM_AGENTRIX_KV_PLACEMENT_SHADOW` / `VLLM_AGENTRIX_KV_PLACEMENT_ACTIVE` | 0/0；两种模式 |
| `VLLM_AGENTRIX_KV_PLACEMENT_SCAN_BUDGET` | 64；active 的 allocation 下限见上文 |
| `VLLM_AGENTRIX_KV_PROACTIVE_BACKUP` / `VLLM_AGENTRIX_KV_PROACTIVE_ASYNC` | 0/0；独立启用 |
| `VLLM_AGENTRIX_KV_BACKUP_HIGH_WATERMARK` | 0.8 |
| `VLLM_AGENTRIX_KV_BACKUP_SCAN_BUDGET` | 32 |
| `VLLM_AGENTRIX_KV_BACKUP_BATCH_BLOCKS` | 64 |
| `VLLM_AGENTRIX_KV_BACKUP_MAX_INFLIGHT_BLOCKS` | 256 |

Placement/proactive 自动启用所需 index。Proactive 要求 non-layerwise local CPU storage，
不与 CacheBlend 组合。DP 配置见 [路由指南](dp_routing.md)。

## 验证入口

`benchmark/scripts` 中提供 `profile_kv_residency_shadow.py`、
`profile_kv_placement_shadow.py`、`profile_proactive_backup.py`、
`profile_kv_restore.py` 和 `run_lmcache_mooncake_smoke.sh`。

组合测试应覆盖压力准入、缓存淘汰、恢复失败、抢占和请求完成后的资源释放。
