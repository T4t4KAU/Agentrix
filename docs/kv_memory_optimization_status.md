# KV 内存管理：设计、状态与验证边界

可选 KV 管理功能包含驻留索引、压力准入、淘汰、backup/restore 和 DP 反馈，
需要分别验证及按需启用。

## 结构与安全约束

`BlockPool` 仅依赖 observer 协议；`KVCacheManager` 挂载可选 residency index。
不向通用 KV block 增添策略字段。关闭时保留空 observer 检查。

[Residency](../vllm/vllm/v1/core/kv_residency.py) 跟踪
FREE、UNHASHED、ACTIVE、WARM、COOLING、COLD 及共享变体：

- 请求 adoption 与内部 pin/unpin 分离；offload 临时引用不增加共享度。
- 并发 request fanout 达 2，或复用次数达到阈值，进入该 generation 的共享分类。
- block 复用或 key 失效更新 generation，共享统计重置。
- Backup 使用 `(block_id, generation, tier, operation_id)`，拒绝迟到、重复或错误 ACK。
- 老化每步限制 transition 数，默认 256；O(1) 状态计数与快照，完整一致性检查仅用于测试。
- Pin 恢复有效的原队列位置；锚点已失效时刷新为 warm 尾项，不把旧时间戳插到新项后面。

## GPU 淘汰与准入

[Placement](../vllm/vllm/v1/core/kv_placement.py) 与 observer 分离。
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
