# DP 路由与 GPU KV 亲和性

## 策略与边界

Agentrix 扩展 vLLM 的 internal DP 路由，不增加第二套调度器。
`VLLM_AGENTRIX_DP_ROUTING_POLICY` 选择互斥策略：

| 策略 | 行为 |
| --- | --- |
| `native` | 按原有队列负载选择 replica |
| `prefix_aware` | 在负载和工作量边界内优先选择已有长前缀的 replica |
| `session_aware` | 首轮保留 native 选择；后续轮优先会话归属，检查前缀与过载条件 |

旧开关 `VLLM_FORK_ATTN_DP_PREFIX_ROUTING=1` 是 prefix-aware 的兼容入口。
Session-aware 是 prefix-aware 的扩展选择，不是再叠加一次 DP。

请求可在 `vllm_xargs` 中提供：

```json
{
  "agentrix_session_id": "conversation-42",
  "agentrix_turn": 2,
  "agentrix_history_tokens": 3072
}
```

后续轮有会话映射时优先该 rank；没有映射则参考最长已知前缀。
历史覆盖不足时回到 baseline；过载时从负载合格的 rank 中按估算工作量再平衡。
会话映射受 TTL 和容量约束，缓存 reset 时清理。
未显式指定 turn 的 chat 请求可从已有 assistant 消息推断。

工作量估计包括未命中的 prompt 和加权最大输出长度，不是实际剩余 GPU 时间。
当前评分没有直接纳入可分配 KV 容量；请求数均衡也不保证瞬时内存压力均衡。

## 可选 GPU 驻留反馈

`VLLM_AGENTRIX_DP_KV_EVENTS=1` 使用 GPU store/remove/reset 事件校正逻辑提示，默认关闭。

- Scheduler 在合并 connector 事件前，将 GPU 事件随已有 engine 输出送到 frontend。
- 有界索引只承认连续祖先链；缺失父节点或容量淘汰会低估命中，不应虚构完整前缀。
- 后续轮可迁往有更长已验证 GPU 前缀的 rank，仍受原有负载和工作量限制。
- 事件可能在途，因此只是路由提示。目标 engine 仍自行校验 hash、恢复或重算。
- Router 不 pin GPU block，不同步调用 scheduler，也不自行跨 GPU 复制 KV。

已覆盖一个 API frontend、internal DP、普通 token prompt、单 full-attention 或 MLA
cache group。LoRA、salt、多模态身份、embedding、hybrid group 和其他不支持布局
沿用逻辑路由路径。Dense DP 的事件门控使用内部完成报告契约，不能只检查
`data_parallel_size > 1`，因为 replica 内该值可能已被改写。

实现入口：

- [前缀与会话选择](../vllm/vllm/v1/engine/prefix_router.py)
- [GPU 事件索引](../vllm/vllm/v1/engine/kv_routing.py)
- [Frontend 接入](../vllm/vllm/v1/engine/core_client.py)

## 实验结论

当前完整系统结论见 [TraceLab](tracelab_timeline_replay.md)：92 会话、132 请求的
4x 时间线回放中，组合配置慢 19.05%，不能用旧的强亲和 microbenchmark 宣称普遍提速。

较早的 matched GPU-event 对照为 Qwen3-VL-8B、2 x RTX 5090、384 blocks/replica、
12 会话、每请求输出 1 token，三轮中位数：

| 指标 | 逻辑提示 | GPU 事件提示 |
| --- | ---: | ---: |
| 后续轮 P50 TTFT | 394.48 ms | 396.27 ms |
| 后续轮 GPU cached-token rate | 53.00% | 53.00% |
| Router-only P50 | 35.30 us | 72.54 us |

它没有建立显著加速，router-only 不包含事件生产、序列化和消费的全部成本。
旧 prefix/session 对照统一列在 [历史索引](historical_experiments.md#dp)。
`dp_placement_physical` 的早期实验没有开启 dense-DP producer，不能作为事件路径证据；
有效记录是 `dp_placement_physical_verified`。

## 验证入口

仅在服务器上运行，模型与输出路径显式配置，结果目录每轮独立：

- `benchmark/scripts/run_agent_session_dp_profile.sh`：prefix/session 策略对照。
- `benchmark/scripts/profile_dp_kv_events.py`：router-only 成本。
- `benchmark/scripts/profile_tracelab.py`：当前原始 vLLM/Agentrix 对照。

CPU/Mooncake 集成曾完成双 replica 写入，但该 DP trace 的 external retrieval 为零，
不能据此声称跨 replica restore 或 RDMA 加速。后来的恢复失败与当前暂停状态见
[KV 内存管理](kv_memory_optimization_status.md)。
