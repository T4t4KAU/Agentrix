# DP 路由与 GPU KV 亲和性

## 策略与边界

Agentrix 扩展 vLLM 的 internal DP 路由，不增加第二套调度器。
`VLLM_AGENTRIX_DP_ROUTING_POLICY` 选择互斥策略：

| 策略 | 行为 |
| --- | --- |
| `native` | 按原有队列负载选择 replica |
| `prefix_aware` | 在负载和工作量边界内优先选择已有长前缀的 replica |
| `session_aware` | 首轮保留 native 选择；后续轮优先会话归属，检查前缀与过载条件 |
| `session_sticky` | 优先维持显式会话归属，不应用普通亲和性的过载改派规则 |

当前版本使用 `VLLM_AGENTRIX_DP_ROUTING_POLICY`，默认 `native`；旧开关
`VLLM_FORK_ATTN_DP_PREFIX_ROUTING` 已不在当前代码中。
亲和性策略要求一个 API frontend、固定 DP ranks 和开启 APC，不支持 elastic EP。
Session-aware 是 prefix-aware 的扩展选择，不是再叠加一次 DP。
路由与 attention backend 独立，FlashAttention 也能使用。

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
`agentrix_turn` 需要调用方显式提供，当前入口不会从 chat 消息自动推断。

### 两卡负载与执行进度

前缀深度相同时，先比较估算剩余工作、原生负载，再按轮转顺序打破平局。
历史访问次数不代表额外可复用 token，不能仅因一张卡访问过更多次就持续偏向它。
冷请求保持原生选择；只有命中达到最小深度的候选才应用亲和性。

工作量初值为未命中的 prompt 长度加 `16 × max_tokens`。
首个实际生成 token 到达后扣除已完成的 prefill，随后按输出 token 数扣减剩余 decode
预算。收到抢占通知时，恢复当前上下文的保守重算成本；仅有重新调度事件不会扣除它，
需要后续模型输出确认恢复。取消和结束只释放剩余预算，缓存清空后仍更新执行进度。

当前 frontend 使用的负载容差为 4，工作量容差为 8,192 token 单位。
原生负载已有精确的本 frontend 在途请求数下界；有等待队列时，KV 使用率会加重
排队惩罚。亲和性仍受这些限制，`session_sticky` 的显式会话绑定除外。

工作量是剩余输出预算的启发式估计，不是 GPU 时间预测；首 token 前不能精确跟踪
chunked prefill 的进度，也不能预知提前停止。评分没有逐请求的可分配 KV 容量，
请求数均衡不保证瞬时内存压力均衡。

## 缓存提示的边界

当前实现根据已经执行的请求维护有界逻辑前缀提示，默认 TTL 为 300 秒，最多保留
1,024 个完成请求。生成的最后一个 token 尚未作为输入计算 KV，不提前计入命中。
Salt、LoRA 和可识别多模态内容隔离命名空间；不支持的请求保持原生路径。

此前文档描述的 `VLLM_AGENTRIX_DP_KV_EVENTS` 和 `kv_routing.py` 不在当前固定的
CUDA 子模块中，不能按 GPU store/remove 事件索引解释当前实验。
缓存淘汰可能让逻辑提示过时，目标 engine 会自行验证并重算。
Router 不 pin GPU block，不同步调用 scheduler，也不跨 GPU 复制 KV。
清空缓存前会废弃路由提示；DP reset 接口只有所有 replica 都成功时才返回成功，
避免把单卡清空成功误报为整个 DP 的缓存已清空。

实现入口：

- [前缀与会话选择](../vllm/vllm/v1/engine/prefix_router.py)
- [Frontend 接入](../vllm/vllm/v1/engine/core_client.py)

## 验证入口

仅在服务器上运行，模型与输出路径显式配置，结果目录每轮独立：

- `vllm/tests/v1/engine/test_prefix_router.py`：无需模型的路由、执行进度与抢占回归。
- `benchmark/scripts/benchmark_prefix_aware_dp.py`：前缀重访、双卡已复制热点和冷请求对照；
  保存逐卡完成数、缓存统计、输出 token 哈希、TTFT 和批次耗时。
- `benchmark/scripts/benchmark_agent_session_dp.py`：多轮会话流量。

先在私有环境中设置 `MODEL_DIR` 和 `RESULTS_DIR`，分别指向模型目录和服务器结果目录。
两卡服务示例（使用同一模型、配置分别切换 `native` / `prefix_aware`）：

```bash
CUDA_VISIBLE_DEVICES=0,1 VLLM_SERVER_DEV_MODE=1 \
VLLM_AGENTRIX_DP_ROUTING_POLICY=prefix_aware \
vllm/.venv/bin/vllm serve "${MODEL_DIR}/Qwen3-8B" \
  --host 127.0.0.1 --port 8000 --served-model-name agentrix-dp \
  --data-parallel-size 2 --data-parallel-size-local 2 --api-server-count 1 \
  --attention-config '{"backend":"FLASH_ATTN"}' --dtype bfloat16 \
  --enforce-eager --no-async-scheduling \
  --max-model-len 8192 --kv-cache-memory-bytes 8589934592 \
  --max-num-seqs 32 --max-num-batched-tokens 2048 \
  --enable-prefix-caching --enable-prompt-tokens-details
```

`VLLM_SERVER_DEV_MODE=1` 用于测试期间清空前缀缓存，服务只监听 localhost。
`replicated` 测试用 `--warm-rank-counts 8,1` 明确构造访问次数不均的已复制前缀；
只在预热阶段指定 rank，计时请求由服务自行路由。

例如在服务器运行前缀重访测试：

```bash
vllm/.venv/bin/python benchmark/scripts/benchmark_prefix_aware_dp.py \
  --base-url http://127.0.0.1:8000 --model agentrix-dp \
  --policy-label prefix_aware --workload revisit --documents 12 \
  --prefix-tokens 4096 --suffix-tokens 64 --output-tokens 32 \
  --trials 3 --revisit-order shuffled --seed 20260927 \
  --output "${RESULTS_DIR}/dp-revisit/result.json"
```

每轮先确认 cache reset 成功，计时只覆盖重访/突发批次；预热和指标收集不计入。
TTFT 从 HTTP 请求发出到第一个实际输出 token ID，空文本特殊 token 也计入，
不把仅含角色或空 choices 的消息当作首 token。脚本核对每个请求的输入、输出 token 数，
并确认逐卡完成数之和等于本批请求数。

## 两张 H100 实测（2026-09-27）

实验使用两张 H100 PCIe 80 GB，逻辑 GPU 编号为 0/1。
Qwen3-8B BF16、FlashAttention 3、eager、同步调度、每卡 8 GiB KV 池，
其他参数使用上述服务命令。比较当前分支 `native`、修改前 `prefix_aware` 和新版
`prefix_aware`；三组共同使用“两卡 reset 都成功”的修复，只改变路由策略或实现。
原始工程的安装版本字符串不作为源码提交证明，实际代码以服务器 manifest 的哈希为准。

正式矩阵为两个 seed（`20260927` / `20261020`），分别按 native/old/new 和
new/old/native 顺序启动服务。每种场景、每个 seed 测三轮；下表为六轮指标的中位数，
P95 是每轮请求的 P95 再取中位数，不是所有请求混合计算的 P95。

- 前缀重访：12 份不同的 4,096-token 前缀，先逐个预热，再乱序并发重访，
  每个请求另有 64-token 后缀，输出 32 token。
- 复制热点：两卡都预热同一前缀，历史访问次数为 8/1，再发送 16 个并发请求，
  每个请求输出 256 token。预热固定 rank，计时阶段完全由服务路由。
- 冷请求：12 个不同前缀、输出 32 token，不预热；每轮清空两卡缓存。

| 场景 / 指标 | native | 旧 prefix-aware | 新 prefix-aware |
| --- | ---: | ---: | ---: |
| 重访：缓存 token 命中率 | 49.23% | 98.46% | 98.46% |
| 重访：TTFT P95，ms | 626.85 | 160.10 | 151.23 |
| 重访：批次耗时，ms | 1,576.18 | 1,087.41 | 1,178.63 |
| 复制热点：两卡请求数，每轮 | 8/8 | 9/7 | 8/8 |
| 复制热点：缓存 token 命中率 | 98.46% | 98.46% | 98.46% |
| 复制热点：TTFT P95，ms | 174.83 | 182.73 | 169.98 |
| 复制热点：批次耗时，ms | 7,835.92 | 7,840.83 | 7,828.28 |
| 冷请求：TTFT P95，ms | 1,069.16 | 1,068.17 | 1,069.76 |
| 冷请求：批次耗时，ms | 1,994.80 | 1,991.74 | 1,989.91 |

相对旧版，复制热点的 9/7 倾斜在六轮中全部修正为 8/8，缓存命中率不变；
TTFT P95 中位数下降 6.98%，批次耗时基本不变。重访 TTFT P95 下降 5.54%，
但批次耗时中位数增加 8.39%，不能据此宣称新版有整体吞吐收益。
相对 native 的大幅命中与延迟改善主要来自原有前缀亲和性，应与本轮改动区分。
冷请求全部保持 6/6、零缓存命中，批次耗时约 1.99 秒。

针对重访耗时增加，追加固定 seed `20260927` 的 old/new/new/old 配对复测，
每次启动三轮，同样为每种方式六轮，代码、模型和服务参数不变：

| 追加重访复测 | 旧版 | 新版 |
| --- | ---: | ---: |
| 缓存 token 命中率 | 98.46% | 98.46% |
| TTFT P95 中位数，ms | 147.19 | 178.91 |
| 批次耗时中位数，ms | 1,150.55 | 1,179.49 |

这次新版批次耗时增加 2.52%，TTFT P95 增加 21.55%；首轮的重访尾延迟改善
没有稳定复现。**本轮确认的是热点分配与状态处理的改进，没有建立总体性能提升，
重访延迟存在回退风险。** 两批数据分别保留，不用追加结果替换原始矩阵。
默认策略继续为 `native`，`prefix_aware` 仍需显式选择。

720 个正式计时请求均完成预期 token 数，逐卡完成计数与客户端一致、零抢占。
追加复测还有 144 个计时请求，使用相同检查；两批合计 864 个计时请求，
另外 120 个算术对话功能检查全部正确。13 项路由/进度/reset 回归测试和 vLLM
针对改动文件的 pre-commit 检查全部通过。测试覆盖取消、抢占后恢复、缓存清空和
未计算 token 的处理；性能矩阵未触发抢占，不能用它证明真实抢占负载的性能收益。

这是随机 token 构造的路由负载和小规模功能检查，不是官方 AgentX，也不是全面的
模型质量评估。只测试 Qwen3-8B、两卡、eager；没有验证其他模型、图执行、多 API
frontend 或高压淘汰场景。KV 池预分配容量没有减少。

原始数据仅保存在 H100 服务器：

```text
${RESULTS_DIR}/prefix-aware-dp-h100-20260927/
```

`source-manifest.json` 保存源码、模型元数据和硬件信息；`runtime-old/`、
`runtime-new/` 为隔离运行环境，`*-server-command.json` 为实际服务配置。
`controller.py`、`*-command.json` 保存调用，`analyze.py` 重算 `aggregate.json`，
追加配对复测由 `recheck.py` 执行，`analyze_recheck.py` 重算 `recheck-aggregate.json`。
日志、逐请求数据、GPU 采样和测试输出均在该目录。正式矩阵之前的两次预检单独归档，
不纳入上表。

验证后的两个运行时文件、benchmark 和回归测试已同步到服务器
`${REPO_ROOT}/` 对应路径；替换前核对旧文件哈希，并在实验目录
`deployment-backup/` 保留原文件。测试服务已停止，两卡恢复空闲，未更改默认路由策略。
