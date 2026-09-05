# 历史实验索引

本页替代旧的逐轮报告。历史实验并非全部无效，但其硬件、分支、工作负载和
baseline 不同，不能作为当前 Agentrix 的总体性能结论。当前结果见
[TraceLab](coding_agent/tracelab_timeline_replay.md)。

## 原文与数据

整理前已备份完整 docs（包含未提交的 TraceLab 报告）：

- 本机：`/home/hwx/Documents/codes/agentrix-cleanup-20260905-tFg6KA/local-docs-before.tar.gz`。
- 服务器：`/root/autodl-tmp/agentrix-archive/cleanup-20260905/server-docs-before.tar.gz`。
- 已提交的旧报告也可从父仓库 `61ef01a1ed8b10a66c18682ba4d1e20d90ff754c`
  按下表原文件名读取，例如
  `git show 61ef01a:docs/dp_experiment_results.md`。不需要 reset 工作区。
- 小型 CSV、插图和 H20 EPD 绘图输入继续保留在
  [experiment_results](experiment_results)、[assets](assets)、
  [experiments](experiments)，仅用于历史追溯。
- 服务器较早的 residency/proactive/placement smoke 与 microprofile 已集中至
  `/root/autodl-tmp/agentrix-archive/cleanup-20260905/legacy-results.tar.gz`；
  55 个条目在归档内保留原目录名。散落的 Mooncake 日志、旧脚本和未提交源码包
  保存在同目录 `legacy-loose-files.tar.gz`。这两份归档及服务器旧 docs 均已复制到
  上述本机备份目录并核对 SHA256。新的实验不要写回历史目录。
- 服务器旧 LMCache 扩展单独保存在同目录 `legacy-lmcache-binaries.tar.gz`，
  也已备份到本机；它们不属于当前 Python 3.12/CUDA 12.8 的有效运行扩展。

## dp

| 原报告 | 保留的含义与边界 |
| --- | --- |
| `prefix_aware_dp_profile.md` | 15 个预热文档、单 token revisit，8.43→96.08 req/s；强亲和上界，不是生产吞吐 |
| `prefix_aware_dp_affinity_profiling.md` | 包括 shuffled arrivals；不同软件环境的独立实验，不能与前一报告拼成重复样本 |
| `session_aware_dp_profile.md` | 12 会话、单 token、prefix-aware 对 session-aware；不是原始 vLLM 对照 |
| `dp_kv_placement.md` | GPU 事件提示未建立显著加速，router-only P50 35.30→72.54 us；当前机制见 [DP 路由](dp_routing.md) |
| `dp_experiment_results.md` | Adaptive/Pressure、H20 DP=4/6/8、32K/64K/96K 等旧矩阵；容量、模型、人工分支均需随原文解释 |
| `coding_agent_dp8_experiment_results.md` | Coding-agent 系统回放，不等于真实源码任务成功率；质量验证另见 [任务 A/B](coding_agent/coding_agent_quality_ab.md) |

## memory

| 原报告 | 保留的含义与边界 |
| --- | --- |
| `kv_residency_shadow.md` | 逐轮 observer、planner、proactive、restore 调试和微基准；现行安全约束已合并到 [KV 管理](kv_memory_optimization_status.md) |
| `kv_tier_memory.md` | P/A/B/C/D/A 压力 smoke 恢复 512 tokens；单机 TCP/RDMA，不是跨机带宽结果 |
| `kv_lifecycle_reload_experiment.md` | 旧 GPU lifecycle/offload 实验；不覆盖当前 LMCache/Mooncake 长 trace |
| `tool_kv_trimmer.md` 旧实验章节 | Qwen3-0.6B、LangGraph、fanout 生命周期；live KV 减少不等于固定 GPU pool 缩小 |
| `tool_kv_ttl_predictor.md` | 合成工具时长上的 predictor 评估；不等于生产工具分布精度。机制合并至 [TTL/trim](tool_kv_trimmer.md) |

## systems

| 原报告 | 保留的含义与边界 |
| --- | --- |
| `main_experiment_results.md` / `main_experiment_matrix.md` | 多轮 shared-prefix、offload、DP、TP 矩阵；旧版本存在修正和排除组，不能只摘最快数字 |
| `hotpot_agentrix_experiment.md` | HotpotQA/LangGraph 长共享前缀 positive control；不是 TraceLab 原始时间线 |
| `experiment_results/single_gpu.md` / `experiment_results/tp_accuracy.md` | 生成汇总已移出主文档，原始 CSV 保留；TP 输出一致性不等于任务质量 |
| `forkattention_operator_profile.md` / `forkattention_vs_cascade_nsight.md` 的旧结果 | RTX 5070/H20 的特定 tensor geometry；算子指标不能换算为当前端到端加速倍数 |

旧实验入口仍在 benchmark/scripts：
`run_main_experiment.sh`、`run_vllm_dp_full_dataset.sh`、
`run_hotpot_agentrix_e2e.sh`。它们可能默认开启较旧功能，应先阅读脚本，
不要作为当前服务器默认配置。当前 profiling 方法见
[ForkAttention 指南](fork_attention/forkattention_operator_profile.md)。

## multimodal

| 原报告 | 保留的含义与边界 |
| --- | --- |
| `multimodal_agent_dp_experiment_results.md` | Vision encode/prefill 干扰、重算/搬运、cache/load 冲突的独立实验 |
| `multimodal_agent_dp_vllm_omni_comparison.md` | vLLM 0.25 与 Omni/vLLM 0.28 的版本及插件因素混合，不能隔离为 Agentrix 收益 |
| `qwen2_5_vl_7b_h20_epd_experiment_results.md` | H20/Qwen2.5-VL 的 EPD 实验；当前 RTX 5090/Qwen3-VL 不直接适用 |

## TraceLab 旧协议

`tracelab_upstream_comparison.md` 已合并到
[TraceLab 的历史闭环与失败记录](coding_agent/tracelab_timeline_replay.md#legacy-and-failures)。
16-session 闭环的 8.53% 改善与 92-session open-loop 的 19.05% 退化是不同实验；
两组原始数据均保留，不能相互替代。
