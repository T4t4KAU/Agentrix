# 文档导航

当前工作以服务器上的 GPU-only TraceLab 对照为准；旧硬件、旧分支和人工共享前缀实验只作为历史证据。

| 主题 | 入口 |
| --- | --- |
| 当前实验、原始 vLLM baseline、回放协议及失败记录 | [TraceLab 对照](tracelab_timeline_replay.md) |
| KV 生命周期、淘汰、offload/restore 与当前限制 | [KV 内存管理](kv_memory_optimization_status.md) |
| Prefix/session-aware DP 与 GPU 驻留反馈 | [DP 路由](dp_routing.md) |
| 服务器环境、增量构建与清理边界 | [AutoDL 运维](autodl_build_and_benchmark.md) |
| 系统分层和历史应用路径 | [系统概览](agentrix_system_overview.md) |
| 算子、Cascade 和系统级 profiling 方法 | [ForkAttention profiling](forkattention_operator_profile.md) |
| 旧实验分类、原始数据和恢复方法 | [历史实验索引](historical_experiments.md) |

## 专项指南

- 应用：[精确 prompt compaction](application_prompt_compaction.md)、
  [工具等待期间 KV trim 与 TTL](tool_kv_trimmer.md)。
- Coding agent：[数据集](coding_agent_dataset.md)、
  [实时演示](coding_agent_live_demo.md)、[任务质量 A/B](coding_agent_quality_ab.md)。
- 兼容性：[TP 模型](tp_model_compatibility.md)、
  [Qwen3.5/3.6 设计与限制](qwen35_qwen36_forkattention_design.md)。
- SGLang：[ForkAttention](sglang_forkattention_usage.md)、
  [LMCache](sglang_lmcache_usage.md)。
- llama.cpp：[CUDA/MUSA](llama_cpp_forkattention_usage.md)、
  [Apple Metal](llama_cpp_forkattention_apple_metal_support.md)、
  [EulerOS/CentOS](llama_cpp_forkattention_hce_centos_adaptation.md)。

这些专项指南描述各自路径，不意味着都已在当前服务器、模型或子模块版本上重新验证。

## 维护规则

- 同一主题只维护一份主文档，不再为每轮调试新增报告。
- 当前结果必须写清 baseline、源码/数据版本、容量、时间线、并发和重复次数。
- 算子时间、系统吞吐、任务质量、GPU 已分配显存和 live KV 分开报告。
- 失败结果保留原因与证据；新结果不能用旧的正向结果替代。
- 大型日志、安装包和 profiler 原始报告不放入 docs；旧实验散件压缩归档。
- 推理、测试和 profiling 在服务器执行，本机仅编辑与静态检查。
