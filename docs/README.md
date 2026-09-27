# 文档导航

本目录保留系统设计、配置、构建和验证方法。

部署路径统一使用占位符：`${REPO_ROOT}` 为仓库根目录，`${MODEL_DIR}` 为模型目录，
`${RESULTS_DIR}` 为服务器结果目录，`${EXPERIMENT_DIR}` 为服务器实验归档目录。
`${DEPS_DIR}`、`${CACHE_DIR}` 分别表示依赖和安装缓存目录；
`${CUDA_VENV_DIR}`、`${CUDA_INSTALL_DIR}` 表示独立 CUDA 环境和安装目录。
复现前在自己的环境中设置这些变量；实际值和连接参数仅保存在私有配置中。
命令中的 localhost 和示例端口用于说明调用方式，不代表实际部署入口。

| 主题 | 入口 |
| --- | --- |
| 官方 AgentX、Ascend 路由、调度、混合缓存与图执行优化 | [AgentX Ascend](agentx_ascend.md) |
| NVIDIA 优化结果与 vLLM-Ascend 适配方案 | [优化结果与适配方案](nvidia_memory_results_and_ascend_plan.md) |
| KV 生命周期、淘汰、offload/restore 与当前限制 | [KV 内存管理](kv_memory_optimization_status.md) |
| Prefix/session-aware DP 与 GPU 驻留反馈 | [DP 路由](dp_routing.md) |
| 服务器环境、增量构建与清理边界 | [AutoDL 运维](autodl_build_and_benchmark.md) |
| 系统分层和历史应用路径 | [系统概览](agentrix_system_overview.md) |
| 算子、Cascade 和系统级 profiling 方法 | [ForkAttention profiling](fork_attention/forkattention_operator_profile.md) |

## 专题指南

- [ForkAttention](fork_attention/README.md)：算子 profiling、SGLang/llama.cpp 后端适配、模型兼容性与设计。
- [Coding Agent](coding_agent/README.md)：数据集、实时演示与任务质量验证。

## 其他专项指南

- 应用：[精确 prompt compaction](application_prompt_compaction.md)、
  [工具等待期间 KV trim 与 TTL](tool_kv_trimmer.md)。
- [SGLang LMCache](sglang_lmcache_usage.md)。

这些专项指南描述各自路径，不意味着都已在当前服务器、模型或子模块版本上重新验证。

## 维护规则

- 同一主题只维护一份主文档，不再为每轮调试新增报告。
- 本地只保留实验代码、复现方法和主文档中的关键实验数据、验证结论与局限。
- 完整日志、原始报告、逐请求数据和验证输出只保存在实验服务器；文档用占位符说明记录位置，不再按轮次下载到本地。
- 文档不记录实际地址、账号、个人信息、主机名、设备唯一标识、连接参数、凭据或部署绝对路径。
- 删除报告或图表时同步清理文档入口和失效链接。
