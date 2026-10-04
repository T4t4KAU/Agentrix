# 技术文档

以 [框架层内存管理与跨平台推理优化](agentrix_cross_platform_optimizations.md)
作为技术汇报主文，集中说明方案、CUDA 与 Ascend 实验、资源成本和适用范围。
各专题补充实现细节，接口和运行方法维护在对应组件指南中。

## 汇报与专题

| 文档 | 内容 |
| --- | --- |
| [跨平台优化总文](agentrix_cross_platform_optimizations.md) | 整体方案、自主实现与框架能力、工具数据管理、分平台关键结果 |
| [Ascend 推理优化](agentx_ascend.md) | NPU 架构、ForkAttention 算子适配、profiling、混合缓存与图执行 |
| [KV 内存管理](kv_memory_optimization_status.md) | 生命周期、选择性备份、增长上下文与分支、设备容量和恢复代价 |
| [DP 路由](dp_routing.md) | 官方 consistent_hash、文档 QA、路由与 ForkAttention 的配合 |

工具数据的模型级效果见总文的 [H100 实验](agentrix_cross_platform_optimizations.md#cuda-h100-上的工具数据按需加载)，
页面共享、阶段回收和流式读写见 [主存实验](agentrix_cross_platform_optimizations.md#六-工具页面与主存实验)。

## 接口与运行指南

- [应用接口](../application/README.md)：会话提示、Prompt 去重、工具快照与分支生命周期。
- [Benchmark](../benchmark/README.md)：负载、测量方法与复现入口。
- [ForkAttention](fork_attention/README.md)：CUDA profiling、SGLang、llama.cpp 及模型适配。
- [Coding Agent](coding_agent/README.md)：数据集、任务质量评估与演示。
- [SGLang LMCache](sglang_lmcache_usage.md)：独立缓存后端的接入条件与使用方式。

各项收益保留对应的模型、负载、重复次数与资源成本，不能叠加为统一加速比。
组件存在不代表已接入当前运行配置，实验性和未接通能力以专题说明为准。
