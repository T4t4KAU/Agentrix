# 前缀亲和路由与 ForkAttention：设计及适用场景

Prefix-aware DP 的目标是让重复前缀访问同一推理副本，减少缓存未命中后的
prefill 重算。当前直接使用官方 `vllm-router==0.1.15` 的 `consistent_hash`，
应用提供稳定的文档或会话身份；本项目不维护另一套哈希路由和前缀索引。

## 与官方文章的关系

[vLLM AgentX 文章](https://vllm.ai/blog/2026-09-08-vllm-agentx#load-balance-does-not-guarantee-better-performance)
已讨论会话亲和路由与负载均衡之间的取舍。缓存亲和性是已有机制，本项目使用
[官方 router](https://github.com/vllm-project/router) 接入长会话和共享文档负载。
博客中的实验策略与具体发布组件不保证逐行相同，本文结果对应固定发布版本。

## 路由、缓存与算子的分工

| 层次 | 作用 | 不能据此推断的结论 |
| --- | --- | --- |
| consistent_hash | 根据 X-Session-ID 固定请求副本 | 身份相同不保证 KV 仍驻留 |
| 官方 APC | 根据精确前缀复用实际缓存页 | 命中率高不保证多个分支同批执行 |
| ForkAttention | 在实际批次中共享相同物理前缀的计算 | 不新增跨卡 KV 共享，不自动减少缓存池 |

原生 DP 按官方内部负载策略选择 rank；`consistent_hash` 保持身份亲和；
`cache_aware` 使用发布包自身的缓存估计与负载策略。启用
`--intra-node-data-parallel-size` 后，router 将同一后端的 DP ranks 作为
候选并设置 `X-data-parallel-rank`。

控制和指标采集直接访问引擎，推理请求经过 router。不能用 router 的估计
替代实际 KV 驻留与计算计数。稳定身份还需要正确的租户、模型和模板隔离；
离线公共语料的文档哈希不能直接当作多租户缓存安全方案。

## Ascend 合成会话与前缀重访

双 910B2、Qwen3.5-9B BF16、每卡 8 GiB KV、eager、同步调度，对比原生
DP、官方 `consistent_hash` 和 `cache_aware`。两个种子反转策略顺序，每种
策略各三轮；每轮独立启动引擎，各场景独立启动 router，避免估计状态残留。
请求采用 token-ID completions 与明确会话身份，不代表 chat 或官方 AgentX。

下表为六轮平均 TTFT，会话行只统计续聊：

| 场景 | 原生 DP | consistent_hash | cache_aware |
| --- | ---: | ---: | ---: |
| 会话续聊 | 892.33 ms | 453.65 ms | 491.75 ms |
| 前缀重访 | 1,312.79 ms | 396.56 ms | 482.67 ms |
| 双卡已有缓存 | 385.74 ms | 386.03 ms | 460.73 ms |
| 冷请求 | 2,439.58 ms | 2,476.37 ms | 4,805.32 ms |

`consistent_hash` 的续聊和重访缓存 token 比例分别为 97.96% 和 98.46%。
收益主要来自后续复用，双卡都已有缓存时与原生接近，冷请求也没有同类收益。
`cache_aware` 未在该矩阵建立更适合当前负载的优势，继续采用 `consistent_hash`。

## LongBench 共享文档 QA

数据取自 LongBench v1：按完全相同的 context 分组选出十六篇文档、三十四个
原始问题，其中十二篇来自 `multifieldqa_en`、四篇来自 `qasper`。文档长度
7,447～14,847 token，保留原文、问题和参考答案；这是共享文档子集，不是
完整 LongBench 榜单。

统一双 910B2、Qwen3.5-9B BF16、TP1/DP2、eager、同步调度、APC/Mamba
`align`、每卡 8 GiB KV、最大上下文 32,768、batch budget=8,192、最大
序列数 16。客户端并发四、输出上限 256 token，关闭 thinking 和自动重试。
先完成各文档首问，再交错发送后续问题。两个种子改变文档顺序并反转策略
顺序；每种策略、每种子一次独立运行。

下表对两个种子等权合并。TTFT 不含客户端并发排队，总耗时包含排队与生成；
F1/EM 使用相同子集的参考答案评分：

| 指标 | 原生 DP | consistent_hash | cache_aware |
| --- | ---: | ---: | ---: |
| 首问平均 TTFT | 2.013 s | 2.557 s | 1.644 s |
| 后续问题平均 TTFT | 0.823 s | 0.366 s | 0.826 s |
| 每轮本地计算 prompt token 均值 | 236,519 | 168,935 | 248,807 |
| 整批平均耗时 | 79.79 s | 76.65 s | 72.48 s |
| F1 均值 | 52.255% | 52.355% | 52.090% |
| Exact Match | 14.71% | 14.71% | 14.71% |

相对原生 DP，`consistent_hash` 的后续 TTFT 约降低 **55.5%**，本地 prompt
计算减少约 **28.6%**。两种顺序均出现这两项收益；首问均更慢，整批耗时则
一快一慢，合并均值不能表述为每轮都加速。`cache_aware` 整体耗时较短，但
后续缓存复用弱于 `consistent_hash`，策略选择取决于重访比例和首问延迟要求。

聚合质量接近，`consistent_hash` 与原生的逐题预测合计 58/68 完全相同，
不宣称严格输出等价。固定池下未建立设备容量下降；减少 prompt 计算本身
已是有效收益，不要求所有指标同时改善。

## ForkAttention 的联合执行

路由创造同卡复用机会，ForkAttention 还需检查同批请求的物理页确实共享。
CPU 规划、query 打包、FIA、LSE 合并与图参数更新都消耗时间；短前缀、少量
分支或 eager 提交时，这些成本可能抵消共享计算节省。

早期 7K～15K 文档、4K 共享门槛的 eager 对照虽然触发共享路径，但没有建立
计算量、内存或整体延迟收益。真实 QA 的输出长度也会变化，不能将整批时间
差直接当作算子速度差。当前不按会话名推断共享，不为凑批强行延迟 decode。

### 长前缀模型级对照

模型实验固定双 910B2、Qwen3.5-9B BF16、TP1/DP2、官方 `consistent_hash`、
每卡 8 GiB KV、APC/Mamba `align`，最大上下文 131,072、batch budget=8,192。
使用 Decode ACLGraph `FULL_DECODE_ONLY`、捕获大小 `[1,2,4,8]`，关闭
`npugraph_ex`。每分支私有尾部 128 token，固定生成 64 token。

扫描后采用新种子的独立服务对照：两个种子各两组重启配对，反转 Fork 开关
顺序；每次服务先取三个批次均值，再对四组配对等权合并。实验扫描
16K/32K/64K × 每卡四/八分支，共享门槛显式设为 16K 以包含控制形状；
通用默认门槛仍为 32K。下表仅列输出一致且四组均加速的形状：

| 共享前缀 / 每卡分支 | 原生整批均值 | Fork 整批均值 | 变化 |
| --- | ---: | ---: | ---: |
| 32K / 4 | 3.072 s | 2.898 s | −5.67% |

各组耗时改善为 2.81%～10.55%，配对输出一致。初次筛选该形状曾出现正负
变化，因此不推断所有输入均获益。其他扫描形状仍存在生成差异或收益方向
不一致，不因计时下降列入有效结果，也不剔除不一致请求后重算加速比。

双卡 HBM 采样峰值均值 **62,421 → 63,181.5 MiB**，增加 **760.5 MiB**；
进程树 PSS 差异约 12 MiB，不认定主存收益。每五秒采样可能遗漏瞬时峰值。
该对照保持相同路由、APC、图模式和请求计划，只衡量 Fork 增量。

### 输出一致性与真实长文档的边界

固定长度只控制生成工作量，仍需比较输出内容。进一步改变 HTTP 提交方式后，
每卡整批提交改善了各路径自身的可重复性，但部分形状的跨路径差异仍存在。
32K×4 在两种提交模式中保持配对输出一致；其他形状的模型级累积差异尚未
定位，不能一概解释为近似并列候选的舍入换位。

真实文档评价采用 LongBench NarrativeQA 的八份原文、每篇八个原始问题，
文档长度 23,782～48,832 token。先完成首问，再并发十六处理其余问题，正常
EOS、输出上限 256；两个种子只改变到达顺序。任务评分未下降，但完成时间
一快一慢，答案和输出长度也不完全相同，尚无稳定等输出加速结论。

高缓存命中不等于高共享批次覆盖率，单层注意力数值表现也不能替代整模型质量。
因此 ForkAttention 继续显式启用；NPU 架构、分段计算和 profiling 见
[Ascend 专题](agentx_ascend.md)。

## 官方策略之上的探索边界

应用侧 `PrefixPrefillGate` 尝试让同前缀的后续请求等待首个请求返回首 token
后再发送，以减少冷前缀同时 prefill。它是单进程准入原型，首 token 不保证
缓存仍驻留；尚未建立实质内存收益，保持默认关闭。

首次放置负载感知候选未建立优于原版 `consistent_hash` 的收益，已撤回。
历史 CUDA 私有 prefix-aware 路由也已被官方组件替代，其旧收益不移记为
当前 router 的性能结果。H100 当前尚无可用的官方 router 直接对照结论，
Ascend 数字不外推至 CUDA。

在稳定会话归属上，已知终止请求的选择性 CPU 备份已在两平台建立恢复与容量
收益，见 [KV 生命周期与分层缓存](kv_memory_optimization_status.md)。这是
应用生命周期策略，独立于哈希路由算法。

实现和复现入口见 [官方路由启动器](../benchmark/scripts/serve_dp_router.py)、
[路由实验控制器](../benchmark/scripts/run_ascend_router_comparison.py)、
[文档 QA 驱动](../benchmark/src/longbench_qa_runner.py) 及
[benchmark 说明](../benchmark/README.md)。
