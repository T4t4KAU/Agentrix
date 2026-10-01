# 官方 DP 路由与历史实验

当前入口统一使用官方 `vllm-router==0.1.15`。已删除自定义 internal DP
前缀/会话路由器、Ascend 实验代理及其专用性能模拟器。vLLM frontend 恢复官方选路，
仅保留“所有 DP rank 都 reset 成功才返回成功”的正确性修复。

## 与官方文章的关系

[官方文章](https://vllm.ai/blog/2026-09-08-vllm-agentx#load-balance-does-not-guarantee-better-performance)
已比较 session-aware sticky routing 与负载均衡，缓存亲和性不是本项目原创。
现在直接使用 [vllm-project/router](https://github.com/vllm-project/router) 的发布组件，
没有复制或另写一套哈希、前缀索引、过载阈值或会话生命周期算法。
博客中的实验策略与该发布包不保证逐行相同；当前采用的是官方公开组件。

## 当前策略与接入

| 入口 | 行为 |
| --- | --- |
| 直连 vLLM backend | 官方 internal DP 负载均衡，作为 native 对照 |
| 官方 `consistent_hash` | 以 `X-Session-ID` 保持多轮会话归属；当前 AgentX 默认策略 |
| 官方 `cache_aware` | 使用发布包自身的缓存亲和性及负载策略；需要独立对照 |
| 官方 `round_robin` | 官方 router 内的轮转对照 |

官方 router 的 `--intra-node-data-parallel-size` 将同一 backend 的 DP ranks
作为候选，并设置 `X-data-parallel-rank`。推理请求发到 router；清缓存、采集逐卡
指标及显式指定 rank 的预热发到 backend。不要用 router 的 metrics 代替 engine metrics。
没有 session 标识时按官方 fallback 处理，不再维持本项目的旧语义。
`cache_aware` 不承诺等价于已删除的 token 哈希前缀匹配，也不证明 GPU KV 实际驻留；
尤其不能将 chat、token-ID completion 和共享前缀分支视为已经验证相同行为。

`VLLM_AGENTRIX_DP_ROUTING_POLICY`、旧 `VLLM_FORK_ATTN_DP_PREFIX_*` 及
`agentrix_session_id/turn/history_tokens` 路由协议均已停用。引用旧开关的 11 个历史 shell recipes 会立即停止并提示新入口，避免把
已失效的开关当成优化继续测量；新实验通过下面的官方 HTTP 入口执行。

```bash
# 在实验服务器中安装独立路由环境，不改变推理环境依赖。
uv venv "${REPO_ROOT}/.router-venv"
uv pip install --python "${REPO_ROOT}/.router-venv/bin/python" \
  -r "${REPO_ROOT}/benchmark/requirements-router.txt"

# 先使用已有 serve 配置启动 backend，设 BACKEND_URL 和路由监听参数。
"${REPO_ROOT}/.router-venv/bin/python" \
  "${REPO_ROOT}/benchmark/scripts/serve_dp_router.py" \
  --worker-urls "${BACKEND_URL}" --policy consistent_hash \
  --intra-node-data-parallel-size 2 \
  --host "${ROUTER_HOST}" --port "${ROUTER_PORT}" \
  --prometheus-port "${ROUTER_METRICS_PORT}"
```

启动器只检查版本和加载官方 Rust router；`--check` 在启动模型前检查依赖。
Ascend `run_retention.py` 默认采用该入口，并在 manifest 记录官方包、版本和策略；
比较器拒绝把旧代理与新 router 混成仅缓存策略不同的 A/B。

两卡合成会话对照入口为 `benchmark/scripts/run_agent_session_dp_profile.sh`，
默认比较 native / consistent_hash / cache_aware，attention 使用官方 FlashAttention。
请求驱动环境安装 `benchmark[dp]`，提供 aiohttp 与 Prometheus 指标解析依赖。
设置模型、可用卡及服务器 `OUTPUT_ROOT` 后再运行；该脚本不是官方 AgentX。
`benchmark_agent_session_dp.py` 和 `benchmark_prefix_aware_dp.py` 均支持
`--base-url "${ROUTER_URL}" --control-url "${BACKEND_URL}"`。
原始数据只写到服务器 `${RESULTS_DIR}`，每个冷启动对照重启 router 和 backend；
只 reset engine 不会同步清除官方 router 自身的估计状态。

## 验证与限制

真实发布包的 Rust 进程配合模拟双 rank backend，已通过五项 CPU 接口测试：
多轮会话固定 rank、两个 rank 均可被选中、token-ID prompt 与扩展元数据透传、
SSE 首 chunk 在请求完成前送达；也检查 `cache_aware` 的请求透传与流式路径。
这证明接口兼容，不证明前缀命中、硬件输出正确性或性能提升。

```bash
uv pip install --python "${REPO_ROOT}/.router-venv/bin/python" aiohttp pytest
"${REPO_ROOT}/.router-venv/bin/python" -m pytest -q \
  "${REPO_ROOT}/experiments/agentx-ascend/test_session_router.py"
```

原生 DP 十项负载选择、完成计数和全 rank reset 单元检查通过；
缓存与 checkpoint 的 31 项回归、原生准入的六项调度检查通过；单元环境用固定 token IDs
代替 gated tokenizer，不加载模型。本轮未完成新 router 的双卡官方 AgentX 性能复测，
下面的历史收益不能移记为官方 router 的实测收益。

结构化 Agent Hints 父/根路由 API 已在此前撤回；独立的
[分叉 checkpoint 提示](kv_memory_optimization_status.md#agent-hints提前保留分叉状态2026-09-27)
也在后续 H100 完整回放未建立收益后撤回。官方路由与缓存机制继续沿用。

## 两张 H100 实测（2026-09-27）

实验使用两张 H100 PCIe 80 GB，逻辑 GPU 编号为 0/1。
Qwen3-8B BF16、FlashAttention 3、eager、同步调度、每卡 8 GiB KV 池，
历史配置为 eager、同步调度、8,192 上下文、每卡 8 GiB KV、32 并发槽、
2,048 批 token，FlashAttention + APC。比较当时分支 `native`、修改前 `prefix_aware` 和新版
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
当时默认策略为 `native`；本轮已删除自定义 `prefix_aware`，以下数据仅描述历史实现。

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
