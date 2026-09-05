# ForkAttention profiling 指南

只在服务器执行推理与 profiling。本页合并原算子报告和 Cascade/Nsight 操作说明；
旧硬件上的逐点数字见 [历史索引](historical_experiments.md#systems)，不作为当前系统性能结论。

## applicability-boundary

ForkAttention 的收益依赖同批 query 引用相同的物理 KV 前缀，不能仅由 prompt
文本相似或会话数推断。短前缀、长私有 suffix、小 cohort、抢占、错开到达和不支持的
shape 都可能削弱收益，甚至出现退化。原始 vLLM 已有 APC 和条件启用的 Cascade；
baseline 不能描述成完全不共享前缀。

算子实验必须固定 GPU、dtype、head geometry、block size、prefix/suffix、
query 数和物理 block table。先校验输出，再 warmup，最后进入 profiler range。
同一 invocation 的所有 kernel 都计入成本：Fork 的 gather、Cascade 的
prefix/suffix attention 与 merge，不能只比较最长 kernel。

## Nsight Compute

从服务器仓库根目录执行，输出目录每次使用新名字：

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
VLLM_PYTHON="$PWD/benchmark/.venv/bin/python" \
PREFIX_TOKENS=4096 PRIVATE_SUFFIX_TOKENS=128 BRANCHES=8 \
OUTPUT_DIR="$PWD/benchmark/results/fork_cascade_ncu_new" \
bash benchmark/scripts/run_fork_cascade_ncu.sh
```

需要交互式完整页面时使用 `run_fork_cascade_ncu_ui.sh`，参数相同。
保留 `.ncu-rep`、CSV、日志与汇总 JSON。完整 section 会触发多次 kernel replay，
其时长和缓存状态不应直接混入单组 counter 的计时。

重点检查 DRAM 与 L2 字节、CTA 数量、occupancy、stall、指令和 gather 成本：
DRAM 下降不是共享收益的唯一证据，L2 命中也不代表没有重复访问。

Flash/Fork 参数矩阵入口：

```bash
PREFIXES=4096,8192,16384 QUERY_COUNTS=2,4,8 \
MATRIX_DIR="$PWD/benchmark/results/ncu_matrix_new" \
bash benchmark/scripts/run_fork_attention_ncu_matrix.sh
```

矩阵脚本会跳过已有完整 cell；因此变更配置或版本后必须更换目录，
不能把旧 cell 混进新一轮。

## Nsight Systems

```bash
cd /root/autodl-tmp/Agentrix/benchmark
MODEL_PATH=/root/autodl-tmp/models/Qwen3-VL-8B-Instruct \
VLLM_BIN="$PWD/.venv/bin/vllm" \
PREFIX_TOKENS=8192 BRANCHES=16 OUTPUT_TOKENS=64 \
MAX_MODEL_LEN=32768 MAX_NUM_SEQS=16 \
OUTPUT_DIR=results/fork_cascade_nsys_new \
bash scripts/run_fork_cascade_nsight.sh
```

这个入口使用独立进程比较普通 Flash、显式 Cascade 和 Fork。检查实际 kernel 和
fallback，不能仅以 backend 环境变量认定专用路径已执行。
Nsys 包含模型、调度、API、CUDA Graph；Ncu 用于详细算子 counter，两者分开解释。
旧脚本在当前版本上的兼容性应先 smoke 验证，本次文档整理没有重新跑 GPU 实验。

## 系统级对照

当前统一入口为 [TraceLab](tracelab_timeline_replay.md)，报告：

- 到达时间、客户端提交延迟、TTFT/E2E/TPOT 和失败数。
- 两卡 running/waiting、实际 KV 容量、抢占及下层传输。
- 相同数据与代码 hash、准确 baseline、独立重复和单功能消融。

Cached tokens 不能直接换算为 branch 共享收益；抢占恢复可能复用自身缓存。
路由、placement 和 attention 同时开启的对照，不能归因于单个 kernel。
