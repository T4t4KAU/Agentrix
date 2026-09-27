# ForkAttention profiling 指南

本页介绍算子、Cascade 与 Nsight 的 profiling 方法。

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

先通过私有环境设置 `REPO_ROOT` 和 `MODEL_DIR`，分别指向仓库与模型目录。

```bash
cd "${REPO_ROOT}/benchmark"
MODEL_PATH="${MODEL_DIR}/Qwen3-VL-8B-Instruct" \
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

系统级对照可使用 `benchmark/scripts/profile_tracelab.py`，记录：

- 到达时间、客户端提交延迟、TTFT/E2E/TPOT 和失败数。
- 两卡 running/waiting、实际 KV 容量、抢占及下层传输。
- 相同数据与代码 hash、准确 baseline、独立重复和单功能消融。

Cached tokens 不能直接换算为 branch 共享收益；抢占恢复可能复用自身缓存。
路由、placement 和 attention 同时开启的对照，不能归因于单个 kernel。

## FlashInfer Cascade 与 CUDA ForkAttention（2026-09-27）

新增入口 `benchmark/scripts/benchmark_flashinfer_cascade.py` 直接调用 FlashInfer 官方
`MultiLevelCascadeAttentionWrapper`。旧 profiling 脚本的 `CASCADE_ATTN` 调用的是
vLLM FlashAttention 后端的 Cascade，不能用它代替 FlashInfer 的结果。

固定 Qwen3.5-9B full-attention 的 16 Q heads、4 KV heads、D=256、BF16，
每个请求每步一个 query，使用 128-token 页。四条路径共用随机排列的物理页和同一份
vLLM 交错 K/V 缓存：FlashInfer 普通 paged attention、FlashInfer Cascade、
ForkAttention 默认 node 分区，以及可选 flatten 分区。小分支/短前缀用例显式执行算子，
不经过 serving 的准入阈值，因此不能据此判断生产默认配置是否启用 Fork。

先对照独立 FP32 grouped attention，使用 `atol=0.002, rtol=0.02`；再检查捕获图回放，
以及修改 Q 后的回放结果。性能阶段每个图包含 50 次完整调用，轮换执行顺序重复 8 轮。
记录每次 CUDA event 时间、数值误差、GPU 状态、版本及源码/二进制哈希。
这是固定计划、重复访问同一份 KV 的算子测量，包含所有设备端辅助内核，
不包含 CPU 规划、描述符 H2D、KV 写入或模型其他层；没有强制清空 L2。

本机现有 vLLM 扩展缺少与当前源码对应的可靠编译记录，正式测试使用
`benchmark/scripts/build_fork_attention_benchmark.py` 重建的独立扩展。
它使用当前 ForkAttention CUDA 源码和与 vLLM CMake 相同的模板实例，注册到独立命名空间；
原安装包保持不变。构建产物进入本地编译缓存，构建日志和实验输出直接传到服务器。
benchmark 必须通过 `--fork-library` 显式指定构建输出，避免使用旧扩展。

原始记录目录：
`${RESULTS_DIR}/flashinfer-cascade-20260927/`。
其中 `ascend-commits.json` 和 `ascend-committed-source.zip` 保存先行提交的 Ascend 实现，
子仓库提交为 `7ba43c5`，父仓库提交为 `49e5666`。

### 实测结果

执行设备为本机 RTX 5070（SM120，48 SM，48 MiB L2），FlashInfer 0.6.13、
PyTorch 2.11.0+cu130、Triton 3.6.0，独立扩展使用 CUDA 13.1.115 编译。
被测 vLLM 源码固定在 `437cf6727e790d1daa4a0105999ee4d634bda8bf`；
FlashInfer Cascade 的两个层级实际都选择 `fa2` 后端。

主矩阵覆盖 2/4/8/16 分支、1K/8K/32K/64K 前缀、128/1024-token 尾部，共 32 种形状，
每种使用 2026、2027、2028 三个随机种子。下表为尾部 128 token 的代表行，单位 µs；
先取每个种子 8 轮计时的中位数，再取三个种子结果的中位数。
普通 paged 列为同一 FlashInfer FA2 prefill 算子族的单 token 查询，未使用 Cascade。

| 分支 / 前缀 | FlashInfer paged | FlashInfer Cascade | Fork node | Fork flatten |
| --- | ---: | ---: | ---: | ---: |
| 2 / 8K | 36.57 | 33.73 | 60.35 | 98.96 |
| 2 / 32K | 233.02 | 246.90 | 246.65 | 395.09 |
| 8 / 1K | 24.15 | 27.48 | 40.55 | 96.74 |
| 8 / 8K | 109.94 | 58.16 | 73.92 | 101.67 |
| 8 / 32K | 444.63 | 259.84 | 261.35 | 408.64 |
| 8 / 64K | 868.28 | 478.89 | 484.68 | 811.05 |
| 16 / 32K | 1262.33 | 272.09 | 357.48 | 713.07 |
| 16 / 64K | 2507.50 | 494.38 | 670.98 | 1594.11 |

按形状汇总，Cascade 在 31/32 种形状中耗时更低；剩余一行 node 仅领先约 0.1%，
不足以支持稳定优势。Cascade 相对 node 的加速比几何平均为 **1.28×**，
相对 flatten 为 **2.36×**。32K/64K 的部分小分支用例接近持平；
不能将这些矩阵汇总值作为其他 GPU、布局、模型或服务流量上的收益保证。
Cascade 本身也不总优于普通 paged 路径，例如上表 8 分支/1K。

额外 25 个控制用例覆盖：8 个无共享、12 个双共享组、3 个非整页尾部、2 个 FP16。
加上主矩阵共 **121 个用例通过数值与图回放检查**，另有 4 个 smoke 和 1 个独立 profiling 用例。
双共享组的 node 与 Cascade 比较接近，node 在 2/12 个单 seed 用例中领先，最大约 3.6%；
尚未验证该小幅差异的稳定性。本轮没有测试多层嵌套的共享树。
所有路径按上述绝对与相对组合容差通过；全部检查中最大绝对误差约 0.00255，
不应写成全部误差小于 0.002，也不代表逐位一致。

独立 profiling 确认 Cascade 执行前缀和尾部的 FlashInfer attention、内部归并及层间 merge，
共 7 个设备内核；node 为当前源码编译的 2 个 attention 内核及 1 个 gather，
flatten 为 1 个 attention 内核及 1 个 reduce。内核数量更少没有转化成本轮的性能优势。
profiling 耗时不计入主矩阵。桌面显示进程仍在使用这张卡，未锁定 GPU 时钟；
原始记录包含运行前后的设备状态，因此接近 1% 的差异应谨慎解释。

`main.log` 保存完整矩阵，其他控制组分别保存于同名日志；
`summary.json`、`analyze.py`、`build-2.log` 和 `benchmark-source-final.zip`
保存汇总、分析入口、成功构建记录与源码。
`smoke-1.log` 使用旧扩展，只用于环境排查，不进入正式结果；首次失败的构建记录也保留。

复现时先运行 `benchmark/scripts/build_fork_attention_benchmark.py`，
从其 `build_complete.library` 记录设置 `FORK_LIBRARY`，再运行：

```bash
OMP_NUM_THREADS=4 MAX_JOBS=2 vllm/.venv/bin/python -u \
  benchmark/scripts/benchmark_flashinfer_cascade.py \
  --fork-library "$FORK_LIBRARY" --seeds 2026 2027 2028
```

构建和 benchmark 的 stdout/stderr 均应通过 SSH 直接写入服务器的新结果目录，
不在本地保存原始记录。以上均为 CUDA 算子结果，不是官方 AgentX 成绩，
也不能与 Ascend 算子耗时直接相除作为平台加速比。
