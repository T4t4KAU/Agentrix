# Agent 场景性能探索：2026-09-05

本页包含 Qwen3-VL-8B-Instruct 的首轮探索和文末 Qwen3-8B 复测。后者完成
96 次回放、2,364 个请求：长源码分支耗时降低约 28–48%，多轮约 28–47%，
收益仍来自 prefix-aware DP；单卡 Fork 未显示稳定收益。

## 结论

当前服务器上找到了有重复验证的收益点：**长上下文、多轮/多根分支负载中的
prefix-aware internal DP 路由**。有效配置使用 `FLASH_ATTN`，没有开启 active
placement、GPU KV events、CPU/Mooncake 或应用压缩；不能将结果归因于 ForkAttention。

三个源码数据集的三次冷缓存复测中，多轮整批耗时降低 26–47%，分支整批耗时降低
29–62%。将优化组换到原始 vLLM 使用的 GPU 后，单次分支复测仍降低 26–44%，
多轮仍降低 26–47%。Django 分支的 62% 是三次中位数，不能忽略其 15–22 s 波动。

单卡 ForkAttention 探索没有找到稳定的端到端收益；短前缀和部分支持形状反而退化。
这些受控负载结果与 [TraceLab 开环退化](tracelab_timeline_replay.md) 分别成立。
本次没有修改推理引擎。

## 公平性与负载定义

- 服务器：`connect.bjb2.seetacloud.com:50887`；4×RTX 5090，约 32 GiB/卡。
- 模型：本地 `Qwen3-VL-8B-Instruct`，BF16，仅文本；Python 3.12.3、Torch 2.11.0+cu128。
- 原始 baseline：独立安装的官方 vLLM 0.25.0 wheel；源码/编译器来源及 cu129/cu128
  差异沿用 [baseline 说明](tracelab_timeline_replay.md#baseline-isolation)。
- Agentrix vLLM：`1c9a17fd9`；父仓库在实验期间提交文档和 TraceLab 工具至
  `738b2fa`，推理子模块未变化。生成的 `0.0.0+02702bdf4` 不是源码版本依据。
- 正式对照均 TP=1 / DP=2、APC 开启、CUDA Graph 开启、同步调度；每 replica
  4,608 个 16-token KV blocks，即 73,728 token slots；`max_num_seqs=64`、
  `max_num_batched_tokens=8192`、`max_model_len=32768`、GPU utilization budget=0.90。
- 每场景前调用 reset prefix cache；模型启动和通用 warmup 不计时，场景自身的冷
  bootstrap/prefill 计入总耗时。正式复测在 reset 后及最终 counters 读取前等待
  1.1 s，这些等待不计入场景耗时。
- 官方组使用 GPUs 0/3；prefix-aware 与同源码 native 组使用 GPUs 1/2。
  最后在 GPUs 0/3 再跑 prefix-aware。初筛可使用不同 GPU 并发执行；正式组也共享
  主机 CPU，所以提供换卡复测与同源码消融，仍不声称消除了所有运行噪声。
- 正式组 `dp_upstream_confirm`、`dp_prefix_confirm`、`dp_native_confirm`、
  `dp_prefix_gpu_swap` 的 harness hash、workload hash 相同。各组保留实际命令、
  导入路径、源码 hash、harness 快照、GPU 身份/采样、逐请求及 Prometheus 记录。

| 类型 | 数据与编排 | 输出长度 |
| --- | --- | --- |
| 长源码分支 | Django/SQLite/FFmpeg 各取 `cases_30k_b16.jsonl` 的全部 4 个 case；每根截取 Qwen tokenizer 下前 24,000 tokens；4 个根 bootstrap 完成后，每根取已有的前 8 条 private instruction，固定 seed 全局打乱，内部 DP 自行路由 | 每根 bootstrap 32，每分支 256；共 36 请求 |
| 长源码多轮 | 同样的 4 个 case、每会话 24,000-token 初始历史；各自连续 4 轮，每轮将本引擎实际生成 token IDs 与固定跟进观察追加到历史；会话间并发，轮间有因果依赖 | 每轮 128；共 16 请求 |
| SWE-bench 多轮 | Verified 文件前 8 条，初始长度为 713/1194/10000/3067/2551/2818/10000/10000；较长记录截到 10K；4 轮 | 每轮 128；共 32 请求 |
| AgencyBench 多轮 | 文件前 8 条，初始长度 331–8263；4 轮 | 每轮 128；共 32 请求 |
| 宽松容量对照 | Django 4 根、每根 12,000-token 上下文×8 分支，其他规则一致 | 每根 32，每分支 256 |

长源码输入是实际源码材料的截取，没有为了填长度循环重复文本。分支和多轮编排是
受控构造，不是这些数据集的原始在线 trace。API 使用 token-ID completions；
bootstrap 输出未拼入冻结的分支输入，以保证两组分支 prompt 完全一致。多轮则
追加各引擎实际输出，后续内容可能不同，但 token 数一致，保留 prompt/output hash。
使用 `ignore_eos=True` 固定工作量；没有执行真实工具、修补仓库或评估任务成功率。

## 正式复测：官方 vLLM 对 prefix-aware

每格是 **3 次冷缓存运行的中位数**，括号是最小–最大值，不是置信区间。
重复运行复用同一已启动服务；换卡验证另外启动新服务。所有耗时包含冷 bootstrap。

| 场景 | 官方 vLLM，s | Agentrix Flash + prefix-aware，s | 耗时降低 |
| --- | ---: | ---: | ---: |
| Django，4 根×8 分支，24K | 39.94（39.74–40.00） | 15.26（15.19–22.39） | 61.79% |
| SQLite，4 根×8 分支，24K | 39.96（39.86–40.02） | 22.42（22.37–22.52） | 43.90% |
| FFmpeg，4 根×8 分支，24K | 30.56（27.75–30.65） | 21.73（15.18–22.30） | 28.90% |
| Django，4 会话×4 轮，24K | 26.97（24.73–29.33） | 14.32（14.27–14.32） | 46.91% |
| SQLite，4 会话×4 轮，24K | 24.32（19.26–27.17） | 14.29（14.28–14.33） | 41.25% |
| FFmpeg，4 会话×4 轮，24K | 19.36（19.33–22.43） | 14.27（14.25–14.31） | 26.29% |
| SWE-bench，8 会话×4 轮 | 13.39（12.51–13.45） | 10.14（9.89–10.18） | 24.26% |
| AgencyBench，8 会话×4 轮 | 9.92（9.78–10.07） | 8.77（8.50–8.81） | 11.57% |
| Django，4 根×8 分支，12K | 11.85（11.70–11.95） | 11.57（11.46–12.63） | 2.39%，范围重叠 |

不要把耗时降低百分比描述成吞吐提升百分比；例如 39.94→15.26 s 对应约 2.62×
整批输出吞吐，而不是吞吐只提高 61.79%。本实验未计模型启动成本。

## 同源码消融与换卡

同源码 native 组只关闭亲和路由，继续使用 Agentrix `FLASH_ATTN` 和相同容量。
此组与 prefix-aware 正式组均使用 GPUs 1/2，排除了将全部收益解释为 wheel/源码
差异的可能；原始 baseline 的绝对差异仍包含编译和运行噪声。

| 场景 | 同源码 native，中位数/范围，s（n=2） | prefix-aware，s（n=3） | prefix-aware 换到 GPUs 0/3，s（n=1） |
| --- | ---: | ---: | ---: |
| Django 分支 | 45.37（39.32–51.42） | 15.26 | 22.56 |
| SQLite 分支 | 39.44（39.24–39.64） | 22.42 | 22.55 |
| FFmpeg 分支 | 36.11（33.06–39.16） | 21.73 | 22.34 |
| Django 多轮 | 20.87（19.52–22.23） | 14.32 | 14.34 |
| SQLite 多轮 | 22.25（22.20–22.29） | 14.29 | 14.45 |
| FFmpeg 多轮 | 20.93（19.54–22.33） | 14.27 | 14.34 |

收益主要表现为减少重复 prefill，而非更快的 decode kernel：

- Django 分支正式组的 `local_compute` token 中位数从 307,056 降至 96,768；
  SQLite 分支从 307,064 降至 120,680。
- Django 多轮从 262,704 降至 96,288；SQLite 多轮从 285,056 降至 96,288。
  三个源码多轮优化组的计算量均接近初始 96K 上下文加后续少量新增输入。
- 同源码 native 的多轮 TTFT 中位数有时也很低，但少数长历史 miss 会拖慢整批。
  因此不只看请求 P50，也保留总耗时、计算量和逐请求记录。
- **抢占尚未解决。** SQLite 分支优化组每次仍有 11 次抢占，原始组约 3 次；
  换卡后三个分支优化组也各有 11 次。缓存复用减少重算仍能缩短总耗时，但不能
  据此宣称压力管理已完善。Django/FFmpeg 的 15–22 s 波动说明路由与容量交互仍需改进。
- 12K 对照双方可以在每卡缓存更多不同根，少量 prefill 节省未转化为明确端到端收益。

## ForkAttention 与其他初筛结果

初筛比较官方 vLLM、同源码 Flash、同源码 Fork；单卡容量相同，Graph/APC 都开启。
测试了源码分支以及 SWE-bench、AgencyBench、AgentBoard、AppWorld 的首条原生
prompt 扩展为 16 个分支；后四者的共享输入分别为 713/2448/691/81 tokens。
还测试了 2K/8K/16K/24K 的受控前缀与 4–64 分支。

没有发现可推广的单卡 ForkAttention 收益。例如：

- 2K×16：官方约 4.90 s，Fork 5.88 s（各一次）。
- 8K×16：官方 6.91 s，Fork 8.56 s（各一次）。
- 16K×8：官方约 7.66 s，Fork 11.41 s（各两次中位数）。
- 28K 源码×16 和 24K×16/32/64 大体接近 Flash，但不能称为专用算子收益。

当前 `_MAX_FORK_PLAN_BLOCK_VISITS=20_000`。24K×16 在 block size=16 时仅前缀
就需 24,576 次 block 访问，超过阈值而回退 Flash。长前缀不代表 Fork 必然启用。
8K×16 的独立 Torch GPU trace 确认执行了 `fork_fwd_splitkv_kernel` 和 `gather_kernel`，
该场景仍然退化。Profile 运行不纳入正常性能对照。

补充的服务器 CPU microbenchmark 中，构造 8K×16 的 `_build_fork_plan` 中位数
约 2.62 ms，24K×8 约 4.00 ms；它不含完整 metadata/打包成本，且在其他测试运行时
执行，不能视为端到端退化的完整归因。后续值得针对计划增量更新和 CPU 开销做优化。

另有单次 `session_aware` 与提高 `prefix_min_blocks` 的探索，结果保存在全量汇总，
没有将其中的最快值替换正式对照，也没有因此更改默认策略。

## 复现与产物

入口：[运行脚本](../../benchmark/scripts/profile_agent_scenarios.py)、
[汇总脚本](../../benchmark/scripts/summarize_agent_scenarios.py)。
本机仅编辑/静态检查，运行均在服务器。沿用现有 `benchmark/.venv`，无需安装新环境。

```bash
cd /root/autodl-tmp/Agentrix
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# 已有 manifest 不覆盖；重新生成时使用新路径。
benchmark/.venv/bin/python benchmark/scripts/profile_agent_scenarios.py \
  --prepare --workload benchmark/results/agent_scenarios_new/workload.json

# 官方组；确保所选 GPU 空闲，并使用新 output 目录。
benchmark/.venv/bin/python benchmark/scripts/profile_agent_scenarios.py \
  --workload benchmark/results/agent_scenarios_20260905/workload_v3.json \
  --runtime-root benchmark/results/upstream_vllm_0_25_0 \
  --output benchmark/results/agent_scenarios_new/upstream \
  --label upstream --gpus 0,3 --port 8150 --repeats 3 \
  --cases django_forest4_p24000_b8,django_sessions4_turns4

# 原始组结束后，在同一组 GPU 运行优化组。
benchmark/.venv/bin/python benchmark/scripts/profile_agent_scenarios.py \
  --workload benchmark/results/agent_scenarios_20260905/workload_v3.json \
  --runtime-root vllm --backend FLASH_ATTN --policy prefix_aware \
  --output benchmark/results/agent_scenarios_new/prefix \
  --label prefix --gpus 0,3 --port 8150 --repeats 3 \
  --cases django_forest4_p24000_b8,django_sessions4_turns4

benchmark/.venv/bin/python benchmark/scripts/summarize_agent_scenarios.py \
  benchmark/results/agent_scenarios_new
```

同源码消融将 `--runtime-root` 保持为 `vllm`、`--policy` 改成 `native`。
其余参数保持不变。直接部署对应路由开关为
`VLLM_AGENTRIX_DP_ROUTING_POLICY=prefix_aware`；本次使用默认 min blocks=4、
work slack=8192。不要把本实验变成默认同时开启 placement/offload 的启动配置。

产物根目录：服务器和本机的 `benchmark/results/agent_scenarios_20260905/`。

- [全量记录表](../../benchmark/results/agent_scenarios_20260905/all_measurements.md)
- [聚合 CSV](../../benchmark/results/agent_scenarios_20260905/aggregate.csv)
- [逐次完整摘要](../../benchmark/results/agent_scenarios_20260905/all_trials.json)
- [正式 workload](../../benchmark/results/agent_scenarios_20260905/workload_v3.json)
- `fork_profile/profile/` 保存压缩 GPU trace；`planner_micro.json` 保存计划构建测量。

正式 workload 的规范化 JSON SHA256：
`0bdee1ccef61500a848fef98ce7df8caefb3b1d9f0371b3830049b0fea26160f`。
初筛阶段脚本逐步增加场景及 counters 等待；正式四组使用相同冻结版本，不能把
早期初筛当成额外正式重复样本。

总计保存 166 次完成运行（含 1 次独立 GPU profiling）、4,201 个完成请求。
生成接口逐请求验证输入/输出 token 数、最终 usage 和流结束标记，失败即中止，
不做 POST 重试或仅成功请求聚合。正式验证共 72 次运行。新增两个脚本通过 Ruff
静态检查；本次未修改或重新验证推理 kernel 数值正确性。实验结束后所有自有服务
退出，四张 GPU 均恢复空闲。

## Qwen3-8B 复测：2026-09-05

用户新增的 `/root/autodl-tmp/models/Qwen3-8B` 已完成同协议 DP 复测。
模型为纯文本 Qwen3ForCausalLM，36 层、32 Q heads、8 KV heads、head dim 128，
BF16。使用自身 tokenizer 重新生成 manifest，仍使用 token-ID completions，
没有应用 chat template 或切换 thinking 模式；固定输出长度，不评估思考链、
工具执行或任务成功率，也不比较两个模型的任务能力。

沿用本页负载和容量：TP=1、DP=2、APC/Graph 开启、同步调度、每卡 4,608×16
KV token slots、32K 上限、8,192 batched tokens、64 sequences。相同 token
slots 不代表与 VL 模型相同的 KV 字节数；本模型内各组容量一致。官方 wheel 和
Agentrix 源码沿用前述版本。没有修改推理引擎或开启 offload。

`dp_upstream` 在 GPUs 0/3、`dp_prefix` 在 GPUs 1/2 各测 9 场景×3 次；
`dp_native` 在 GPUs 1/2 对 6 个长源码场景各测 2 次；`dp_prefix_swap` 在
GPUs 0/3 各测 1 次。共 72 次 DP 回放全部成功。各组并行时共享主机 CPU；
下表是三次冷缓存中位数（min–max），不是置信区间，耗时包含冷 bootstrap。

### 官方 baseline 与 prefix-aware

| 场景 | 官方，s | Flash + prefix-aware，s | 耗时降低 |
| --- | ---: | ---: | ---: |
| Django 24K 分支 | 42.31（39.09–51.33） | 22.05（15.07–22.13） | 47.88% |
| SQLite 24K 分支 | 39.70（39.56–39.72） | 22.07（21.99–22.18） | 44.40% |
| FFmpeg 24K 分支 | 30.61（27.70–34.26） | 22.03（15.17–22.06） | 28.03% |
| Django 24K 多轮 | 19.79（19.77–22.22） | 14.18（14.16–14.20） | 28.35% |
| SQLite 24K 多轮 | 26.64（19.47–27.07） | 14.21（14.19–18.52） | 46.66% |
| FFmpeg 24K 多轮 | 26.79（19.77–27.06） | 14.13（14.13–14.16） | 47.26% |
| SWE-bench 多轮 | 12.42（12.27–13.00） | 10.01（10.01–10.03） | 19.37% |
| AgencyBench 多轮 | 9.97（9.97–9.99） | 8.70（8.52–8.71） | 12.77% |
| Django 12K 分支 | 11.52（11.49–11.54） | 11.34（11.30–11.54） | 1.57% |

长源码分支耗时降低约 28–48%，多轮约 28–47%；12K 宽松容量对照只有约
1.6% 差异，没有明确的实用收益。

### 同源码消融、换卡和计算量

| 场景 | 同源码 native，n=2，中位数（范围），s | prefix-aware 换卡，n=1，s | 官方→prefix-aware local_compute tokens 中位数 |
| --- | ---: | ---: | ---: |
| Django 24K 分支 | 38.47（38.43–38.52） | 22.34 | 354,784→120,688 |
| SQLite 24K 分支 | 45.20（39.24–51.16） | 22.14 | 307,064→120,680 |
| FFmpeg 24K 分支 | 30.42（30.02–30.81） | 22.23 | 262,440→120,712 |
| Django 24K 多轮 | 21.72（21.50–21.93） | 14.18 | 144,816→96,288 |
| SQLite 24K 多轮 | 22.04（22.01–22.07） | 14.34 | 262,640→96,288 |
| FFmpeg 24K 多轮 | 20.87（19.63–22.11） | 14.21 | 262,640→96,288 |

同源码关闭亲和路由后仍明显变慢，换卡后收益保持，支持减少重复 prefill 的解释。
官方 wheel 与源码编译差异、运行噪声仍存在。分支优化组抢占中位数均为 11，
多轮均为 0；Django/FFmpeg 分支仍有 15–22 s 波动，SQLite 多轮一次为 18.52 s。
不能宣称抢占已经解决或物理显存分配减少。多轮请求 TTFT P50 未必改善，少数
长历史 miss 可拖慢整批，需结合整批耗时和计算量。

### 产物与复现

本轮结果在服务器及本地 `benchmark/results/qwen3_8b_agent_20260905/`。
保留 model_identity、workload、各组 configuration/harness、逐请求及指标采样、
全量汇总。模型身份包含配置/tokenizer/权重索引 hash 及分片大小，未计算完整
权重内容 hash。测试入口新增 workload 与所选模型路径一致性检查，并仅对
含 vision_config 的模型传图像/视频限制参数，以支持此次纯文本模型。

沿用上文复现命令，但准备和运行都显式增加
`--model /root/autodl-tmp/models/Qwen3-8B`，workload 改用本轮的
`benchmark/results/qwen3_8b_agent_20260905/workload.json`，output 使用新目录。
官方组 runtime root 为 `benchmark/results/upstream_vllm_0_25_0`、policy 为
`native`；同源码消融使用 `vllm` 和 `native`；优化组用 `vllm` 和 `prefix_aware`。
重新 prepare 时也需指定新 workload 路径。

### Qwen3-8B 单卡 Flash/Fork 对照

同源码、相同容量和 APC/Graph，Flash 使用 GPU 0，Fork 使用 GPU 1；
每形状各三次，以下为整批中位数（范围），包括冷 bootstrap。

| 共享前缀×分支 | Flash，s | Fork，s |
| --- | ---: | ---: |
| 2K×16 | 4.77（4.76–4.84） | 6.18（6.16–6.23） |
| 8K×16 | 6.82（6.81–6.84） | 8.66（8.65–8.75） |
| 16K×8 | 7.57（7.56–7.61） | 11.30（11.30–11.32） |
| 24K×16 | 12.77（12.74–12.80） | 12.69（12.67–12.76） |

未找到单卡 Fork 的稳定端到端收益。24K×16 超过现有 20,000 block visits
规划阈值，按代码会回退 Flash，不能解释为 Fork 算子收益；本轮未重新采集
Torch kernel trace，不把上一模型的 trace 当成本模型的执行证据。单卡没有换卡
复测，保留 GPU/主机噪声限制。复现时使用 `--gpus 0 --repeats 3`，backend
分别设为 `FLASH_ATTN`/`FORK_ATTN`，policy 为 `native`，cases 使用上表
对应的 `shape_p2048_b16,shape_p8192_b16,shape_p16384_b8,shape_p24576_b16`。

DP 72 次加单卡 24 次，共 96 次成功回放；原始数据包含全部运行，没有筛选最快值。
