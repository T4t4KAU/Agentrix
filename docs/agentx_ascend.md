# 面向长上下文与分支推理的 Ascend 优化：架构、算子与内存管理

本文讨论通用大模型推理中的前缀复用、分支 attention、缓存生命周期和执行开销，
以及这些机制在 vLLM-Ascend 上的实现。目标场景包括多轮对话、共享文档问答和
多分支生成。AgentX 是其中一种评估负载，算子微基准与受控模型负载分别用于
分析局部效率和模型级效果。

方案包含三层：官方路由与 APC 提供前缀复用，混合缓存策略管理可恢复状态，
ForkAttention 与图执行改善设备计算和提交效率。各层的结果来自独立对照，
不叠加成统一加速比。框架层方案见 [跨平台优化](agentrix_cross_platform_optimizations.md)，
当前成果总表见 [技术汇报导航](README.md)。

## 实验平台与评价方法

Ascend 实验采用双 Ascend 910B2 64 GB、Qwen3.5-9B BF16、TP=1/DP=2，
运行时为 vLLM 0.22.1 与 vLLM-Ascend v0.22.1rc1 加本项目适配。
ForkAttention 只处理模型中的 full-attention 层，支持范围在后文单独说明。

早期服务级对照采用官方 [AgentX harness](https://github.com/SemiAnalysisAI/agentx-harness)
的 `inferencex-agentx-mvp`，数据集为 `semianalysisai/cc-traces-weka-062126`，
每轮 900 秒、固定一个随机种子、16 个 session trees、最大上下文 262,144 token。
按完整轨迹峰值上下文过滤，393 条轨迹中保留 220 条；保持官方 prompt、工具等待
时序和输出长度。评估指标包括输出吞吐、TTFT、缓存读取比例及完成量。

算子实验固定 attention 形状、物理 KV 和执行方式；模型实验还计入其他层、
CPU 规划与服务路径开销。Profiling 独立采集，不与正常计时混合。HBM 为板卡
采样占用；固定 KV 池内的复用改善不等于预分配显存下降。

## 请求路由与长 prefill 分块

当前 Prefix-aware DP 直接采用官方 `consistent_hash`。应用提供稳定的文档或
会话身份，使相关请求访问同一副本的 APC；两个副本各自持有 KV，不跨卡共享。
在双卡 LongBench 共享文档子集上，两个种子合并后的后续问题 TTFT 降低约
55.5%，本地 prompt 计算量减少约 28.6%；首问变慢，整体耗时的改善方向因种子
而异。完整条件见 [文档 QA 路由对照](dp_routing.md)。

长 prefill 分块沿用框架调度能力。Qwen3.5 在该 Ascend 组合下的混合缓存管理块
为 1,024 token，非零分块上限与总 batch token budget 必须允许至少一个完整块
推进。早期 AgentX 固定会话粘性路由与总 budget=2,048，仅调整分块上限：

| 长 prefill 上限 | 输出吞吐 | 平均 TTFT | P90 TTFT | 缓存读取比例 |
| --- | ---: | ---: | ---: | ---: |
| 不限制 | 41.35 token/s | 6.47 s | 13.04 s | 78.78% |
| 1,024 token | 43.16 token/s | 4.02 s | 7.92 s | 84.56% |

输出吞吐增加 4.4%，平均 TTFT 降低 37.8%。每组仅一次测量，不能视为通用最优
配置；该历史实验使用当时的会话代理，不作为后来替换为官方 router 的增量结果。

## 混合缓存的选择性保留

Qwen3.5 的可复用前缀同时依赖 full-attention KV 和 GDN 状态。即使 attention
页仍在，缺少对应位置的状态也会使混合缓存命中回退。原 `align` 路径在频繁
prefill 分块后保留大量中间状态，可能挤占更有复用价值的缓存。

当前策略保留输入末尾、已发现的共享前缀边界和 decode 完整块，并按可配置间隔
保留中间检查点。整块输入的重放与追加可能需要不同状态位置，两者分别保留。
共享边界来自本副本的实际前缀哈希命中；到达边界后保存真实计算的状态。

回收时先复用没有有效哈希的空闲块；可选策略进一步优先回收尚未被实际引用的
周期性检查点。周期性状态一旦命中便恢复普通 LRU 待遇。活跃引用、复制源和
同一步计算屏障仍由原引擎保护，缓存池总大小不变。

| 配置 | 含义 |
| --- | --- |
| `interval: null` | 沿用原保留和回收策略 |
| `interval: 0` | 仅保留已知复用边界及 decode 完整块 |
| `interval: 8192` | 额外保留每 8,192 token 的检查点，须与管理块对齐 |
| `prefer_reuse_boundaries: true` | 压力下优先回收未复用的周期性检查点；默认关闭 |

稀疏保留、共享边界和未缓存块优先复用已有上游来源，分别见
[间隔保留](https://github.com/vllm-project/vllm/pull/43447)、
[Mamba 保留](https://github.com/vllm-project/vllm/pull/45845) 和
[共享前缀检查点](https://github.com/vllm-project/vllm/pull/47782)。
本项目针对固定旧版 Ascend 运行时适配这些机制，另加入复用后的晋升策略，
不将其描述为全新缓存算法。

### 同容量下的缓存收益

官方 AgentX 对照固定每卡 36 GiB KV、Decode ACLGraph、`npugraph_ex`、
相同线程亲和配置、会话粘性路由、prefill cap=1,024 和 batch budget=2,048：

| 缓存策略 | 输出吞吐 | 平均 TTFT | P90 TTFT | 缓存读取比例 |
| --- | ---: | ---: | ---: | ---: |
| 原生保留与回收 | 78.91 token/s | 4.968 s | 15.694 s | 80.94% |
| 8,192 间隔 + 复用边界优先 | 90.35 token/s | 1.821 s | 3.518 s | 93.00% |

输出吞吐提高 14.49%，平均 TTFT 降低 63.34%。这是稀疏保留、空闲块回收和
边界保护的组合结果，不能单独归因于某一项。每组一个种子、一次测量，闭环
完成请求组合也会变化；尚未建立重复稳定性，不声称减少预分配 HBM。

实现位于 `vllm_ascend/core/mamba_retention.py` 及对应 manager/scheduler 适配。
当前限于 vLLM 0.22.1、Qwen3.5 的 `align` APC；不与 KV connector、
speculative decoding 或 context parallelism 组合。这与后文 CPU 分层缓存实验
是不同配置，不能将两个实验的收益相加。

## Decode ACLGraph 与编译执行

方案复用官方 ACLGraph 和 GDN 图参数更新能力，以 `VLLM_COMPILE`、
`FULL_DECODE_ONLY`、捕获大小 `[1,2,4,8]` 运行 decode；prefill 和混合批次
沿用相应非图路径。收益包含必要的编译与融合，不能全部归因于图回放。

两组固定每卡 36 GiB KV、相同 8,192-token 保留策略、路由和调度配置，关闭
`npugraph_ex`。按 eager、graph、graph、eager 顺序独立运行官方 AgentX，
每种方式两轮、同一个种子。下表为两轮指标的算术均值：

| 指标 | eager | Decode ACLGraph |
| --- | ---: | ---: |
| 输出吞吐 | 48.28 token/s | 88.97 token/s |
| 平均 TTFT | 2.252 s | 1.941 s |
| 每轮 P90 TTFT 的均值 | 4.052 s | 3.694 s |
| 每用户平均生成速度 | 8.67 token/s | 33.88 token/s |

输出吞吐提高 **84.3%**，平均 TTFT 降低 **13.8%**。P90 列为各轮分位数的
平均，不是合并请求后重新计算的 P90。图模式每卡额外占用约 0.55 GiB 图内存。
结论限于该 256K 过滤负载，不代表所有长上下文工作流。

在另一组固定线程亲和配置的单次对照中，启用官方 `npugraph_ex` 的普通 FX
优化后，输出吞吐 **88.87 → 90.35 token/s**，平均 TTFT **2.005 → 1.821 s**。
未启用 static kernel 或 super kernel；该项尚无重复稳定性结论。这里的候选
与缓存对照复用同一轮测量，不是独立重复。

设备执行间隙的 profiling 分析见下文；线程亲和配置本身未建立独立性能收益。

## Ascend NPU 算子适配设计

本节说明当前 **FIA 版 ForkAttention** 如何在 Ascend 上执行。应用提供会话与分支，
官方 router 决定请求落在哪张卡，APC 负责缓存复用；算子在某个实际 decode 批次中
识别共享物理 KV，把多个分支对相同前缀的注意力计算组织成 FIA 任务。两张 DP 卡
各自执行，不在卡间共享 KV，也不因启用算子而改变请求顺序。

### Ascend 910B 的计算架构

本节以实验使用的 **Ascend 910B2、Atlas A2 架构**为对象。Host CPU 运行调度器、
共享页规划和算子提交；NPU 上的多个计算核执行矩阵、向量与数据搬运任务。
理解算子适配，需要同时看计算单元的组织、数据放在哪里，以及单元之间如何交接数据。

A2 采用 **Cube/Vector 分离架构**：矩阵核 AIC 与向量核 AIV 独立执行，
对应架构的一组计算资源按 **1 个 AIC 配 2 个 AIV** 组织。各核拥有自己的 Scalar
控制单元，负责地址计算、循环及指令发射。AIC 中的 Cube 负责矩阵乘累加；
AIV 中的 Vector 负责向量算术、指数和归约等计算。这里的 1:2 是核类型配比，
设备实际可用核数仍须查询平台信息。参见 [Ascend C A2 架构规格](https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.html)。

以下为与 attention 相关的简化数据通路，省略了指令缓存、标量数据缓存、BiasTable
及部分旁路。图中每个 AIV 各有自己的 UB；箭头表示数据搬运或计算关系。

```mermaid
flowchart TB
    GM["GM 全局地址空间：权重、分页 KV、工作区<br/>设备数据驻留 HBM，访问可由片上 L2 缓存服务"]
    subgraph AIC["AIC 矩阵核"]
        SC["Scalar：独立控制与指令发射"]
        L1["L1：矩阵数据分块"] -->|MTE1| AB["L0A / L0B：矩阵输入"]
        AB --> C["Cube：矩阵乘累加"]
        C --> LC["L0C：累加结果"]
        LC --> FP["FixPipe：搬出及随路转换"]
        SC -.-> C
    end
    subgraph AIV["AIV 向量核 × 2：各自独立的控制与存储"]
        SV["Scalar：独立控制与指令发射"]
        UB["UB：向量输入、输出与临时量"] --> V["Vector：向量计算与归约"]
        V --> UB
        SV -.-> V
    end
    GM -->|MTE2| L1
    FP --> GM
    GM -->|MTE2| UB
    UB -->|MTE3| GM
```

在这代架构上，AIC 与 AIV 通过 GM 交换数据，UB 不能作为两类核任意直读直写的
共同缓冲。GM 是编程模型中的全局存储空间；经 GM 交换的数据可能命中 L2，
因此一次 GM 访问也不等于一次 HBM 访问。矩阵输出由 FixPipe 搬出，向量输出
由 MTE3 搬出。上述通路见 [CANN 计算、存储与搬运单元说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/opdevg/Ascendcopdevg/atlas_ascendc_10_0009.html)。

### 存储层级决定能复用多大的数据块

HBM 保存模型权重、跨 decode 步保留的 KV 和设备工作区；L2 缓存全局访存数据；
L1、L0 与 UB 则是执行当前数据块使用的片上缓冲。算子通过编译器或显式搬运
管理这些局部缓冲，不能把 L1 Buffer 当成自动保存整个会话 KV 的缓存。

| 存储 | A2 架构容量与归属 | attention 中的作用 |
| --- | --- | --- |
| HBM / GM | 本轮每卡 64 GB HBM；GM 为全局地址空间 | 保存分页 KV、Q、最终输出及全局中间结果 |
| L2 Cache | 核外片上缓存，容量取决于具体设备 | 为多个核的全局访问提供缓存；命中率受工作集与访问顺序影响 |
| L1 Buffer | 每个 AIC 512 KiB | 暂存矩阵输入分块，供后续搬入 L0 |
| L0A / L0B | 每个 AIC 各 64 KiB | 分别提供 Cube 左、右矩阵输入 |
| L0C | 每个 AIC 128 KiB | 保存矩阵乘累加结果 |
| Unified Buffer（UB） | 每个 AIV 192 KiB | 保存向量计算输入、归约临时量及输出 |

片上容量来自 [A2 架构规格的存储单元表](https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.html)，
表示硬件容量；编译器和运行库预留会减少实际可用空间，tiling 应以目标平台查询
结果为准。矩阵输入还需要满足内部分形布局及搬运对齐；FIA 接收 TND 或分页
接口布局后，由内部实现处理这些要求，调用方的张量视图不等于 Cube 的内部布局。

以本文的 BF16、4 个 KV heads、head dimension 256 为例，单个 full-attention
层的 32K 前缀仅 K、V 就需要：

```text
2（K 和 V）× 32768（token）× 4（KV heads）× 256（维度）× 2（字节）
= 128 MiB
```

这是根据形状计算的存储量，远大于单核局部缓冲。attention 必须分块搬入并计算。
APC 已让分支引用同一份 HBM 页；ForkAttention 进一步把同一分块对应的多个
分支 query 组织到一起，争取在计算过程中复用已搬入的数据。前者减少重复存储，
后者面向重复读取与执行效率，两者的计量方式不同。

### 异步流水如何影响 attention 的执行

Scalar 将计算和搬运指令发给不同队列，Cube、Vector、MTE 可以异步推进。
通过双缓冲，可在计算当前块时搬入下一块，但这会消耗额外片上空间，且必须保证
消费者读完后才能覆盖缓冲。核内队列依赖与 AIC/AIV 之间的交接都需要同步。
Ascend C 用队列和事件封装这些关系，见 [CANN 异步执行与同步模型](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0034.html)。

attention 的 `QKᵀ` 和 `P·V` 适合矩阵单元，softmax 的最大值、指数、求和及
归一化适合向量单元。二者交替执行，性能取决于矩阵块形状、搬运是否及时、
向量归约是否跟得上以及交接等待。具体 FIA 分支的核分工由 CANN 和输入形状
决定，不能仅凭这个数学分解就断定所有 decode 形状都按同一条流水执行。

由上述结构可以解释当前 ForkAttention 的三个设计取舍：

| 架构约束 | 当前设计 | 需要同时衡量的成本 |
| --- | --- | --- |
| 单 token query 较窄，多分支分别处理可能重复搬入共享 KV | 把同一共享段的多个 query 打包给 FIA，扩大可复用的查询集合 | Q 打包成本；实际复用仍取决于 FIA 的 tiling |
| 长 KV 必须分块，小批次可独立调度的任务有限 | 将公共前缀切成多个独立分段，为 FIA 提供更多并行任务 | 增加部分输出、LSE 和合并工作；分段数不是启动核数 |
| 分段结果经全局工作区交接，向量归约也消耗带宽 | 以一个辅助内核完成稳定加权合并，并复用固定缓冲 | 工作区 HBM 占用、读写量和辅助内核耗时 |

这些是依据架构作出的设计解释，不代表已直接调优 FIA 的内部流水。当前实现
控制分组、分段、Q 打包、结果合并与工作区；Cube tile、内部布局、MTE 双缓冲
和 AIC/AIV 同步由官方 FIA 负责。ACLGraph 则处理 Host 提交与设备任务依赖，
不能替代内核内部同步，也不会自动消除 KV 搬运。

### 模型适配与实现分工

本轮 Qwen3.5-9B 的 full-attention 形状为 16 个 Q heads、4 个 KV heads、
head dimension 256。32 层中只有 8 层 full attention 进入这条路径；GDN 状态更新
继续走原有算子。因此，单层 attention 的加速比例不能直接当成整个模型的提升。

当前实现将 attention 主体交给 CANN FIA，新增部分集中在共享任务的组织：

| 工作 | 当前实现 | NPU 上的设计目的 |
| --- | --- | --- |
| 判断共享、选择分片 | CPU `ForkBatchPlanner` | 使用 runner 已有 CPU 页表，避免为规划回读设备数据 |
| 组织 Q | Triton-Ascend `pack_fork_batch` | 复制较小的查询，为同一 KV 分段提供多个分支的 query 行 |
| QK、softmax、PV | 原生 paged FIA | 复用官方 attention 的计算、搬运与内部切分能力 |
| 合并分段输出 | Triton-Ascend `merge_fork_batch` | 按每个 query/head 做 FP32 稳定归约，一次写回最终结果 |
| 重复 decode 提交 | 官方 ACLGraph 与 graph-task update | 保持图结构和缓冲地址固定，更新本轮长度及页映射 |

这里没有自行控制 FIA 内部的 Cube tile 或 MTE 流水。优化目标是让同一共享段
服务多个 query，并通过分片增加独立任务；实际片上复用程度与并行效率仍由 FIA
实现及输入形状决定，不能声称共享 KV 只从 HBM 读取一次。

### 数据布局与共享准入

在线路径采用以下数据约定，`B` 是本卡实际 decode 请求数，`P` 是缓存页数：

| 数据 | 形状或含义 |
| --- | --- |
| 原始 Q | `[B, 16, 256]`，每个请求本步只有一个 query token |
| 分页 K、V | 逻辑视图为 `[P, 128, 4, 256]`，FIA 接口使用连续的 `[P, 128, 1024]` 视图 |
| `block_table` | int32 内核页编号；每一行描述一个 KV 分段，物理页无需相邻 |
| 打包后的 Q | `[T, 16, 256]`，`T` 为各分段的 query 行数之和 |
| `actual_seq_lengths` | TND 布局的累计 query 行结束位置，不是 KV 长度 |
| `actual_seq_lengths_kv` | 每个分段的有效 KV token 数，限制末页的可见范围 |
| 分段结果及 LSE | `[T, 16, 256]` 和 FP32 `[T, 16, 1]` |

混合缓存管理块与 attention 内核页是不同粒度。例如 1,024-token 管理块通过
runner 的块表映射为 128-token 内核页；算子读取映射后的页表，不重新分配或
复制一份共享 KV。FIA 的 TND、分页和 LSE 接口说明见
[官方 FIA API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score.md)。
该链接用于解释接口语义，本文支持范围以固定实验运行时和本仓库代码为准。

`ForkBatchPlanner` 从请求起始页开始比较物理页编号。只有连续的公共前缀进入
共享段；相同文本、相同会话名或局部相同的后缀都不足以证明这里可共享。
分组按完整 128-token 页截断，并保证每个 query 保留非空私有尾部。当前新增的
token 及不足一页的部分仍由其自己的尾部处理，兄弟分支不能访问彼此的私有页。
一批可以含多个共享组和独立请求，独立请求保留完整的自身 KV 序列。

### 四分支、32K 前缀如何转换成一次 FIA 调用

假设本卡四个请求共用 32,768-token 前缀，各有 129、130、131、132 token
私有尾部。共享前缀为 256 个内核页，按当前规则分为四段，每段 8,192 token。
规划结果包含四个共享段和四个私有段：

```text
共享段 S0：query = [q0, q1, q2, q3]，KV = 公共页   0..63
共享段 S1：query = [q0, q1, q2, q3]，KV = 公共页  64..127
共享段 S2：query = [q0, q1, q2, q3]，KV = 公共页 128..191
共享段 S3：query = [q0, q1, q2, q3]，KV = 公共页 192..255
私有段 T0..T3：每段只有对应的 qi 和自己的私有页

query_ends = [4, 8, 12, 16, 17, 18, 19, 20]
kv_lengths = [8192, 8192, 8192, 8192, 129, 130, 131, 132]
```

这里的公共页编号表示前缀中的逻辑位置，实际页编号可以是碎片化的。
Q 打包产生 20 行 query；一次 FIA 调用计算八个分段，而非发出八次 Python
算子调用。每个原始 query 得到四个共享段结果和一个私有段结果。

Fork 路径使用 `input_layout="TND"`、`block_size=128`、`sparse_mode=0`
及 `softmax_lse_flag=True`。打包到同一共享段的 query 来自不同分支，不是
同一序列相邻的时间步，因此不在这些 query 行之间施加三角因果掩码。每行仅
访问已经对该分支可见的前缀或私有尾部，因果约束由分段和有效长度保证。
这也是当前准入严格限制为单 token decode、不能直接扩展到 prefill 的原因。

```mermaid
flowchart LR
    A[CPU 内核页表与长度] --> B[共享组与分段计划]
    B --> C[页描述符和查询映射]
    Q[本轮 Q] --> D[Q 打包]
    C --> D
    C --> E[paged FIA]
    K[原有分页 K 和 V] --> E
    D --> E
    E --> F[部分输出与 LSE]
    F --> G[FP32 加权合并]
    G --> O[原请求顺序的输出]
```

### 合并分段 softmax，保持注意力语义

分段 FIA 输出的是各段独立归一化的结果，不能直接求和或平均。
对某个 query/head，设第 `j` 段输出为 `O_j`，自然对数域的归一化量为 `L_j`：

```text
m   = max_j L_j
w_j = exp(L_j - m)
O   = sum_j(w_j * O_j) / sum_j(w_j)
```

该公式在数学上恢复完整 KV 集合上的 softmax 加权输出，仍需通过浮点容差和
模型输出检查。`merge_fork_batch` 按 query/head 并行，局部将部分结果转为 FP32，
完成加权归约后写回原输出精度，避免额外生成整张 FP32 中间张量。
无效分段通过映射中的 `-1` 屏蔽，图捕获的补齐行输出为零。

在线规划最多 16 个共享分片，故每个 query 最多合并 **17 段**。
独立算子接口可合并最多 33 段，在线路径的上限仍为 17 段。

### 控制 CPU 规划、搬运与工作区成本

同一 metadata builder 保存最近一次计划，每步检查 CPU 页表和页数。页拓扑
未变时只更新尾长；跨页、页重排或有效页数变化时重新规划。长度回退若没有
改变页数，可以复用拓扑并更新长度。不能仅凭请求 ID 复用旧计划。

设备端预分配 Q、部分结果、LSE、页表、查询映射及合并映射。在线最大容量为
`8 × (16 + 1) = 136` 个 query 行；页表宽度由最大上下文决定。
页表或映射变化时才从 pinned CPU 缓冲异步复制描述符，普通 token 增长复用
原有描述符。KV 始终从引擎原有分页缓存读取。

FIA workspace 在图捕获前按支持的描述符上限申请。同一 KV group 的各层和
不同图批大小共用一份 `ForkDecodeBuffers`，避免为每层、每个图各建一套池。
每张 DP 卡拥有自己的池；同池依赖模型执行串行使用，因此当前不支持 DBO
或并发复用。预留工作区本身是显存成本，不能将减少 KV 重复读取写成显存节省。

分片存在取舍：更细的分片增加并行任务，同时增加 Q 打包、部分结果与归约工作。
在线规则为共享前缀不足 64K 时四分片、达到 64K 时十六分片，且不超过页数。
这是已测形状的配置，不是自动搜索器；缩短前缀或减少分支后应重新测量收益。

### ACLGraph 中如何更新而不重新捕获

图内结构固定为 `等待事件 → pack → FIA → merge`。每层的 FIA 任务由
`ForkCapturedAttention` 保存图更新句柄；workspace 与描述符地址保持稳定，
每步更新本轮的 query/KV 长度、页表视图和输出目标。

当前 serving 顺序先排入 graph replay，再由 update stream 准备描述符并执行
`graph_task_update_begin/end`，最后记录 ExternalEvent。图中的等待点必须放在
pack 之前，使查询映射和启用状态更新完成后，设备才能读取并打包 Q；只在 FIA
之前等待不能保护先执行的 pack。该事件顺序属于本实现的运行约定。

无共享组时，FIA 切回原生参数并直接写最终输出，`Count=0` 使 pack/merge
执行空分支；同一张图可以完成 Fork→原生→Fork 切换。回退仍有两个空内核的
提交成本。关闭 Fork 配置则保持原生图路径，不能混淆两者的基线。

### 当前支持范围与启用方式

在线实现固定在 vLLM 0.22.1 的 Qwen3.5 路径，TP=1、PP=1，可使用多卡 DP。
每卡实际批次为 2～8 个单 token causal decode 请求，使用连续 BF16/FP16 张量、
16/4/256 的 heads/head-dim 配置和 128-token 内核页，默认共享门槛 32K。
滑窗、ALiBi、sinks、非零 logits soft cap、量化 KV、speculative decode、CP
及 DBO 不在此路径范围内。全局不支持的配置在启用时拒绝；不合格的 attention
批次或没有共享组时沿用原生计算。默认保持关闭。

在匹配版本的服务命令中加入以下配置：

```bash
--additional-config '{"fork_attention":{"enabled":true,"min_shared_tokens":32768,"diagnostics":true}}'
```

如果已有 `additional_config`，合并该字段并保留其他设置。对照只切换 `enabled`，
保持原生 attention、图模式、模型、KV 预算、输入与输出长度相同。
诊断同时检查 planner 的 `selected` 和 backend 的实际执行记录；CPU 生成计划
不等于 attention 内核确实采用了它。正式计时与 profiler 分开运行。

### Profiling：共享分组减少了哪些工作

独立算子剖析固定 **4 分支、32,768-token 共享前缀、128-token 私有尾部、4 分片**，
使用 Qwen3.5-9B 的 16/4/256 attention 形状、BF16 和同一份分页 KV。
三条路径分别回放五次，下表为每次回放的均值。耗时为所记录 kernel duration
的合计，包含 profiling 影响，不是完整图回放墙钟时间。

| 路径 | 每次回放的设备算子 | kernel 耗时合计（µs） | AIC GM→L1（KB/回放） |
| --- | --- | ---: | ---: |
| 原生 FIA | FIA | 279.960 | 543,232 |
| 仅分片，不合并共享 query | Q 打包 + FIA + 合并 | 244.084 | 543,488 |
| ForkAttention | Q 打包 + FIA + 合并 | 121.784 | 142,848 |

ForkAttention 相对原生路径，kernel 耗时合计降低 **56.5%**，AIC GM→L1 计数
降低 **73.7%**。仅分片路径的该搬运计数基本不变，而 ForkAttention 相对仅分片
路径的耗时进一步降低 **50.1%**。这组对照支持共享 query 分组在增加分片并行
之外减少了重复搬运；不能把全部收益归因于分片数量或更少的算子启动。

按上一节的硬件通路，GM→L1 表示矩阵核将全局数据搬入局部缓冲的流量，
其变化与“同一共享段为多个 query 服务”的设计一致。这里的 KB 保留 profiler
原始单位，不换算成 HBM 带宽，也不用于推算显存节省。L2 可服务全局访问，
因此这组计数不能证明 HBM DRAM 读取也下降相同比例。`aic_read_main_memory` 同样按 CANN 计数口径解释，不作为 HBM 实测。

同一形状的独立、关闭 profiler 的卡 1 算子计时为原生 **289.55 µs**、仅分片
**261.69 µs**、ForkAttention **126.12 µs**，趋势一致。该计时包含 Q 打包、
FIA、合并及图回放间隙，与上表的 kernel 时间合计属于不同测量口径，不能混合
求平均。完整形状矩阵见 [算子性能与对照](#算子性能与对照)。

图执行还有另一组独立的服务级诊断：缓存命中后，每卡 8 并发、每请求输出
32 token，非剖析短请求耗时约 **4.00 → 1.63 s**；对应剖析窗口内，设备 kernel
执行区间的并集占比约 **16% → 50%–57%**，每卡观测到 32 次
`aclmdlRIExecuteAsync`。这些记录与图执行减少设备任务之间空隙的解释一致；
区间并集占比不等于 Cube/Vector 硬件利用率。该结果属于启用官方图执行及相关
编译、融合路径的组合收益，与 ForkAttention 剖析独立，详见
[Decode ACLGraph 对照](#decode-aclgraph-与编译执行)。

这组设备计数不足以确定 Host/Device 重叠比例、Cube 利用率、MTE 等待占比
或内部 tile 命中率，故不据此推导完整瓶颈占比。

### 算子性能与对照

每卡扫描 2/3/4/8 分支、8K/32K/64K 共享前缀、128/1024 token 私有尾部、
1/4/8/16/32 个前缀分片，共 24 个形状、每形状 5 个候选。
三条路径共用同一份碎片化物理 KV：服务当前使用的因果 paged FIA、
ForkAttention、以及不合并共享查询的分片对照。
每个候选采用三条路径的全部 6 种执行顺序，每次计时连续回放 50 次固定 plan 图，报告中位数。
NPU event 时间包括 Q 打包、attention、结果合并及图回放间隙；不包括 CPU 规划或逐步图任务更新。

下表取尾部 128 token；32K 固定 4 分片，64K 固定 16 分片。
卡 0 使用后续确认轮，卡 1 使用完整矩阵；并非挑每行最快分片后的结果。

| 分支 / 共享前缀 | 分片 | 卡 0 FIA → Fork（µs） | 卡 1 FIA → Fork（µs） | 卡 1 仅分片（µs） |
| --- | --- | --- | --- | --- |
| 2 / 32K | 4 | 278.46 → 113.91 | 277.48 → 116.79 | 187.45 |
| 4 / 32K | 4 | 286.01 → 115.81 | 289.55 → 126.12 | 261.69 |
| 8 / 32K | 4 | 553.45 → 124.06 | 561.32 → 131.47 | 458.50 |
| 2 / 64K | 16 | 827.85 → 240.22 | 832.53 → 247.45 | 388.55 |
| 4 / 64K | 16 | 855.55 → 272.32 | 837.67 → 264.45 | 559.63 |
| 8 / 64K | 16 | 1723.64 → 292.68 | 1706.28 → 290.88 | 973.31 |

上述结果衡量独立算子效率。分片和共享查询分组都有贡献；不能把总加速全部归于减少 HBM 读取。
单个长段直接执行在小分支数下可能变慢，分片越多也不一定更快；当前在线启发式不保证其他形状最优。
单实例显式缓冲在本轮形状中约 369～373 MiB，以 FIA 保守最大 workspace 为主。
在线实现按执行流复用工作区，避免每层、每个图重复保留。

独立 profiling 的三路径耗时、搬运计数和架构解释见
[Profiling：共享分组减少了哪些工作](#profiling共享分组减少了哪些工作)。
## NPU ForkAttention 在线 decode 接入

在线路径复用 CPU 物理页计划、设备缓冲和 FIA 工作区，减少逐 token 重建与
逐层分配。独立主机诊断中，4 分支/32K 的批次计划重建约 179 µs、复用约
86 µs；8 分支/64K 分别约 260 µs、101 µs。复用仍需页身份检查和尾长更新，
这些规划时间不等于模型延迟节省。

模型级对照固定双 910B2、Qwen3.5-9B BF16、官方 `consistent_hash`、TP1/DP2、
每卡 8 GiB KV、APC/Mamba `align`、Decode ACLGraph，关闭 `npugraph_ex`。
共享前缀 32K、每卡四分支、私有尾部 128 token，每请求固定输出 64 token。
两个种子各做两组独立重启配对，反转策略顺序；每次服务取三次批次均值，
再对四组配对等权合并：

| 指标 | 原生路径 | ForkAttention |
| --- | ---: | ---: |
| 整批平均耗时 | 3.072 s | 2.898 s |
| 双卡 HBM 采样峰值均值 | 62,421 MiB | 63,181.5 MiB |

该形状平均耗时降低 **5.67%**，四组对照均更快，配对输出一致；代价是双卡
HBM 增加 **760.5 MiB**。这是受控模型执行收益，未建立容量收益。其他形状
仍有模型输出差异或收益方向不一致，不能由单层数值等价直接推出整模型等价。
详见 [模型级对照条件与边界](dp_routing.md#长前缀模型级对照)。

官方 AgentX 的早期在线对照未触发 32K 门槛下的共享路径，尚无该负载的整体
收益结论。高 APC 命中率只说明历史可复用，不保证多个分支在同卡同批次执行。
ForkAttention 保持显式启用，收益取决于共享长度、分支数量、实际批次和图模式。

## 内存管理与适用边界

ForkAttention 复用现有物理页，主要优化执行组织；共享 KV 的存储去重由 APC
承担。FIA 工作区会增加 HBM 占用，不能将其搬运计数下降写成容量节省。

框架层选择性备份则控制哪些请求值得写入 CPU 缓存。在双 Ascend 的顺序压力
负载中，每卡同为 1 GiB 设备 KV + 2 GiB CPU KV，恢复平均 TTFT 从
768.68 降至 152.13 ms；与每卡 8 GiB APC 比较，双卡 HBM 采样峰值少
14.02 GiB，代价为总计 4 GiB CPU 预算和 17.30 ms 恢复延迟增加。
增长上下文的独立数据见 [KV 生命周期与容量](kv_memory_optimization_status.md)。
该路径复用官方连接器，与前述混合状态稀疏保留及图执行采用不同实验配置。

工具等待主动迁移、恢复前预取尚未接通；选择性备份不等于按工具事件卸载。
各项优化应按负载选择，不能将所有开关同时启用视为已验证的组合方案。

实现索引：[页表与计划](../vllm-ascend/vllm_ascend/attention/fork_batch.py)、
[共享缓冲与图任务](../vllm-ascend/vllm_ascend/ops/fork_decode.py)、
[辅助内核](../vllm-ascend/vllm_ascend/ops/triton/fork_attention.py)、
[attention 后端](../vllm-ascend/vllm_ascend/attention/attention_v1.py)。
实验入口见 [Ascend 实验代码](../experiments/agentx-ascend) 和
[benchmark 说明](../benchmark/README.md)。
