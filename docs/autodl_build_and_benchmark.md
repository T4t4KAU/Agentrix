# AutoDL 环境、构建与清理

## 当前服务器

`ssh -p 50887 root@connect.bjb2.seetacloud.com`。
推理、测试、profiling 仅在此服务器执行；本机负责编辑和传文件。

| 路径/配置 | 当前用途 |
| --- | --- |
| `/root/autodl-tmp/Agentrix` | 部署源码；当前为文件同步目录，没有父仓库 .git |
| `benchmark/.venv` | 唯一项目 Python 环境，Python 3.12.3 / Torch 2.11.0+cu128 |
| `vllm/.venv` | 指向 `../benchmark/.venv` 的兼容 symlink，不是第二套环境 |
| `/usr/local/cuda-12.8` | 当前编译 toolkit，SM120 |
| `vllm/cmake-build-cu128` | 保留的增量构建目录 |
| `/root/autodl-tmp/deps/vllm` | CMake 正在引用的依赖源码，不可当临时文件删除 |
| `/root/autodl-tmp/models` | 用户模型；当前使用 Qwen3-VL-8B-Instruct |
| `benchmark/results/upstream_vllm_0_25_0` | 独立原始 vLLM baseline，不能被 Agentrix editable install 覆盖 |
| `/root/autodl-tmp/uv-cache` | 可再下载的安装缓存，不是运行环境 |

服务器有 4 张 RTX 5090（每张约 32 GiB）。当前 matched DP=2 使用 GPUs 0/1，
不是固定占用全部 GPU。父仓库及子模块版本见 [TraceLab 来源](tracelab_timeline_replay.md#provenance)；
不要依赖生成后未更新的 vLLM version 字符串。

## 环境

从服务器仓库根目录设置：

```bash
cd /root/autodl-tmp/Agentrix
export PATH="$PWD/benchmark/.venv/bin:/usr/local/cuda-12.8/bin:/root/.local/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.8
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export UV_LINK_MODE=copy
export TORCH_CUDA_ARCH_LIST=12.0
export VLLM_USE_FLASHINFER_SAMPLER=0
```

仅在使用 Agentrix 源码时设置 `PYTHONPATH="$PWD/vllm:$PWD/LMCache"`。
原始 vLLM 对照用 harness 的 `--runtime-root` 选择，不能通过带源码路径的
`vllm` wrapper 启动。现有 wrapper 是 `benchmark/scripts/vllm_source_cli.py` 的链接。

不要为每次实验创建新的 .venv，也不要直接安装 CUDA 13 的 test lock 覆盖 cu128。
必要依赖使用 `uv pip install --python benchmark/.venv/bin/python ...`；
升级 Torch/CUDA 需要另行评估 native extension ABI，不是清理操作的一部分。

LMCache 当前缺少可选 `lmcache.cuda_ops`，沿用 Torch fallback；清理前的 tiered
实验日志已有该警告。本次只归档移除了旧 Python 3.11、旧 `c_ops` 和
`native_storage_ops` 二进制，保留现行 Python 3.12 的 common C++ 扩展。
没有为补扩展安装 CUDA 13，也没有把环境清理当作 offload 恢复故障的修复。

## 同步与增量构建

GitHub 下载慢时在本机使用代理 `127.0.0.1:7897`，再将源码传至服务器。
主路径需要 vLLM、LMCache、Mooncake；子模块 URL 以根目录 .gitmodules 为准。
保留服务器 .venv、CMake 输出、依赖、模型和 results，不从本机覆盖这些平台相关产物。
在没有 .git 的服务器部署目录中，不使用 `git checkout` 或 `git submodule update`。

C++/CUDA 变更后，复用已配置的 build tree：

```bash
cd /root/autodl-tmp/Agentrix
benchmark/.venv/bin/cmake --build vllm/cmake-build-cu128 \
  --target install --parallel 16 --verbose
```

并行数按 CPU/内存余量调整；现有配置的编译器、Python、依赖路径保存在
`CMakeCache.txt`，重配前先核对，不在同一个 build tree 混用 cu128/cu130。
CMake FetchContent 的 Triton override 应指向 Python package，不是需要 LLVM 构建的仓库根。
Python-only 变更不需要重编译 vLLM CUDA 扩展。

## 检查与实验入口

```bash
cd /root/autodl-tmp/Agentrix/benchmark
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python -m pytest tests/test_tracelab.py -q
```

当前协议、模型、容量、baseline 和完整运行命令统一见
[TraceLab 对照](tracelab_timeline_replay.md)。每次使用新的输出目录，结束后确认
进程和 GPU 已释放。CPU/Mooncake 恢复仍暂停，不要为环境 smoke 自动开启完整 offload。

## 清理与归档规则

- 保留当前环境、兼容 symlink、增量构建和被 CMake 引用的依赖。
- 保留原始 baseline、TraceLab 数据/manifest、最近有效对照及失败原始日志。
- 旧 smoke/microprofile 打包到 `/root/autodl-tmp/agentrix-archive/cleanup-20260905/`，
  保持相对目录结构；先校验归档并备份到本机，再移除散件。
- 重复安装包只有在校验本机备份 hash 后才从服务器删除。
- 下载缓存用 `uv cache clean --cache-dir <已核对的缓存目录>` 清理；
  缓存不可原样恢复，但可重新下载，不卸载已安装环境。
- 不清理用户模型、其他项目（如 InfiniCore/InfiniLM）、共享服务或用途不明的 checkout。
- 不执行针对仓库根、模型根、home 的递归删除，不用整个部署目录的盲目 rsync --delete。

完整旧文档的恢复位置见 [历史索引](historical_experiments.md)。
