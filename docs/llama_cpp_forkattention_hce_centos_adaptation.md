# llama.cpp ForkAttention OS Adaptation: Huawei Cloud EulerOS 2.0 vs. CentOS 7

## 1. Scope

This document records the operating-system-level work required to build and run the Qwen3 ForkAttention implementation in `llama.cpp` on the same NVIDIA Tesla T4 class of server under two operating systems:

- Huawei Cloud EulerOS (HCE) 2.0
- CentOS Linux 7.9.2009

The comparison is intentionally limited to the operating system, package repositories, compiler toolchain, build utilities, locale, and shell environment. The CUDA/ForkAttention implementation itself is shared by both systems and did not require OS-specific source-code branches.

## 2. Summary

Both systems successfully built the CUDA backend for Turing (`sm_75`) and ran Qwen3 ForkAttention on a Tesla T4 with CUDA 11.4. HCE 2.0 was the simpler environment because its normal repositories supplied a sufficiently recent compiler and CMake. CentOS 7 required additional work because its base toolchain is too old and several CentOS 7 repository endpoints have reached end of life.

| Area | Huawei Cloud EulerOS 2.0 | CentOS Linux 7.9 | Practical impact |
|---|---|---|---|
| Package manager | `dnf` | `yum` | Installation commands differ. |
| Repository condition | Active HCE repositories | CentOS 7 is EOL; some mirror lists are no longer usable | CentOS SCL repositories had to be redirected to fixed Huawei Cloud URLs. |
| Default GCC used for the build | GCC 10.3 | System GCC 4.8.5 is too old; SCL GCC 10.2.1 was installed | CentOS builds must enable `devtoolset-10` in every new shell. |
| CMake | CMake 3.22 from the OS repository | EPEL CMake 3.17.5 is below the project minimum; CMake 3.26.4 was installed with pip | CentOS must put `/root/.local/bin` on `PATH`. |
| Ninja | Ninja 1.8 from the OS repository | Ninja 1.10.2 from EPEL | No source-level difference. |
| Python/pip | Not required for the build toolchain | Python 3.6.8; pip upgraded from 9.0.3 to 21.3.1 | The last pip version compatible with Python 3.6 was used. |
| glibc | Provided by the HCE image | glibc 2.17 | The CentOS CMake wheel must be compatible with manylinux2014/glibc 2.17. |
| Locale | No blocking issue encountered | `C.UTF-8` is unavailable | CentOS commands use `LC_ALL=C` and `LANG=C`. |
| CUDA location | CUDA 11.4 under `/usr/local/cuda-11.4`; tools may not be on `PATH` | CUDA 11.4 under `/usr/local/cuda-11.4`, with `/usr/local/cuda` available | Both systems require explicit CUDA environment variables when the image does not configure them. |
| GPU target | Tesla T4, `sm_75` | Tesla T4, `sm_75` | The CMake CUDA architecture is identical. |

## 3. Common Hardware and CUDA Requirements

The OS migration did not change the GPU-side requirements:

- GPU: NVIDIA Tesla T4
- CUDA toolkit: 11.4
- CUDA architecture: 75 (`sm_75`)
- Qwen3 KV cache: FP16 for the Turing ForkAttention path
- Build generator: Ninja
- Native CPU tuning disabled for portable server binaries

The common CMake options are:

```bash
-DGGML_CUDA=ON
-DGGML_CUDA_FA=ON
-DCMAKE_CUDA_ARCHITECTURES=75
-DGGML_NATIVE=OFF
-DCMAKE_BUILD_TYPE=Release
```

The resulting CentOS CUDA library was inspected with `cuobjdump` and contained `sm_75` cubins, confirming that the binary was built for the T4 rather than relying on an unrelated architecture.

## 4. Huawei Cloud EulerOS 2.0 Adaptation

### 4.1 Package installation

HCE uses `dnf`, not Debian/Ubuntu `apt`. Its repositories provided all required build tools at adequate versions:

```bash
dnf install -y gcc gcc-c++ cmake ninja-build
```

The tested toolchain was:

```text
GCC:    10.3
CMake:  3.22
Ninja:  1.8
CUDA:   11.4
```

No Software Collections environment or user-installed CMake was required.

### 4.2 CUDA environment

The stock image had CUDA under `/usr/local/cuda-11.4`, but CUDA tools were not necessarily present on the default shell path. The following environment was sufficient:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

### 4.3 Build

```bash
cmake -S . -B build-cuda-t4 -G Ninja \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-cuda-t4 -j
```

HCE required only package-manager and CUDA-path adjustments. There was no OS-specific compiler workaround.

## 5. CentOS Linux 7.9 Adaptation

### 5.1 Base-system limitations

The tested CentOS environment reported:

```text
CentOS Linux release 7.9.2009 (Core)
Kernel:  3.10.0-1160.92.1.el7.x86_64
glibc:   2.17
GCC:     4.8.5 (system default)
Python:  3.6.8
pip:     9.0.3 (initial version)
CUDA:    11.4.152
```

The system GCC 4.8.5 cannot compile the current `llama.cpp` codebase. The available EPEL CMake 3.17.5 is also below the project's minimum required CMake version.

### 5.2 Repository adaptation

The Base, Updates, and Extras repositories used Huawei Cloud mirrors. EPEL and Software Collections repository definitions are installed with:

```bash
yum install -y epel-release centos-release-scl
```

Installing `centos-release-scl` added SCL repository definitions that still referenced the retired CentOS 7 mirror-list service. Those definitions were changed to fixed Huawei Cloud repository URLs:

```text
https://repo.huaweicloud.com/centos/7/sclo/$basearch/rh/
https://repo.huaweicloud.com/centos/7/sclo/$basearch/sclo/
```

The affected files were:

```text
/etc/yum.repos.d/CentOS-SCLo-scl-rh.repo
/etc/yum.repos.d/CentOS-SCLo-scl.repo
```

After changing the `baseurl` entries and disabling the obsolete `mirrorlist` entries, the metadata was refreshed:

```bash
yum clean metadata
yum makecache fast
```

This repository repair is specific to the CentOS 7 EOL environment; it was not needed on HCE 2.0.

### 5.3 Compiler and Ninja installation

The modern compiler was installed from Software Collections:

```bash
yum install -y \
  devtoolset-10-gcc \
  devtoolset-10-gcc-c++ \
  devtoolset-10-binutils \
  ninja-build
```

The tested versions were GCC/G++ 10.2.1 and Ninja 1.10.2. Unlike a normal system compiler installation, the SCL compiler must be enabled in every new shell:

```bash
source /opt/rh/devtoolset-10/enable
```

Without this command, CMake can select the original GCC 4.8.5 and the build will fail.

### 5.4 pip and CMake installation

The initial pip 9.0.3 did not correctly select the available CMake wheel and attempted an unsuitable source installation. pip was therefore upgraded to the final release supporting Python 3.6, and CMake was installed from the Tsinghua PyPI mirror:

```bash
python3 -m pip install --user --upgrade \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  'pip==21.3.1'

/root/.local/bin/pip3 install --user \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  'cmake==3.26.4'
```

Because this is a user-level installation, `/root/.local/bin` must be added to `PATH` before invoking CMake.

### 5.5 Locale and build environment

CentOS 7 did not provide the `C.UTF-8` locale inherited by the SSH client. This generated locale warnings before the remote shell environment was corrected. The build itself used the portable `C` locale:

```bash
export LC_ALL=C
export LANG=C
```

The complete shell initialization used for the CentOS build was:

```bash
export LC_ALL=C
export LANG=C

source /opt/rh/devtoolset-10/enable

export PATH=/root/.local/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

### 5.6 Build

```bash
cd /root/llama.cpp-fork-attn

CC=gcc CXX=g++ cmake -S . -B build-cuda-t4 -G Ninja \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-cuda-t4 \
  --target test-backend-ops llama-parallel llama-cli \
  -j16
```

The CUDA host compiler selected by CMake was GCC 10.2.1 from `devtoolset-10`.

## 6. Validation on CentOS 7

The transferred model was:

```text
/root/Qwen3-0.6B-UD-IQ1_S.gguf
```

Its SHA-256 checksum matched the local source file:

```text
fcb165efedaee2cfbdefe02bd3bbf22c80cfdb728915fbbe54fa809a8556710a
```

### 6.1 CUDA backend correctness

```bash
build-cuda-t4/bin/test-backend-ops test \
  -b CUDA0 \
  -o FLASH_ATTN_EXT \
  -p 'fork=1' \
  -j 4
```

Result:

```text
3/3 tests passed
Backend CUDA0: OK
```

The cases included Qwen3-compatible head dimension 128, four queries, and a long shared prefix (`n_common=1024`).

### 6.2 End-to-end Qwen3 smoke test

```bash
build-cuda-t4/bin/llama-parallel \
  -m /root/Qwen3-0.6B-UD-IQ1_S.gguf \
  -ngl 99 \
  -fa on \
  --fork-attn \
  -np 4 \
  -ns 4 \
  -pps \
  -n 4 \
  --temp 0 \
  -s 123 \
  -c 4096 \
  -v
```

Observed result:

```text
GPU:             Tesla T4
fork_attn:       true
parallel paths:  4
shared prefix:   273 tokens
saved KV reads:  819
cache misses:    0
```

All four branches completed generation. No CUDA error, crash, or residual GPU process was observed. This was a functional smoke test, not a performance benchmark.

## 7. Non-blocking Build Warnings

The following messages did not indicate ForkAttention failures:

- `Git not found` or an unknown build commit occurs when a source archive is copied without its `.git` directory. It affects build metadata only.
- Missing OpenSSL disables HTTPS support in HTTP-related targets but does not affect inference from a local GGUF file.
- Missing NCCL is not relevant to the tested single-GPU T4 configuration.
- CUDA 11.4 emitted several template warnings about missing return statements in unreachable or architecture-dependent template branches. The final binaries linked successfully and passed the backend tests.
- SSH may print a `C.UTF-8` warning before the remote command exports `LC_ALL=C`; this is an SSH locale-forwarding issue, not a CUDA or model error.

## 8. Source-Code Portability Conclusion

No operating-system-specific ForkAttention code was required. Both HCE 2.0 and CentOS 7 use the same:

- Qwen3 planner and runtime dispatch
- Turing FP16 ForkAttention CUDA kernel
- `sm_75` build target
- GGUF model format
- command-line interface

The complete OS-specific delta is confined to:

1. Package-manager commands (`dnf` versus `yum`).
2. CentOS 7 repository repair after upstream EOL.
3. Installation and activation of a modern GCC on CentOS.
4. Installation of a sufficiently recent CMake through pip on CentOS.
5. CentOS locale and user-level `PATH` initialization.

HCE 2.0 can use its repository-provided build stack directly, while CentOS 7 requires a maintained SCL toolchain and a user-installed CMake. Once those environment differences are resolved, the CUDA build and ForkAttention runtime behavior are the same.
