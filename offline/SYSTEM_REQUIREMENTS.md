# Datus Agent 离线 Bundle — 系统依赖清单

适用于 `datus-agent-offline-linux-{x86_64,arm64}-py312.tar.gz` 离线安装包。

目标读者：拿到 bundle、准备在目标机器上部署 datus-agent 的运维同学。

---

## 1. 三道硬门槛（所有 Linux 通用）

| 门槛 | 要求 |
|---|---|
| **glibc** | ≥ **2.27** |
| **Python 3.12** | **bundle 自带**（python-build-standalone），宿主 `python3.12` 可选复用 |
| **架构** | `x86_64` 或 `aarch64`（两份 bundle 各自独立） |
| **libc 实现** | **必须是 glibc**，不支持 musl |

**不支持的环境**：

- 任何 glibc < 2.27 的发行版（CentOS 7 / RHEL 7 / Amazon Linux 1 / Amazon Linux 2 / SLES 12 等）。
- Alpine Linux / postmarketOS 等 musl-based 发行版。bundle 里是 manylinux wheel，musl libc 加载不了。

> 从本版本起，bundle 自带一套 CPython 3.12（python-build-standalone，放在 `python-runtime/` 目录），客户机器 **不再需要预装 Python 3.12**。默认 venv 模式会自动解压 bundle 里的 Python 并建 venv；若宿主已有 `python3.12`，则优先复用宿主版本。

---

## 2. 发行版匹配矩阵

| 发行版 | 版本 | glibc | 支持 | 备注 |
|---|---|---|---|---|
| **Debian** | 10 Buster | 2.28 | ✓ | EOL 2024-06 |
| | 11 Bullseye | 2.31 | ✓ | LTS 到 2026 |
| | 12 Bookworm | 2.36 | ✓ | 当前稳定 |
| | 13 Trixie | ~2.40 | ✓ | — |
| **Ubuntu** | 18.04 | 2.27 | ✓（刚好） | EOL（ESM） |
| | 20.04 | 2.31 | ✓ | LTS（ESM 到 2030） |
| | 22.04 | 2.35 | ✓ | LTS |
| | 24.04 | 2.39 | ✓ | LTS，系统自带 `python3.12` |
| **RHEL / Rocky / AlmaLinux** | 7 | 2.17 | ✗ | 不支持 |
| | 8 | 2.28 | ✓ | Rocky/Alma 8 仍在维护 |
| | 9 | 2.34 | ✓ | 当前主流 |
| | 10 | ~2.39 | ✓ | 2025 发布 |
| **CentOS Stream** | 9 / 10 | 2.34 / ~2.39 | ✓ | — |
| **Amazon Linux** | AL1 | 2.17 | ✗ | 不支持 |
| | AL2 | 2.26 | ✗ | glibc 低于门槛，不支持 |
| | AL2023 | 2.34 | ✓ | — |
| **Oracle Linux** | 7 | 2.17 | ✗ | 不支持 |
| | 8 / 9 | 2.28 / 2.34 | ✓ | 同 RHEL |
| **SUSE / openSUSE** | SLES 12 | 2.22 | ✗ | 不支持 |
| | SLES 15 / Leap 15.x | 2.31 | ✓ | — |
| **Fedora** | 38+ | ≥ 2.37 | ✓ | 滚动更新 |
| **Arch / Manjaro** | rolling | 最新 | ✓ | — |
| **Alpine** | 任意 | musl | ✗ | 未支持 |

### 两步法则

1. 跑 `ldd --version | head -1`：
   - glibc ≥ 2.27 → **支持**
   - glibc < 2.27 或 musl → **不支持**，需换系统
2. 跑 `uname -m`：
   - `x86_64` → 选 `-linux-x86_64-` 的 tar.gz
   - `aarch64` → 选 `-linux-arm64-` 的 tar.gz

---

## 3. Python 3.12 的获取方式（可选——大多数情况可跳过）

默认场景下 bundle 自带 Python 3.12，**客户不需要读这一节**。仅以下两种情况需要宿主 Python 3.12：

1. 希望用 `--user` 或 `--system` 模式把包装到宿主 Python 里（bundle 自带的 PBS 只支持 venv 模式）。
2. 希望复用 Conda / pyenv / 系统包里已有的 Python 3.12 而不解压 bundle 自带的那份（省 ~150MB 磁盘）。

跨发行版最省事的方案仍是 **Miniconda**。

| 发行版 | 系统是否直接有 3.12 | 推荐路径 |
|---|---|---|
| Ubuntu 24.04 | ✓ | `apt install python3.12 python3.12-venv` |
| Ubuntu 22.04 | 经 deadsnakes PPA | `add-apt-repository ppa:deadsnakes/ppa && apt install python3.12 python3.12-venv` |
| Debian 12 / 13 | 需 backports 或手装 | Miniconda |
| RHEL / Rocky / Alma 9 | AppStream 默认 3.11；3.12 需 EPEL 或自装 | Miniconda 或源码编译 |
| RHEL / Rocky / Alma 8 | 只有 3.6 / 3.9 | **Miniconda（推荐）** |
| Fedora 39+ | ✓ | `dnf install python3.12` |
| SUSE / openSUSE 15 | 包管理器有 3.11 | Miniconda |
| Amazon Linux 2023 | AppStream 有 3.11，3.12 需 dnf-plugins-core / 自装 | Miniconda |
| 其他 | — | Miniconda |

### Miniconda 安装模板（全发行版通用）

```bash
# x86_64
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# aarch64
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh

bash Miniconda3-latest-Linux-*.sh -b -p /opt/miniconda3
/opt/miniconda3/bin/conda create -n datus python=3.12 -y
source /opt/miniconda3/bin/activate datus

# 然后再跑 bundle 的 install 脚本（任选一种模式）：
cd datus-agent-offline-linux-x86_64-py312
./install_offline.sh --user          # 装到 ~/.local/
# 或
./install_offline.sh --system        # 装到当前激活环境（Conda env 内）
# 或
./install_offline.sh /opt/datus/.venv  # 装到独立 venv
```

> Conda 的 `python3.12` 自带一套匹配的 libc 接口，但 **bundle 里 `manylinux_2_27_x86_64` 的原生扩展在运行时仍然 dlopen 宿主 glibc**，所以即使用了 Conda，glibc ≥ 2.27 的门槛依然要满足。

---

## 4. bundle 里"强 glibc 依赖"的 wheel

这些是 pip 装完后真正在运行时会被 `dlopen` 的原生扩展——glibc 过低时 datus 起不来会报 `GLIBC_2.XX not found` 类错误。

| 包 | 版本 | glibc 最低 |
|---|---|---|
| `duckdb` | 1.3.0 | **2.27** |
| `onnxruntime` | 1.19.2 | **2.27** |
| `onnx` | 1.21.0 | **2.27** |
| `ml_dtypes` | 0.5.4 | **2.27** |
| `rapidfuzz` | 3.14.5 | **2.27** |
| `greenlet` | 3.4.0 | 2.24 |
| `pyarrow` / `numpy` / `pandas` / `cryptography` | 锁定版本 | 2.17 |
| `grpcio` / `grpcio-status` / `protobuf` | 1.73.1 / 1.71.2 / 5.29.6 | 2.17 |
| `aiohttp` / `lxml` / `pydantic-core` / `httpx-core` / 等等 | 最新兼容版 | 2.17 |

其余几百个依赖都是纯 Python 或自带 `manylinux2014` wheel，不卡门槛。决定整体门槛的是 `duckdb` / `onnxruntime` / `onnx` / `ml_dtypes` / `rapidfuzz` 这几个 **2.27** 的包。

---

## 5. 运行时还需要的系统命令

| 工具 | 用途 | 安装命令 |
|---|---|---|
| `git` | `datus-metricflow` 的 semantic adapter 运行时要调 | `apt install git` / `yum install git` / `zypper install git` / `dnf install git` |
| `tar` + `gzip` | 解压 bundle tar.gz | 发行版均自带 |
| `curl` / `wget` | 可选——用来下载 Miniconda 或 bundle | 发行版均自带 |

---

## 6. bundle **包含** / **不包含** 的东西

### 6.1 包含（无需客户自备）

| 项 | 说明 |
|---|---|
| `python-runtime/` | 一份 python-build-standalone 打包好的 CPython 3.12 tar.gz（~30MB），首次 install 时自动解压到 `python/` |
| `wheelhouse/` | datus-agent 及其所有 Python 依赖的 wheel |
| `requirements.lock` | 锁定版本清单，`pip install` 依赖它做完整一致性安装 |
| `assets/fastembed-cache/` | 预下载的 Hugging Face 模型快照（默认含 `qdrant/all-MiniLM-L6-v2-onnx`，是 `sentence-transformers/all-MiniLM-L6-v2` 的 ONNX 导出，~80MB）。`install_offline.sh` 会把它复制到 `~/.cache/huggingface/fastembed/`，避免首次跑 KB embedding 时联网拉模型。需要扩展时改 `offline/huggingface-snapshots.txt` |
| `install_offline.sh` | 一键安装脚本（venv / --user / --system 三种模式，外加 `--skip-runtime-assets` 可跳过运行时资产拷贝） |
| `manifest.json` | bundle 元数据（PBS 版本、wheel 数、HF 快照 commit、生成时间等） |

### 6.2 不包含（客户自备）

| 项 | 说明 |
|---|---|
| `git` | 参见第 5 节 |
| `~/.datus/conf/agent.yml` | LLM API key、数据库连接、代理等配置 |
| `~/.datus/benchmark/*` | 测试数据（如用 benchmark 场景） |
| 数据库连接凭证 | 客户自行提供 |
| 外网访问（LLM API） | 如使用远端 LLM，需客户机器能访问对应 endpoint |
| 宿主 Python 3.12 | 仅当客户希望用 `--user` / `--system` 时需要，参见第 3 节 |

---

## 7. 客户侧一键自检脚本

把下面脚本发给客户，让他们在目标机器跑一次，把输出贴回来即可判断是否可用：

```bash
set -e
echo "=== distro ==="
cat /etc/os-release | grep -E '^(NAME|VERSION_ID|PRETTY_NAME)=' || true
echo "=== arch ==="
uname -m
echo "=== libc ==="
if ldd --version 2>/dev/null | head -1 | grep -qi glibc; then
  ldd --version | head -1
elif ldd --version 2>&1 | head -1 | grep -qi musl; then
  echo "musl detected (Alpine?) — NOT SUPPORTED"
else
  echo "unknown libc"
fi
echo "=== python 3.12 ==="
command -v python3.12 && python3.12 --version || echo "python3.12 NOT FOUND"
echo "=== git ==="
command -v git && git --version || echo "git NOT FOUND"
```

### 判定表

| 客户环境 | 交付文件 |
|---|---|
| libc 为 musl | **不支持** |
| glibc < 2.27 | **不支持**（含 CentOS 7 / RHEL 7 / Amazon Linux 2 / SLES 12 等） |
| `x86_64` + glibc ≥ 2.27 | `datus-agent-offline-linux-x86_64-py312.tar.gz` |
| `aarch64` + glibc ≥ 2.27 | `datus-agent-offline-linux-arm64-py312.tar.gz` |
| `python3.12 NOT FOUND` | **不是问题**，bundle 自带；默认 venv 模式会自动解压 |
| `git NOT FOUND` | 让客户用系统包管理器装 git |

---

## 8. 安装模式速查

`install_offline.sh` 支持三种安装落点：

| 模式 | 命令 | 安装位置 | 使用 bundle 自带 Python? | 适合场景 |
|---|---|---|---|---|
| venv（默认） | `./install_offline.sh [/path/to/venv]` | venv 内部 | **✓ 支持**（宿主无 python3.12 时自动解压 bundle 自带的） | 测试、多用户隔离、最常用 |
| `--user` | `./install_offline.sh --user` | `~/.local/` | ✗ 需宿主 python3.12 | 单用户、无 sudo 权限 |
| `--system` | `./install_offline.sh --system` | `/usr/local/` | ✗ 需宿主 python3.12 | 专用机器、全局可用 |

Debian / Ubuntu 下 `--system` 模式会自动加 `--break-system-packages --ignore-installed`，绕开 PEP 668 和 dpkg-managed Python 包的 RECORD 缺失问题。

### Python 解析顺序

install 脚本按以下优先级找 Python 3.12：

1. 环境变量 `PYTHON_BIN`（若已设置）
2. PATH 里的 `python3.12`
3. bundle 自带的 `python-runtime/` 里的 tar.gz（首次运行自动解压到 `python/`）

若最终用的是 bundle 自带 Python，只能走 venv 模式；`--user` / `--system` 会直接报错退出。

---

## 9. 构建端常用命令（运维人员/CI 参考）

```bash
# 默认 bundle（glibc 2.27+，含自带 Python 3.12）
make offline-bundle-x86_64
make offline-bundle-arm64

# 使用 PyPI 稳定版 datus-agent（不从本地源码打本地 wheel）
make offline-bundle-x86_64 PYPI_VERSION=0.2.6
```

产物都在 `offline/dist/`：

- `datus-agent-offline-linux-x86_64-py312.tar.gz`
- `datus-agent-offline-linux-arm64-py312.tar.gz`

### 调整 bundle 自带的 Python 版本 / PBS release

构建器通过 `python-build-standalone` 下载 CPython tar.gz（默认 pinning 见
`scripts/build_offline_bundle.py` 里的 `DEFAULT_PBS_RELEASE` / `DEFAULT_PBS_PYTHON_VERSION`）。
临时覆写：

```bash
./scripts/build_offline_bundle_x86_64.sh \
    --pbs-release 20250723 \
    --pbs-python-version 3.12.11
```

下载会缓存到 `offline/.pbs-cache/`，并用 PBS 官方 `SHA256SUMS` 在线校验。

### 构建不含 Python 的"精简"bundle

若目标环境已保证有宿主 `python3.12`，可以跳过 PBS 下载/打包（节省 ~30MB）：

```bash
./scripts/build_offline_bundle_x86_64.sh --skip-python-runtime
```

这种情况下 install 脚本找不到宿主 python3.12 时会直接报错退出。

### 续跑：从中断的 build 继续，不要从零开始

下载几百个 wheel + HF 快照偶尔会被网络抖断（典型如 pip 报 `IncompleteRead`）。
这种时候**不要重跑干净 build**，加 `--resume` / `RESUME=1` 即可：脚本会保留
`offline/dist/<bundle>/wheelhouse/` 里已下好的 wheel、`.seed-wheels/` 里已编译的 seed
wheel、`assets/fastembed-cache/` 里已拉下的 HF 快照，pip 只补缺失的；HF Hub 的
snapshot_download 是基于 blob hash 的，已存在的文件直接跳过。

```bash
# 干净构建中断后续跑
make offline-bundle-arm64 RESUME=1 PYPI_VERSION=0.3.0-rc3

# 或直接调脚本
./scripts/build_offline_bundle_arm64.sh --resume --pypi-version 0.3.0-rc3
```

注意：`--resume` 假设输入（specs 文件、`--pypi-version`、镜像 index 等）和上一次
完全一致。换了任何输入，请去掉 `--resume` 重新干净构建。

### 调整或跳过预下载的运行时资产（HF 快照）

构建时默认会把 [`offline/huggingface-snapshots.txt`](huggingface-snapshots.txt) 列出的
仓库下载到 `assets/fastembed-cache/`，目前只有 `qdrant/all-MiniLM-L6-v2-onnx`（默认
embedding 模型的 ONNX 导出，~80MB）。新增模型只要往清单里加一行 `<repo_id>[@<revision>]` 就行。

跳过整个步骤（bundle 更小，但目标机首次跑 KB embedding 时必须能访问 huggingface.co）：

```bash
./scripts/build_offline_bundle_x86_64.sh --skip-runtime-assets
```

target 机上同样支持 `./install_offline.sh --skip-runtime-assets`，只跳过把资产
铺到 `~/.cache/huggingface/fastembed/` 这一步——bundle 里的拷贝仍然在，可手动复制
或用 `FASTEMBED_CACHE_PATH=$BUNDLE_DIR/assets/fastembed-cache` 直接指过去。
