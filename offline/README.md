# Datus Agent Offline Delivery

This directory contains the scripts and manifests for generating offline delivery bundles for Datus Agent.

## What gets packaged

The generated bundle includes:

- `datus-agent` built from the current `Datus-agent-public` source tree
- `datus-metricflow`
- `datus-semantic-metricflow`
- `datus-bi-superset`
- `datus-starrocks`
- `datus-hive`
- `datus-mysql`
- `datus-oceanbase`
- `datus-postgresql`
- all transitive Python dependencies as wheels

## Why Docker/manylinux

`datus-hive` pulls in dependencies such as `thrift` that may require a platform-specific Linux wheel. To make the offline bundle reproducible, the build runs inside a matching manylinux container:

- `linux-x86_64` uses `quay.io/pypa/manylinux2014_x86_64`
- `linux-arm64` uses `quay.io/pypa/manylinux2014_aarch64`

That avoids producing host-only wheels from macOS or Windows.

The bundle builder now accepts newer upstream wheel tags as well, because some
dependencies no longer publish `manylinux2014` wheels. For example, `duckdb==1.3.0`
only publishes Linux wheels tagged `manylinux_2_27` / `manylinux_2_28`, and
`lancedb==0.18.0` publishes an `aarch64` wheel tagged `manylinux_2_24`.
The resolver still prefers the oldest compatible wheel first, but the target
machine must satisfy the actual wheel tags that end up in the bundle.

## Build the offline bundles

If your packages are hosted on a private index, export it first:

```bash
export PIP_INDEX_URL="https://<your-private-index>/simple"
export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"
```

Then build:

```bash
cd /Users/lyf/GitHub/Datus-agent-public

./scripts/build_offline_bundle_x86_64.sh
./scripts/build_offline_bundle_arm64.sh
```

## Output

The generated files are written to `offline/dist/`:

- `datus-agent-offline-linux-x86_64-py312.tar.gz`
- `datus-agent-offline-linux-arm64-py312.tar.gz`

Each archive contains:

- `wheelhouse/`
- `requirements.lock`
- `install_offline.sh`
- `README.md`
- `manifest.json`

The checked-in install script templates are here:

- [install_offline_linux-x86_64.sh](/Users/lyf/GitHub/Datus-agent-public/offline/templates/install_offline_linux-x86_64.sh)
- [install_offline_linux-arm64.sh](/Users/lyf/GitHub/Datus-agent-public/offline/templates/install_offline_linux-arm64.sh)

## Install on the customer machine

Copy the matching archive to the target machine, then run:

```bash
tar -xzf datus-agent-offline-linux-x86_64-py312.tar.gz
cd datus-agent-offline-linux-x86_64-py312
./install_offline.sh /opt/datus-agent/.venv
```

If installation fails on an older Linux distribution, check the system glibc
version against the bundled wheels. With the current dependency set, `duckdb==1.3.0`
requires glibc `2.27+` on Linux.

## Files you may customize

- `offline/external-package-specs.txt`: the top-level packages to ship
- `offline/source-only-package-specs.txt`: packages that should be wheel-built inside manylinux before dependency resolution

If you want a fully pinned release, change those package names to exact version specs such as `datus-metricflow==0.1.2`.
