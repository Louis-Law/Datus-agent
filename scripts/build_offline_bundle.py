#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import tomllib
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL_SPECS_FILE = REPO_ROOT / "offline" / "external-package-specs.txt"
DEFAULT_SOURCE_ONLY_SPECS_FILE = REPO_ROOT / "offline" / "source-only-package-specs.txt"
DEFAULT_EXTRA_CONSTRAINTS_FILE = REPO_ROOT / "offline" / "extra-constraints.txt"
DEFAULT_HF_SNAPSHOTS_FILE = REPO_ROOT / "offline" / "huggingface-snapshots.txt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "offline" / "dist"

# Bundle-relative directory for pre-fetched runtime assets (HF snapshots,
# eventually any other lazy-download fixtures). The install script mirrors
# `fastembed-cache/` into the user's fastembed cache so first-run KB ops do
# not reach out to huggingface.co.
RUNTIME_ASSETS_DIRNAME = "assets"
FASTEMBED_CACHE_SUBDIR = "fastembed-cache"

# --- python-build-standalone (PBS) runtime ---
# The bundle ships a self-contained CPython so target machines do not have to
# pre-install Python 3.12. These pins decide which upstream build gets shipped;
# bump them to pick up a newer 3.12 patch release.
#
# Release index: https://github.com/astral-sh/python-build-standalone/releases
# We verify the downloaded tarball against the release's SHA256SUMS file at
# build time, so there is no per-version hash to maintain here.
DEFAULT_PBS_RELEASE = "20250723"
DEFAULT_PBS_PYTHON_VERSION = "3.12.11"

# python-build-standalone release-asset target triples per bundle target.
# Filename pattern: cpython-<pyver>+<release>-<triple>-install_only.tar.gz
PBS_TRIPLE_MAP = {
    "linux-x86_64": "x86_64-unknown-linux-gnu",
    "linux-arm64": "aarch64-unknown-linux-gnu",
    "darwin-arm64": "aarch64-apple-darwin",
}

PBS_CACHE_DIR = REPO_ROOT / "offline" / ".pbs-cache"

LOCAL_TOP_LEVEL_PACKAGE = "datus-agent"

DEFAULT_EXTERNAL_TOP_LEVEL_PACKAGES = [
    "datus-metricflow",
    "datus-semantic-metricflow",
    "datus-bi-superset",
    "datus-starrocks",
    "datus-hive",
    "datus-mysql",
    "datus-oceanbase",
    "datus-postgresql",
]

TARGETS = {
    "linux-x86_64": {
        # Keep tags ordered from oldest to newest so pip prefers the most
        # widely compatible wheel when multiple upstream variants exist.
        "platform_tags": [
            "manylinux2014_x86_64",
            "manylinux_2_17_x86_64",
            "manylinux_2_24_x86_64",
            "manylinux_2_27_x86_64",
            "manylinux_2_28_x86_64",
            "linux_x86_64",
        ],
        "seed_platform_fragments": [
            "manylinux2014_x86_64",
            "manylinux_2_17_x86_64",
            "manylinux_2_24_x86_64",
            "manylinux_2_27_x86_64",
            "manylinux_2_28_x86_64",
            "linux_x86_64",
        ],
        "expected_uname_s": "Linux",
        "expected_uname_m": ["x86_64", "amd64"],
        "bundle_suffix": "linux-x86_64",
    },
    "linux-arm64": {
        "platform_tags": [
            "manylinux2014_aarch64",
            "manylinux_2_17_aarch64",
            "manylinux_2_24_aarch64",
            "manylinux_2_27_aarch64",
            "manylinux_2_28_aarch64",
            "linux_aarch64",
        ],
        "seed_platform_fragments": [
            "manylinux2014_aarch64",
            "manylinux_2_17_aarch64",
            "manylinux_2_24_aarch64",
            "manylinux_2_27_aarch64",
            "manylinux_2_28_aarch64",
            "linux_aarch64",
        ],
        "expected_uname_s": "Linux",
        "expected_uname_m": ["aarch64", "arm64"],
        "bundle_suffix": "linux-arm64",
    },
    "darwin-arm64": {
        # macOS wheel platform tags. `macosx_<major>_<minor>_<arch>` declares the
        # minimum macOS the wheel supports — we list 11.0 first so pip prefers
        # the most broadly compatible wheel, then escalate when only newer
        # wheels are published. `universal2` wheels run on both arm64 and
        # x86_64 so they are accepted under arm64 as well.
        "platform_tags": [
            "macosx_11_0_arm64",
            "macosx_11_0_universal2",
            "macosx_12_0_arm64",
            "macosx_13_0_arm64",
            "macosx_14_0_arm64",
            "macosx_15_0_arm64",
        ],
        "seed_platform_fragments": [
            "macosx_11_0_arm64",
            "macosx_11_0_universal2",
            "macosx_12_0_arm64",
            "macosx_13_0_arm64",
            "macosx_14_0_arm64",
            "macosx_15_0_arm64",
        ],
        "expected_uname_s": "Darwin",
        "expected_uname_m": ["arm64"],
        "bundle_suffix": "darwin-arm64",
    },
}


def normalize_dist_name(name: str, separator: str = "-") -> str:
    return re.sub(r"[-_.]+", separator, name).lower()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    location = cwd or REPO_ROOT
    print(f"+ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=location, check=True)


def can_import_module(python_executable: str, module_name: str) -> bool:
    result = subprocess.run(
        [python_executable, "-c", f"import {module_name}"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def can_run_pip(python_executable: str) -> bool:
    result = subprocess.run(
        [python_executable, "-m", "pip", "--version"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def maybe_bootstrap_pip(python_executable: str) -> bool:
    if can_run_pip(python_executable):
        return True
    if not can_import_module(python_executable, "ensurepip"):
        return False

    result = subprocess.run(
        [python_executable, "-m", "ensurepip", "--upgrade"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0 and can_run_pip(python_executable)


def candidate_python_executables() -> list[str]:
    candidates: list[str] = []

    env_override = os.environ.get("OFFLINE_BUNDLE_PIP_PYTHON")
    if env_override:
        candidates.append(env_override)

    candidates.append(sys.executable)

    if sys.prefix != sys.base_prefix:
        base_python = Path(sys.base_prefix) / "bin" / Path(sys.executable).name
        candidates.append(str(base_python))
        candidates.append(str(Path(sys.base_prefix) / "bin" / "python3"))
        candidates.append(str(Path(sys.base_prefix) / "bin" / "python"))

    path_python3 = shutil.which("python3")
    path_python = shutil.which("python")
    if path_python3:
        candidates.append(path_python3)
    if path_python:
        candidates.append(path_python)

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        resolved = str(Path(candidate).expanduser())
        if resolved in seen:
            continue
        if not Path(resolved).exists():
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)

    return unique_candidates


def resolve_pip_python() -> str:
    checked: list[str] = []
    for python_executable in candidate_python_executables():
        checked.append(python_executable)
        if maybe_bootstrap_pip(python_executable):
            return python_executable

    checked_display = "\n".join(f"- {candidate}" for candidate in checked)
    raise RuntimeError(
        "Unable to find a Python interpreter with pip support for offline bundle dependency collection.\n"
        "Set OFFLINE_BUNDLE_PIP_PYTHON to an interpreter that has pip available.\n"
        f"Checked:\n{checked_display}"
    )


def load_project_version(pyproject_path: Path) -> str:
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    return data["project"]["version"]


def load_project_dependencies(pyproject_path: Path) -> list[str]:
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    return data["project"].get("dependencies", [])


def load_specs(specs_file: Path, required: bool) -> list[str]:
    if not specs_file.exists():
        if required:
            raise FileNotFoundError(f"Specs file not found: {specs_file}")
        return []

    specs: list[str] = []
    for raw_line in specs_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        specs.append(line)

    if required and not specs:
        raise ValueError(f"No package specs found in {specs_file}")

    return specs


def extract_package_name(spec: str) -> str:
    match = re.match(r"^([A-Za-z0-9_.-]+)", spec.strip())
    if not match:
        raise ValueError(f"Unsupported package spec: {spec}")
    return normalize_dist_name(match.group(1), "-")


def normalize_requirement_constraint(spec: str) -> tuple[str, str]:
    requirement = Requirement(spec)
    package_name = normalize_dist_name(requirement.name, "-")
    constraint = package_name
    if requirement.specifier:
        constraint += str(requirement.specifier)
    if requirement.marker:
        constraint += f"; {requirement.marker}"
    return package_name, constraint


def build_base_constraint_lines(project_dependencies: list[str]) -> list[str]:
    constraints: dict[str, str] = {}
    for spec in project_dependencies:
        package_name, constraint_line = normalize_requirement_constraint(spec)
        constraints[package_name] = constraint_line
    return [constraints[package_name] for package_name in sorted(constraints)]


def merge_extra_constraint_lines(base_constraint_lines: list[str], extra_specs: list[str]) -> list[str]:
    """Append extra >= lower-bound hints to the base constraints.

    Pyproject-derived constraints always win: any extra whose package is already
    pinned by `pyproject.toml` is dropped so this file cannot loosen or override
    the runtime source of truth. The resulting list is only used as a resolver
    hint passed to `pip download --constraint`.
    """

    merged_names = {extract_package_name(line) for line in base_constraint_lines}
    merged_lines = list(base_constraint_lines)
    for spec in extra_specs:
        package_name, constraint_line = normalize_requirement_constraint(spec)
        if package_name in merged_names:
            continue
        merged_lines.append(constraint_line)
        merged_names.add(package_name)
    return merged_lines


def build_local_wheel(pip_python: str, local_dist_dir: Path) -> tuple[Path, str]:
    if local_dist_dir.exists():
        shutil.rmtree(local_dist_dir)
    local_dist_dir.mkdir(parents=True, exist_ok=True)

    project_version = load_project_version(REPO_ROOT / "pyproject.toml")
    run(
        [
            pip_python,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(local_dist_dir),
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
    )

    wheel_candidates = sorted(local_dist_dir.glob("datus_agent-*.whl"))
    if not wheel_candidates:
        raise RuntimeError(f"No datus-agent wheel was created in {local_dist_dir}")

    return wheel_candidates[0], project_version


def build_pip_download_command(
    pip_python: str,
    wheelhouse_dir: Path,
    platform_tags: list[str],
    python_version: str,
    package_specs: list[str],
    find_links_dirs: list[Path],
    constraints_file: Path | None,
    index_url: str | None,
    extra_index_urls: list[str],
) -> list[str]:
    version_no_dot = python_version.replace(".", "")
    abi = f"cp{version_no_dot}"

    cmd = [
        pip_python,
        "-m",
        "pip",
        "download",
        "--dest",
        str(wheelhouse_dir),
        "--only-binary=:all:",
        "--python-version",
        python_version,
        "--implementation",
        "cp",
        "--abi",
        abi,
    ]

    for find_links_dir in find_links_dirs:
        cmd.extend(["--find-links", str(find_links_dir)])

    for platform_tag in platform_tags:
        cmd.extend(["--platform", platform_tag])

    if constraints_file:
        cmd.extend(["--constraint", str(constraints_file)])

    if index_url:
        cmd.extend(["--index-url", index_url])
    for extra_index_url in extra_index_urls:
        cmd.extend(["--extra-index-url", extra_index_url])

    cmd.extend(package_specs)
    return cmd


def is_universal_wheel(path: Path) -> bool:
    return path.name.endswith("-none-any.whl")


def is_compatible_seed_wheel(path: Path, target: dict[str, object]) -> bool:
    if is_universal_wheel(path):
        return True

    platform_tag = path.name[:-4].rsplit("-", 1)[-1]
    return any(fragment in platform_tag for fragment in target["seed_platform_fragments"])


def build_seed_wheels(
    pip_python: str,
    seed_wheels_dir: Path,
    source_only_specs: list[str],
    target: dict[str, object],
    index_url: str | None,
    extra_index_urls: list[str],
    resume: bool = False,
) -> None:
    if not source_only_specs:
        if seed_wheels_dir.exists():
            shutil.rmtree(seed_wheels_dir)
        seed_wheels_dir.mkdir(parents=True, exist_ok=True)
        return

    if resume and seed_wheels_dir.exists():
        existing = sorted(seed_wheels_dir.glob("*.whl"))
        if existing:
            for wheel_path in existing:
                if not is_compatible_seed_wheel(wheel_path, target):
                    raise RuntimeError(
                        "Seed wheel reused from a prior --resume run is not compatible with the "
                        f"requested target: {wheel_path.name}. Re-run without --resume to rebuild."
                    )
            print(f"[resume] Reusing {len(existing)} seed wheel(s) in {seed_wheels_dir}")
            return

    if seed_wheels_dir.exists():
        shutil.rmtree(seed_wheels_dir)
    seed_wheels_dir.mkdir(parents=True, exist_ok=True)

    for spec in source_only_specs:
        seed_wheels_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            pip_python,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(seed_wheels_dir),
        ]

        if index_url:
            cmd.extend(["--index-url", index_url])
        for extra_index_url in extra_index_urls:
            cmd.extend(["--extra-index-url", extra_index_url])

        cmd.append(spec)
        print(f"Building seed wheel for {spec}")
        run(cmd, cwd=REPO_ROOT)

    built_wheels = sorted(seed_wheels_dir.glob("*.whl"))
    if not built_wheels:
        raise RuntimeError(f"No seed wheels were created from: {source_only_specs}")

    for wheel_path in built_wheels:
        if not is_compatible_seed_wheel(wheel_path, target):
            raise RuntimeError(
                "Seed wheel build produced a wheel that is not compatible with the requested target. "
                f"Build this bundle on a matching manylinux target or publish a compatible wheel first: {wheel_path.name}"
            )


def find_downloaded_version(package_name: str, wheelhouse_dir: Path) -> str:
    versions = collect_wheel_versions(wheelhouse_dir)
    normalized_name = normalize_dist_name(package_name, "-")
    if normalized_name not in versions:
        raise FileNotFoundError(f"Could not find a wheel for {package_name} in {wheelhouse_dir}")
    return versions[normalized_name]


def collect_wheel_versions(wheelhouse_dir: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for wheel_path in sorted(wheelhouse_dir.glob("*.whl")):
        parts = wheel_path.name[:-4].split("-")
        if len(parts) < 5:
            raise RuntimeError(f"Unsupported wheel filename: {wheel_path.name}")

        package_name = normalize_dist_name(parts[0], "-")
        version = parts[1]
        existing_version = versions.get(package_name)
        if existing_version and existing_version != version:
            raise RuntimeError(
                f"Expected one resolved version for {package_name}, found: {sorted({existing_version, version})}"
            )
        versions[package_name] = version

    return versions


def write_constraints_file(destination: Path, base_constraint_lines: list[str]) -> None:
    merged: dict[str, str] = {}
    for line in base_constraint_lines:
        package_name = extract_package_name(line)
        merged[package_name] = line

    destination.write_text("\n".join(merged[package_name] for package_name in sorted(merged)) + "\n", encoding="utf-8")


def build_resolved_wheel_pins(wheelhouse_dir: Path) -> list[str]:
    """Return `pkg==version` pins for every wheel currently in the wheelhouse.

    Used to progressively tighten the pip constraints across staged resolutions:
    once a stage lands a wheel, downstream stages must not backtrack past that
    version, otherwise the resolver's search space keeps growing and eventually
    trips the depth limit.
    """

    versions = collect_wheel_versions(wheelhouse_dir)
    return [f"{name}=={versions[name]}" for name in sorted(versions)]


def refresh_constraints_with_wheelhouse(
    constraints_path: Path,
    base_constraint_lines: list[str],
    wheelhouse_dir: Path,
) -> None:
    wheel_pins = build_resolved_wheel_pins(wheelhouse_dir)
    # Wheel pins go last so `write_constraints_file`'s last-wins dedupe lets
    # them override any looser pyproject/extras entry for the same package.
    write_constraints_file(constraints_path, [*base_constraint_lines, *wheel_pins])


def copy_wheels_to_directory(paths: list[Path], destination_dir: Path) -> None:
    for wheel_path in paths:
        destination = destination_dir / wheel_path.name
        if not destination.exists():
            shutil.copy2(wheel_path, destination)


def replace_wheelhouse_from_stage(stage_dir: Path, wheelhouse_dir: Path, preserved_wheels: list[Path]) -> None:
    # Preserved wheels are kept verbatim; this matters when --resume is in
    # play and `preserved_wheels` includes paths that live inside
    # `wheelhouse_dir` itself (carried over from a partial prior run).
    keep_names = {wheel_path.name for wheel_path in preserved_wheels}
    if wheelhouse_dir.exists():
        for wheel_path in wheelhouse_dir.glob("*.whl"):
            if wheel_path.name not in keep_names:
                wheel_path.unlink()
    else:
        wheelhouse_dir.mkdir(parents=True, exist_ok=True)

    copy_wheels_to_directory(preserved_wheels, wheelhouse_dir)
    copy_wheels_to_directory(sorted(stage_dir.glob("*.whl")), wheelhouse_dir)


def download_top_level_packages(
    pip_python: str,
    wheelhouse_dir: Path,
    platform_tags: list[str],
    python_version: str,
    datus_agent_spec: str,
    external_specs: list[str],
    base_constraint_lines: list[str],
    preserved_wheels: list[Path],
    index_url: str | None,
    extra_index_urls: list[str],
) -> None:
    constraints_path = wheelhouse_dir.parent / ".constraints.txt"
    stage_dir = wheelhouse_dir.parent / ".stage-wheelhouse"
    find_links_dirs = [wheelhouse_dir]
    write_constraints_file(constraints_path, base_constraint_lines)

    resolved_top_level_specs: list[str] = []
    stages: list[tuple[str, str]] = [(extract_package_name(spec), spec) for spec in external_specs]

    for stage_label, stage_spec in stages:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Always pin datus-agent alongside each stage. When building from
        # local source, `datus_agent_spec` is the freshly built wheel path so
        # pip cannot race a stale same-version wheel on PyPI. When shipping a
        # PyPI stable, it is a `datus-agent==X.Y.Z` spec that pip resolves
        # normally.
        package_specs = [*resolved_top_level_specs, stage_spec, datus_agent_spec]
        print(f"Resolving dependencies for {stage_label}: {stage_spec}")
        run(
            build_pip_download_command(
                pip_python=pip_python,
                wheelhouse_dir=stage_dir,
                platform_tags=platform_tags,
                python_version=python_version,
                package_specs=package_specs,
                find_links_dirs=find_links_dirs,
                constraints_file=constraints_path,
                index_url=index_url,
                extra_index_urls=extra_index_urls,
            ),
            cwd=REPO_ROOT,
        )

        resolved_version = find_downloaded_version(stage_label, stage_dir)
        resolved_top_level_specs.append(f"{stage_label}=={resolved_version}")
        replace_wheelhouse_from_stage(stage_dir, wheelhouse_dir, preserved_wheels)
        refresh_constraints_with_wheelhouse(constraints_path, base_constraint_lines, wheelhouse_dir)

    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    final_package_specs = [*resolved_top_level_specs, datus_agent_spec]
    stage_dir.mkdir(parents=True, exist_ok=True)
    print(f"Resolving final dependency set for datus-agent: {datus_agent_spec}")
    run(
        build_pip_download_command(
            pip_python=pip_python,
            wheelhouse_dir=stage_dir,
            platform_tags=platform_tags,
            python_version=python_version,
            package_specs=final_package_specs,
            find_links_dirs=find_links_dirs,
            constraints_file=constraints_path,
            index_url=index_url,
            extra_index_urls=extra_index_urls,
        ),
        cwd=REPO_ROOT,
    )
    replace_wheelhouse_from_stage(stage_dir, wheelhouse_dir, preserved_wheels)
    shutil.rmtree(stage_dir)


def write_requirements_lock(
    destination: Path,
    external_specs: list[str],
    local_version: str,
    wheelhouse_dir: Path,
) -> list[str]:
    lines = [f"{LOCAL_TOP_LEVEL_PACKAGE}=={local_version}"]
    seen: set[str] = set()
    for spec in external_specs:
        package_name = extract_package_name(spec)
        if package_name in seen:
            continue
        seen.add(package_name)
        version = find_downloaded_version(package_name, wheelhouse_dir)
        lines.append(f"{package_name}=={version}")

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def install_script_template_path(target_name: str) -> Path:
    return REPO_ROOT / "offline" / "templates" / f"install_offline_{target_name}.sh"


def pbs_filename(pbs_release: str, pbs_python_version: str, target_name: str) -> str:
    triple = PBS_TRIPLE_MAP[target_name]
    return f"cpython-{pbs_python_version}+{pbs_release}-{triple}-install_only.tar.gz"


def _http_get(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def download_pbs_runtime(
    pbs_release: str,
    pbs_python_version: str,
    target_name: str,
    cache_dir: Path,
) -> Path:
    """Download (or reuse cached) python-build-standalone tarball for `target_name`.

    Verified against the release's published SHA256SUMS. Returns the path to
    the cached tarball, ready to be copied into a bundle's python-runtime/ dir.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = pbs_filename(pbs_release, pbs_python_version, target_name)
    cached = cache_dir / filename

    base_url = f"https://github.com/astral-sh/python-build-standalone/releases/download/{pbs_release}"
    sums_url = f"{base_url}/SHA256SUMS"
    print(f"Fetching PBS SHA256SUMS from {sums_url}")
    sums_text = _http_get(sums_url).decode("utf-8")

    expected_hash: str | None = None
    for line in sums_text.splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and parts[1] == filename:
            expected_hash = parts[0]
            break
    if not expected_hash:
        raise RuntimeError(
            f"SHA256SUMS for PBS release {pbs_release} does not list {filename}. "
            "Check --pbs-release / --pbs-python-version or update "
            "DEFAULT_PBS_RELEASE / DEFAULT_PBS_PYTHON_VERSION."
        )

    if cached.exists():
        current = hashlib.sha256(cached.read_bytes()).hexdigest()
        if current == expected_hash:
            print(f"Using cached PBS tarball: {cached.name}")
            return cached
        print(f"Cached {cached.name} hash mismatch; re-downloading")
        cached.unlink()

    url = f"{base_url}/{filename}"
    print(f"Downloading PBS runtime: {url}")
    tmp = cached.with_suffix(cached.suffix + ".tmp")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)

    actual_hash = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        tmp.unlink()
        raise RuntimeError(f"SHA256 mismatch for {filename}: expected {expected_hash}, got {actual_hash}")
    tmp.rename(cached)
    return cached


def download_huggingface_snapshots(
    snapshot_specs: list[str],
    cache_dir: Path,
) -> list[dict[str, str]]:
    """Pre-download HF snapshots into ``cache_dir`` using the standard HF cache layout.

    Each spec is ``<repo_id>`` or ``<repo_id>@<revision>``. We rely on
    ``huggingface_hub.snapshot_download`` so the resulting directory is a
    drop-in for ``$HF_HOME/fastembed`` (or ``$FASTEMBED_CACHE_PATH``) on the
    target machine. Returns a list of ``{repo_id, revision, commit}`` records
    for ``manifest.json``.
    """

    if not snapshot_specs:
        return []

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars
    except ImportError as exc:  # pragma: no cover - build-host concern
        raise RuntimeError(
            "huggingface_hub must be installed in the build environment to "
            "pre-download offline runtime assets. Run `uv sync` (or "
            "`pip install huggingface_hub`) before invoking the bundle "
            "builder."
        ) from exc

    disable_progress_bars()

    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []

    for spec in snapshot_specs:
        if "@" in spec:
            repo_id, revision = spec.split("@", 1)
        else:
            repo_id, revision = spec, None

        kwargs: dict[str, object] = {"repo_id": repo_id, "cache_dir": str(cache_dir)}
        if revision:
            kwargs["revision"] = revision

        print(f"Downloading HF snapshot {repo_id}" + (f"@{revision}" if revision else ""))
        snapshot_path = snapshot_download(**kwargs)

        commit = Path(snapshot_path).name  # snapshots/<commit_hash>
        records.append(
            {
                "repo_id": repo_id,
                "revision": revision or "",
                "commit": commit,
            }
        )

    return records


def render_bundle_readme(
    bundle_name: str,
    python_version: str,
    python_runtime: dict | None,
    runtime_assets: list[dict[str, str]] | None = None,
    target_name: str = "linux-x86_64",
) -> str:
    if python_runtime:
        runtime_block = textwrap.dedent(
            f"""\
            ## Bundled Python runtime
            This bundle ships a self-contained CPython {python_runtime["python_version"]}
            (python-build-standalone release {python_runtime["release"]}) under
            `python-runtime/`. `install_offline.sh` will reuse a host `python3.12`
            when available, and otherwise extract the bundled runtime to
            `python/` and install into a virtualenv built from it.
            """
        )
    else:
        runtime_block = textwrap.dedent(
            """\
            ## Python runtime
            This bundle does NOT ship a Python runtime. The target machine must
            provide `python3.12` on PATH before running `install_offline.sh`.
            """
        )

    if runtime_assets:
        asset_lines = "\n".join(
            f"          - `{record['repo_id']}` @ `{record['commit']}`" for record in runtime_assets
        )
        assets_block = (
            textwrap.dedent(
                """\
                ## Pre-fetched runtime assets
                The bundle ships the Hugging Face snapshots Datus would otherwise
                lazy-download on first use. `install_offline.sh` mirrors them into
                `~/.cache/huggingface/fastembed/` so the embedding model never
                touches the network. Override the destination by exporting
                `FASTEMBED_CACHE_PATH` (or `HF_HOME`) before running Datus, or
                pass `--skip-runtime-assets` to `install_offline.sh` to leave the
                user cache alone.

                Snapshots included:
                """
            )
            + asset_lines
            + "\n"
        )
    else:
        assets_block = ""

    if target_name.startswith("darwin-"):
        arch_label = target_name.split("-", 1)[1]
        prereq_block = textwrap.dedent(
            f"""\
            ## Prerequisites
            - macOS target machine (Darwin), version 11 (Big Sur) or newer
            - Apple Silicon CPU; the bundle's arch must match (`{arch_label}`)
            - `venv` module available in the Python installation (PBS runtime includes it)

            """
        )
    else:
        prereq_block = textwrap.dedent(
            """\
            ## Prerequisites
            - Linux target machine, glibc 2.27+
            - Architecture matches the bundle (x86_64 or aarch64)
            - `venv` module available in the Python installation (PBS runtime includes it)

            """
        )

    return (
        textwrap.dedent(
            f"""\
        # Datus Agent Offline Bundle

        This bundle was generated for `{bundle_name}` and Python `{python_version}`.

        ## Included top-level packages
        - datus-agent
        - datus-metricflow
        - datus-semantic-metricflow
        - datus-bi-superset
        - datus-starrocks
        - datus-hive
        - datus-mysql
        - datus-oceanbase
        - datus-postgresql

        ## Install
        ```bash
        tar -xzf {bundle_name}.tar.gz
        cd {bundle_name}
        ./install_offline.sh /opt/datus-agent/.venv
        ```

        """
        )
        + prereq_block
        + runtime_block
        + assets_block
    )


def create_bundle(
    output_root: Path,
    target_name: str,
    python_version: str,
    external_specs: list[str],
    source_only_specs: list[str],
    extra_constraint_specs: list[str],
    huggingface_snapshots: list[str],
    pypi_version: str | None,
    index_url: str | None,
    extra_index_urls: list[str],
    include_python_runtime: bool,
    pbs_release: str,
    pbs_python_version: str,
    resume: bool = False,
) -> Path:
    pip_python = resolve_pip_python()
    target = TARGETS[target_name]
    bundle_name = f"datus-agent-offline-{target['bundle_suffix']}-py{python_version.replace('.', '')}"
    bundle_dir = output_root / bundle_name
    archive_path = output_root / f"{bundle_name}.tar.gz"

    if resume:
        if not bundle_dir.exists():
            print(f"[resume] {bundle_dir} does not exist; nothing to resume — falling back to a clean build")
        else:
            print(f"[resume] Reusing {bundle_dir}")
    else:
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        if archive_path.exists():
            archive_path.unlink()

    wheelhouse_dir = bundle_dir / "wheelhouse"
    local_dist_dir = bundle_dir / ".local-dist"
    seed_wheels_dir = bundle_dir / ".seed-wheels"
    wheelhouse_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot wheels already in the wheelhouse so the staged resolver does not
    # wipe them between stages. Empty on a clean build.
    pre_existing_wheels: list[Path] = []
    if resume:
        pre_existing_wheels = sorted(wheelhouse_dir.glob("*.whl"))
        if pre_existing_wheels:
            print(
                f"[resume] Found {len(pre_existing_wheels)} wheel(s) in {wheelhouse_dir}; will preserve across stages"
            )

    platform_tags = list(target["platform_tags"])

    print(f"Using pip interpreter: {pip_python}")

    project_dependencies = load_project_dependencies(REPO_ROOT / "pyproject.toml")
    if pypi_version:
        print(f"Using PyPI datus-agent=={pypi_version} (skipping local wheel build)")
        datus_agent_spec = f"datus-agent=={pypi_version}"
        local_version = pypi_version
        local_wheel = None
    else:
        local_wheel, local_version = build_local_wheel(pip_python, local_dist_dir)
        datus_agent_spec = str(local_wheel)

    base_constraint_lines = build_base_constraint_lines(project_dependencies)
    base_constraint_lines = merge_extra_constraint_lines(base_constraint_lines, extra_constraint_specs)
    build_seed_wheels(
        pip_python,
        seed_wheels_dir,
        source_only_specs,
        target,
        index_url,
        extra_index_urls,
        resume=resume,
    )

    # `preserved_wheels` is the set the staged resolver must keep across each
    # `replace_wheelhouse_from_stage`. With --resume we extend it with whatever
    # was already downloaded in a prior partial run.
    preserved_by_name: dict[str, Path] = {}
    for wheel_path in pre_existing_wheels:
        preserved_by_name[wheel_path.name] = wheel_path
    for wheel_path in sorted(seed_wheels_dir.glob("*.whl")):
        preserved_by_name[wheel_path.name] = wheel_path
    if local_wheel is not None:
        preserved_by_name[local_wheel.name] = local_wheel
    preserved_wheels = list(preserved_by_name.values())
    copy_wheels_to_directory(preserved_wheels, wheelhouse_dir)

    download_top_level_packages(
        pip_python=pip_python,
        wheelhouse_dir=wheelhouse_dir,
        platform_tags=platform_tags,
        python_version=python_version,
        datus_agent_spec=datus_agent_spec,
        external_specs=external_specs,
        base_constraint_lines=base_constraint_lines,
        preserved_wheels=preserved_wheels,
        index_url=index_url,
        extra_index_urls=extra_index_urls,
    )

    requirements_lock_path = bundle_dir / "requirements.lock"
    locked_packages = write_requirements_lock(
        destination=requirements_lock_path,
        external_specs=external_specs,
        local_version=local_version,
        wheelhouse_dir=wheelhouse_dir,
    )

    install_script = bundle_dir / "install_offline.sh"
    template_path = install_script_template_path(target_name)
    if not template_path.exists():
        raise FileNotFoundError(f"Install script template not found: {template_path}")
    shutil.copy2(template_path, install_script)
    install_script.chmod(0o755)

    python_runtime_meta: dict | None = None
    if include_python_runtime:
        runtime_dir = bundle_dir / "python-runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        pbs_tar = download_pbs_runtime(
            pbs_release=pbs_release,
            pbs_python_version=pbs_python_version,
            target_name=target_name,
            cache_dir=PBS_CACHE_DIR,
        )
        shutil.copy2(pbs_tar, runtime_dir / pbs_tar.name)
        python_runtime_meta = {
            "source": "python-build-standalone",
            "release": pbs_release,
            "python_version": pbs_python_version,
            "tarball": pbs_tar.name,
        }

    runtime_asset_records: list[dict[str, str]] = []
    if huggingface_snapshots:
        fastembed_cache_dir = bundle_dir / RUNTIME_ASSETS_DIRNAME / FASTEMBED_CACHE_SUBDIR
        runtime_asset_records = download_huggingface_snapshots(
            snapshot_specs=huggingface_snapshots,
            cache_dir=fastembed_cache_dir,
        )

    readme_path = bundle_dir / "README.md"
    readme_path.write_text(
        render_bundle_readme(
            bundle_name,
            python_version,
            python_runtime_meta,
            runtime_asset_records,
            target_name=target_name,
        ),
        encoding="utf-8",
    )

    manifest = {
        "bundle_name": bundle_name,
        "target": target_name,
        "python_version": python_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "top_level_packages": locked_packages,
        "wheel_count": len(list(wheelhouse_dir.glob("*"))),
        "python_runtime": python_runtime_meta,
        "runtime_assets": {
            "huggingface_snapshots": runtime_asset_records,
        },
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    constraints_path = bundle_dir / ".constraints.txt"
    if constraints_path.exists():
        constraints_path.unlink()

    if local_dist_dir.exists():
        shutil.rmtree(local_dist_dir)
    shutil.rmtree(seed_wheels_dir)
    shutil.make_archive(str(bundle_dir), "gztar", root_dir=output_root, base_dir=bundle_name)
    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline Datus Agent bundle.")
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--external-specs-file", type=Path, default=DEFAULT_EXTERNAL_SPECS_FILE)
    parser.add_argument("--source-only-specs-file", type=Path, default=DEFAULT_SOURCE_ONLY_SPECS_FILE)
    parser.add_argument("--extra-constraints-file", type=Path, default=DEFAULT_EXTRA_CONSTRAINTS_FILE)
    parser.add_argument(
        "--huggingface-snapshots-file",
        type=Path,
        default=DEFAULT_HF_SNAPSHOTS_FILE,
        help=(
            "List of Hugging Face repo ids to pre-download into "
            f"{RUNTIME_ASSETS_DIRNAME}/{FASTEMBED_CACHE_SUBDIR}/. The install "
            "script mirrors this directory into the user's fastembed cache so "
            "Datus never has to reach huggingface.co at runtime. Use "
            "--skip-runtime-assets to opt out entirely."
        ),
    )
    parser.add_argument(
        "--skip-runtime-assets",
        action="store_true",
        help=(
            "Do not pre-fetch Hugging Face snapshots. The bundle will then be "
            "smaller but the target machine must reach huggingface.co the "
            "first time the embedding model loads."
        ),
    )
    parser.add_argument(
        "--pypi-version",
        default=None,
        help=(
            "Ship a stable PyPI-published datus-agent version instead of building "
            "from the current source tree. Example: --pypi-version 0.2.7. "
            "Omit to use the local source build (default)."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--index-url")
    parser.add_argument("--extra-index-url", action="append", default=[])
    parser.add_argument(
        "--pbs-release",
        default=DEFAULT_PBS_RELEASE,
        help=(
            "python-build-standalone release date to ship with the bundle "
            f"(default: {DEFAULT_PBS_RELEASE}). "
            "See https://github.com/astral-sh/python-build-standalone/releases"
        ),
    )
    parser.add_argument(
        "--pbs-python-version",
        default=DEFAULT_PBS_PYTHON_VERSION,
        help=(
            "CPython version inside the PBS tarball "
            f"(default: {DEFAULT_PBS_PYTHON_VERSION}). Must match a filename "
            "present in the release's SHA256SUMS."
        ),
    )
    parser.add_argument(
        "--skip-python-runtime",
        action="store_true",
        help=(
            "Do not ship a Python runtime inside the bundle; target machines "
            "must provide python3.12 on PATH. Useful for CI smoke builds where "
            "the PBS download would slow things down."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue a previous bundle build instead of starting from scratch. "
            "Existing wheels in wheelhouse/ are preserved across pip stages, "
            "seed wheels are reused, and HF snapshots / PBS runtime are "
            "redownloaded only if missing. Use after a transient network "
            "failure mid-build. Inputs (specs files, --pypi-version, etc.) "
            "must match the original run."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    major_minor = tuple(int(part) for part in args.python_version.split("."))
    if major_minor != (3, 12):
        raise SystemExit("This bundle builder is currently pinned to Python 3.12.")

    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11+ is required to run this script.")

    os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    external_specs = load_specs(args.external_specs_file.resolve(), required=True)
    source_only_specs = load_specs(args.source_only_specs_file.resolve(), required=False)
    extra_constraint_specs = load_specs(args.extra_constraints_file.resolve(), required=False)
    huggingface_snapshots: list[str] = []
    if not args.skip_runtime_assets:
        huggingface_snapshots = load_specs(args.huggingface_snapshots_file.resolve(), required=False)

    archive_path = create_bundle(
        output_root=output_root,
        target_name=args.target,
        python_version=args.python_version,
        external_specs=external_specs,
        source_only_specs=source_only_specs,
        extra_constraint_specs=extra_constraint_specs,
        huggingface_snapshots=huggingface_snapshots,
        pypi_version=args.pypi_version,
        index_url=args.index_url,
        extra_index_urls=args.extra_index_url,
        include_python_runtime=not args.skip_python_runtime,
        pbs_release=args.pbs_release,
        pbs_python_version=args.pbs_python_version,
        resume=args.resume,
    )

    print(f"Offline bundle created: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
