#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL_SPECS_FILE = REPO_ROOT / "offline" / "external-package-specs.txt"
DEFAULT_SOURCE_ONLY_SPECS_FILE = REPO_ROOT / "offline" / "source-only-package-specs.txt"
DEFAULT_EXTRA_CONSTRAINTS_FILE = REPO_ROOT / "offline" / "extra-constraints.txt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "offline" / "dist"

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
) -> None:
    if seed_wheels_dir.exists():
        shutil.rmtree(seed_wheels_dir)
    seed_wheels_dir.mkdir(parents=True, exist_ok=True)

    if not source_only_specs:
        return

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
    if wheelhouse_dir.exists():
        for wheel_path in wheelhouse_dir.glob("*.whl"):
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
    local_wheel: Path,
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

        # Always pass the freshly built local wheel as a direct spec so pip is
        # forced to use its metadata instead of racing a same-versioned wheel
        # that may already exist on PyPI with stale/divergent Requires-Dist.
        package_specs = [*resolved_top_level_specs, stage_spec, str(local_wheel)]
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

    final_package_specs = [*resolved_top_level_specs, str(local_wheel)]
    stage_dir.mkdir(parents=True, exist_ok=True)
    print(f"Resolving dependencies for local package: {local_wheel.name}")
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


def render_bundle_readme(bundle_name: str, python_version: str) -> str:
    return textwrap.dedent(
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

        ## Prerequisites
        - Linux target machine
        - A glibc version compatible with the bundled Linux wheels (for example,
          DuckDB 1.3.0 currently requires glibc 2.27+)
        - Python {python_version}
        - `venv` module available in the Python installation
        """
    )


def create_bundle(
    output_root: Path,
    target_name: str,
    python_version: str,
    external_specs: list[str],
    source_only_specs: list[str],
    extra_constraint_specs: list[str],
    index_url: str | None,
    extra_index_urls: list[str],
) -> Path:
    pip_python = resolve_pip_python()
    target = TARGETS[target_name]
    bundle_name = f"datus-agent-offline-{target['bundle_suffix']}-py{python_version.replace('.', '')}"
    bundle_dir = output_root / bundle_name
    archive_path = output_root / f"{bundle_name}.tar.gz"

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    if archive_path.exists():
        archive_path.unlink()

    wheelhouse_dir = bundle_dir / "wheelhouse"
    local_dist_dir = bundle_dir / ".local-dist"
    seed_wheels_dir = bundle_dir / ".seed-wheels"
    wheelhouse_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using pip interpreter: {pip_python}")
    local_wheel, local_version = build_local_wheel(pip_python, local_dist_dir)
    project_dependencies = load_project_dependencies(REPO_ROOT / "pyproject.toml")
    base_constraint_lines = build_base_constraint_lines(project_dependencies)
    base_constraint_lines = merge_extra_constraint_lines(base_constraint_lines, extra_constraint_specs)
    build_seed_wheels(pip_python, seed_wheels_dir, source_only_specs, target, index_url, extra_index_urls)
    preserved_wheels = [local_wheel, *sorted(seed_wheels_dir.glob("*.whl"))]
    copy_wheels_to_directory(preserved_wheels, wheelhouse_dir)

    download_top_level_packages(
        pip_python=pip_python,
        wheelhouse_dir=wheelhouse_dir,
        platform_tags=list(target["platform_tags"]),
        python_version=python_version,
        local_wheel=local_wheel,
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

    readme_path = bundle_dir / "README.md"
    readme_path.write_text(render_bundle_readme(bundle_name, python_version), encoding="utf-8")

    manifest = {
        "bundle_name": bundle_name,
        "target": target_name,
        "python_version": python_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "top_level_packages": locked_packages,
        "wheel_count": len(list(wheelhouse_dir.glob("*"))),
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    constraints_path = bundle_dir / ".constraints.txt"
    if constraints_path.exists():
        constraints_path.unlink()

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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--index-url")
    parser.add_argument("--extra-index-url", action="append", default=[])
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

    archive_path = create_bundle(
        output_root=output_root,
        target_name=args.target,
        python_version=args.python_version,
        external_specs=external_specs,
        source_only_specs=source_only_specs,
        extra_constraint_specs=extra_constraint_specs,
        index_url=args.index_url,
        extra_index_urls=args.extra_index_url,
    )

    print(f"Offline bundle created: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
