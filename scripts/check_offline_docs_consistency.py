#!/usr/bin/env python3
"""Lint offline/SYSTEM_REQUIREMENTS.md against the actual build system.

Catches documentation drift when CLI flags, make variables, install-script
modes, or referenced file paths change but the user-facing docs forget to
follow. Exits non-zero on any mismatch so CI can block the merge.

Checks performed:

  1. For each user-facing make variable (PYPI_VERSION), the full chain holds:
     doc mentions it -> Makefile declares `VAR ?=` -> Makefile forwards
     `$(VAR)` into the matching builder CLI flag -> builder argparse
     registers that flag.
  2. Install-script modes the doc advertises (--user, --system) have matching
     case branches in both install_offline template scripts.
  3. Any file path mentioned in the doc exists on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "offline" / "SYSTEM_REQUIREMENTS.md"
BUILDER = ROOT / "scripts" / "build_offline_bundle.py"
MAKEFILE = ROOT / "Makefile"
INSTALL_TEMPLATES = {
    "x86_64": ROOT / "offline" / "templates" / "install_offline_linux-x86_64.sh",
    "arm64": ROOT / "offline" / "templates" / "install_offline_linux-arm64.sh",
}
REFERENCED_FILES = [
    ROOT / "offline" / "extra-constraints.txt",
    *INSTALL_TEMPLATES.values(),
]

# User-facing make variables: each one must (a) appear in the doc,
# (b) be declared in the Makefile as `VAR ?=`, and (c) forward to the
# builder CLI flag listed below. The chain is what customers actually
# rely on (`make offline-bundle-x86_64 PYPI_VERSION=0.2.6`), so breaking
# any link silently breaks the customer command.
MAKE_VAR_TO_BUILDER_FLAG = {
    "PYPI_VERSION": "--pypi-version",
}
INSTALL_MODES_REQUIRED_IN_DOC = ("--user", "--system")


def check() -> list[str]:
    errors: list[str] = []

    if not DOC.exists():
        return [f"{DOC.relative_to(ROOT)} does not exist"]

    doc = DOC.read_text(encoding="utf-8")
    builder = BUILDER.read_text(encoding="utf-8")
    makefile = MAKEFILE.read_text(encoding="utf-8")

    # 1. Follow the full chain for each user-facing make variable:
    #      doc  -->  Makefile declaration  -->  Makefile forwarding  -->  builder argparse
    for var, flag in MAKE_VAR_TO_BUILDER_FLAG.items():
        if var not in doc:
            errors.append(f"doc is missing reference to make variable {var}")
        if f"{var} ?=" not in makefile:
            errors.append(f"doc references make variable {var} but '{var} ?=' was not found in Makefile")
        # Makefile must forward the variable into the builder flag.
        if f"$({var})" not in makefile or flag not in makefile:
            errors.append(
                f"Makefile must forward {var} to the builder as {flag} "
                f"(expected both '$({var})' and '{flag}' to appear in Makefile)"
            )
        # Builder must register the flag.
        if f'"{flag}"' not in builder:
            errors.append(
                f"{MAKEFILE.relative_to(ROOT)} forwards {var} as {flag} but "
                f"parser.add_argument('{flag}') was not found in {BUILDER.relative_to(ROOT)}"
            )

    # 4: install_offline.sh modes
    for mode in INSTALL_MODES_REQUIRED_IN_DOC:
        if mode not in doc:
            errors.append(f"doc is missing install_offline mode {mode}")
        branch = mode.lstrip("-")
        for label, template in INSTALL_TEMPLATES.items():
            text = template.read_text(encoding="utf-8")
            if f"  {branch})" not in text:
                errors.append(
                    f"doc claims mode {mode} works but case branch '{branch})' "
                    f"was not found in {template.relative_to(ROOT)} (arch={label})"
                )

    # 3: referenced file paths exist
    for path in REFERENCED_FILES:
        rel = str(path.relative_to(ROOT))
        if rel in doc and not path.exists():
            errors.append(f"doc references {rel} but that file does not exist")

    return errors


def main() -> int:
    errors = check()
    if errors:
        for msg in errors:
            print(f"[FAIL] {msg}", file=sys.stderr)
        print(
            f"\n{len(errors)} inconsistency(ies) between {DOC.relative_to(ROOT)} and source of truth",
            file=sys.stderr,
        )
        return 1
    print(f"[OK] {DOC.relative_to(ROOT)} is consistent with source of truth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
