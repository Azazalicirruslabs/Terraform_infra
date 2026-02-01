#!/usr/bin/env python
"""
bootstrap.py – Fully automatic setup for the repo
Runs exactly once when someone does `pip install -r requirements.txt`

What it does (in order):
1. Creates .venv if it does not exist
2. Upgrades pip silently
3. Installs pre-commit (from requirements-dev.txt)
4. Installs the git hooks
5. Runs pre-commit autoupdate (keeps all hooks fresh)
"""

import os
import subprocess
import sys
from pathlib import Path


def run(cmd: str, **kwargs):
    """Helper to run commands and print what is happening"""
    print(f"Running → {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kwargs)


def main():
    repo_root = Path(__file__).parent.parent  # scripts/.. → repo root
    os.chdir(repo_root)

    venv_path = repo_root / ".venv"

    # 1. Create virtual environment if missing
    if not venv_path.exists():
        print("Creating virtual environment (.venv) ...")
        run(f"{sys.executable} -m venv .venv")

    # 2. Determine correct activation command & python binary
    if os.name == "nt":  # Windows
        python = str(venv_path / "Scripts" / "python.exe")
        print(f"To activate the virtual environment, run: {venv_path}\\Scripts\\activate.bat")
    else:  # Mac / Linux
        python = str(venv_path / "bin" / "python")
        print(f"To activate the virtual environment, run: source {venv_path}/bin/activate")

    # 3. Upgrade pip inside the venv
    print("Upgrading pip inside .venv ...")
    run(f'"{python}" -m pip install --upgrade --quiet pip setuptools wheel')

    # 4. Install pre-commit (and any other dev tools from requirements-dev.txt)
    print("Installing pre-commit and dev dependencies ...")
    run(f'"{python}" -m pip install --quiet pre-commit')

    # 5. Install the git hooks (force + install hooks)
    print("Installing pre-commit hooks ...")
    subprocess.run(
        [python, "-m", "pre_commit", "install", "-f", "--install-hooks"],
        check=True,
    )

    # 6. Keep all pre-commit hooks up-to-date automatically
    print("Running pre-commit autoupdate ...")
    subprocess.run([python, "-m", "pre_commit", "autoupdate"], check=True)

    print("\nAll done! pre-commit is fully installed and up-to-date.")
    print("You can now commit safely – hooks will run automatically.\n")


if __name__ == "__main__":
    main()
