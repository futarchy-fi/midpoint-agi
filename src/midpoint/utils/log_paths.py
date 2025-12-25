"""
Utilities for resolving log paths in a repo-safe way.

We intentionally resolve logs to an absolute `{repo_root}/logs` directory so that:
- logs are not affected by the current working directory (e.g. running from `.goal/`)
- tests that create temporary git repos still get logs rooted in that temp repo
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def find_repo_root(start: Optional[Union[str, Path]] = None) -> Path:
    """
    Find the repository root by walking upward looking for `.git/` or `pyproject.toml`.

    This prefers the *runtime* repo (based on the provided start path or CWD) rather than the
    installed package location, so it works correctly when tools run inside a temp git repo.
    """
    base = Path(start) if start is not None else Path.cwd()
    base = base.resolve()

    for p in [base, *base.parents]:
        if (p / ".git").is_dir():
            return p
        if (p / "pyproject.toml").is_file():
            return p

    # Fallback: best-effort. This keeps behavior deterministic even outside a repo.
    return base


def get_logs_dir(repo_root: Optional[Union[str, Path]] = None) -> Path:
    """Return absolute `{repo_root}/logs` and ensure it exists."""
    root = find_repo_root(repo_root)
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir

