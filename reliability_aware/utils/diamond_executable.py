from __future__ import annotations

import os
import shutil
from pathlib import Path

DIAMOND_ENV_VAR = "DIAMOND_EXECUTABLE"
CONTAINER_DIAMOND_PATH = Path("/app/diamond")


def _usable_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def resolve_diamond_executable(
    *,
    explicit: str | os.PathLike[str] | None = None,
    container_path: Path = CONTAINER_DIAMOND_PATH,
) -> Path | None:
    """Resolve DIAMOND using the deployment-wide precedence policy."""
    configured = os.environ.get(DIAMOND_ENV_VAR)
    if configured:
        return _resolve_configured_executable(configured)

    if explicit:
        return _resolve_configured_executable(os.fspath(explicit))

    if _usable_executable(container_path):
        return container_path.resolve()

    resolved = shutil.which("diamond")
    return Path(resolved).resolve() if resolved else None


def _resolve_configured_executable(configured: str) -> Path | None:
    candidate = Path(configured).expanduser()
    if _usable_executable(candidate):
        return candidate.resolve()

    resolved = shutil.which(configured)
    return Path(resolved).resolve() if resolved else None


def diamond_resolution_error() -> str:
    return (
        "DIAMOND executable not found. Set DIAMOND_EXECUTABLE to an executable "
        "file, provide /app/diamond, or install 'diamond' on PATH."
    )
