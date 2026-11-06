"""Run metadata capture for experiment reproducibility.

Captures git state, environment, timestamps, and writes structured
metadata alongside experiment results. Also generates human-readable
summaries in _dev/experiments/.
"""

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from src.utils.paths import PROJECT_ROOT


@dataclass
class RunMetadata:
    """Metadata captured at experiment start."""

    experiment_name: str
    timestamp: str
    git_hash: str
    git_dirty: bool
    python_version: str
    platform: str
    hardware: str
    script: str
    args: dict


def capture_run_metadata(
    experiment_name: str,
    script: str,
    args: dict | None = None,
) -> RunMetadata:
    """Capture current environment metadata for an experiment run."""
    # Git state
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, text=True,
        ).strip()
    except Exception:
        git_hash = "unknown"

    try:
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, text=True,
        ).strip()
        git_dirty = len(git_status) > 0
    except Exception:
        git_dirty = True

    # Hardware
    machine = platform.machine()
    proc = platform.processor() or machine
    try:
        import torch
        if torch.cuda.is_available():
            hw = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif torch.backends.mps.is_available():
            hw = f"Apple MPS ({proc})"
        else:
            hw = f"CPU ({proc})"
    except Exception:
        hw = f"CPU ({proc})"

    return RunMetadata(
        experiment_name=experiment_name,
        timestamp=datetime.now().isoformat(),
        git_hash=git_hash,
        git_dirty=git_dirty,
        python_version=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()}",
        hardware=hw,
        script=script,
        args=args or {},
    )


def save_run_metadata(output_dir: Path, metadata: RunMetadata) -> None:
    """Save run metadata JSON alongside experiment results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(asdict(metadata), f, indent=2)
