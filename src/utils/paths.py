"""Canonical path resolution."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def probe_results_dir(model_name: str, probe_type: str) -> Path:
    """Return results/probes/{model}/{probe_type}/, creating if needed."""
    d = RESULTS_DIR / "probes" / model_name / probe_type
    d.mkdir(parents=True, exist_ok=True)
    return d


def vector_results_dir(model_name: str, behavior: str) -> Path:
    d = RESULTS_DIR / "vectors" / model_name / behavior
    d.mkdir(parents=True, exist_ok=True)
    return d


def contrast_pairs_dir() -> Path:
    d = DATA_DIR / "contrast_pairs"
    d.mkdir(parents=True, exist_ok=True)
    return d
