"""Load/save trained probes by (model, behavior, layer)."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils.paths import probe_results_dir


def load_probe(
    model_name: str, probe_type: str, layer: int | None = None
) -> tuple[LogisticRegression, StandardScaler, dict]:
    """Load a trained probe.

    If layer is None, loads the best probe. Otherwise loads the probe at that layer
    (only best is saved by default, so this will raise if a non-best layer is requested).

    Returns:
        (probe, scaler, metadata_dict)
    """
    d = probe_results_dir(model_name, probe_type)
    probe_path = d / "best_probe.pkl"
    if not probe_path.exists():
        raise FileNotFoundError(f"No trained probe at {probe_path}")

    with open(probe_path, "rb") as f:
        data = pickle.load(f)

    import json
    with open(d / "best_layer.json") as f:
        meta = json.load(f)

    if layer is not None and meta["layer"] != layer:
        raise ValueError(
            f"Requested layer {layer} but only best layer {meta['layer']} is saved."
        )

    return data["probe"], data["scaler"], meta
