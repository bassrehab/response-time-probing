"""CAA (Contrastive Activation Addition) vector extraction.

Computes steering vectors as the mean activation difference between
positive (harmful) and negative (benign) prompt sets at a given layer.
Vectors are normalized to unit length for consistent steering strength control.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def compute_caa_vector(
    positive_activations: np.ndarray,
    negative_activations: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Compute a CAA steering vector from contrast activations.

    vector = mean(positive) - mean(negative), then normalized to unit length.

    Args:
        positive_activations: (n_pos, hidden_dim) activations for harmful prompts.
        negative_activations: (n_neg, hidden_dim) activations for benign prompts.

    Returns:
        (unit_vector, metadata) where metadata contains norms and cosine stats.
    """
    mean_pos = positive_activations.mean(axis=0)
    mean_neg = negative_activations.mean(axis=0)

    raw_vector = mean_pos - mean_neg
    raw_norm = float(np.linalg.norm(raw_vector))

    if raw_norm < 1e-8:
        raise ValueError("Steering vector has near-zero norm — positive and negative "
                         "activations are indistinguishable at this layer.")

    unit_vector = raw_vector / raw_norm

    # Compute cosine similarity between individual samples and the vector
    # for quality diagnostics
    pos_cosines = positive_activations @ unit_vector / (
        np.linalg.norm(positive_activations, axis=1) + 1e-8
    )
    neg_cosines = negative_activations @ unit_vector / (
        np.linalg.norm(negative_activations, axis=1) + 1e-8
    )

    metadata = {
        "raw_norm": raw_norm,
        "n_positive": len(positive_activations),
        "n_negative": len(negative_activations),
        "hidden_dim": int(positive_activations.shape[1]),
        "pos_cosine_mean": float(pos_cosines.mean()),
        "pos_cosine_std": float(pos_cosines.std()),
        "neg_cosine_mean": float(neg_cosines.mean()),
        "neg_cosine_std": float(neg_cosines.std()),
        "separation": float(pos_cosines.mean() - neg_cosines.mean()),
    }

    return unit_vector, metadata


def save_steering_vector(
    output_dir: Path,
    vector: np.ndarray,
    metadata: dict,
    layer: int,
) -> None:
    """Save steering vector and metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        torch.from_numpy(vector).float(),
        output_dir / f"vector_layer{layer}.pt",
    )

    metadata["layer"] = layer
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_steering_vector(path: Path) -> Tensor:
    """Load a saved steering vector."""
    return torch.load(path, weights_only=True)
