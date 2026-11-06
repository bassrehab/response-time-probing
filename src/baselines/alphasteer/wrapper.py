"""AlphaSteer baseline wrapper.

Wraps the AlphaSteer codebase (github.com/AlphaLab-USTC/AlphaSteer) for
use in our evaluation harness.

AlphaSteer uses learned Δ matrices with null-space constraints for
input-dependent activation steering:
  h' = h + λ · Δ̃* · P̂ · h

Where:
  - Δ̃* = learned transformation matrix (per layer)
  - P̂ = null-space projection matrix (ensures benign inputs unaffected)
  - λ = steering strength

The null-space projection is key: P̂ h ≈ 0 for benign activations,
so steering only affects inputs that deviate from the benign subspace.

Setup:
  1. Clone: git clone https://github.com/AlphaLab-USTC/AlphaSteer external/AlphaSteer
  2. Install: cd external/AlphaSteer && pip install -e .
  3. Train Δ matrices: python scripts/05_train_alphasteer.py (A100 required)
  4. Use this wrapper for inference

References:
  - Paper: "AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint" (ICLR 2026)
  - Their reported DSR: Mistral N/A, Gemma 98.2%, Qwen 97.3%
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn, Tensor

from src.models.hooks import ActivationHook
from src.utils.paths import RESULTS_DIR


@dataclass
class AlphaSteerConfig:
    """Configuration for AlphaSteer inference."""

    model_name: str
    matrices_dir: Path          # Directory with delta_*.pt and projection_*.pt files
    steering_layers: list[int]  # Which layers to steer
    strength: float = 1.0       # λ parameter


class AlphaSteerDefense:
    """Wrapper that applies AlphaSteer's Δ·P̂ steering during inference.

    Loads pre-computed Δ̃* and P̂ matrices and registers forward hooks
    on specified layers. Compatible with our evaluation harness.
    """

    def __init__(self, model: nn.Module, config: AlphaSteerConfig) -> None:
        self.model = model
        self.config = config
        self._handles = []

        # Load matrices per layer
        self._delta_matrices: dict[int, Tensor] = {}
        self._proj_matrices: dict[int, Tensor] = {}

        for layer_idx in config.steering_layers:
            delta_path = config.matrices_dir / f"delta_layer{layer_idx}.pt"
            proj_path = config.matrices_dir / f"projection_layer{layer_idx}.pt"

            if not delta_path.exists() or not proj_path.exists():
                raise FileNotFoundError(
                    f"AlphaSteer matrices not found at {config.matrices_dir}. "
                    f"Run scripts/05_train_alphasteer.py on A100 first."
                )

            self._delta_matrices[layer_idx] = torch.load(delta_path, weights_only=True)
            self._proj_matrices[layer_idx] = torch.load(proj_path, weights_only=True)

        self._register_hooks()

    def _register_hooks(self) -> None:
        for layer_idx in self.config.steering_layers:
            layer = ActivationHook._resolve_layer(self.model, layer_idx)
            handle = layer.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)

    def _make_hook(self, layer_idx: int):
        delta = self._delta_matrices[layer_idx]
        proj = self._proj_matrices[layer_idx]
        strength = self.config.strength

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # h' = h + λ · Δ̃* · P̂ · h
            # Apply to last token only (for causal generation)
            h = hidden[:, -1, :]  # (batch, hidden_dim)
            d = delta.to(h.device, dtype=h.dtype)
            p = proj.to(h.device, dtype=h.dtype)

            steering = strength * (h @ p.T @ d.T)  # (batch, hidden_dim)
            hidden[:, -1, :] += steering

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        return hook_fn

    def set_strength(self, strength: float) -> None:
        self.config.strength = strength
        # Need to re-register hooks with new strength
        self.remove()
        self._register_hooks()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def compute_null_space_projection(
    benign_activations: np.ndarray,
    rank_threshold: float = 0.01,
) -> np.ndarray:
    """Compute null-space projection P̂ from benign activations.

    P̂ projects onto the null space of the benign activation subspace,
    ensuring that steering has no effect on benign inputs.

    Args:
        benign_activations: (n_samples, hidden_dim) matrix of benign activations.
        rank_threshold: Eigenvalue threshold for null-space boundary.

    Returns:
        (hidden_dim, hidden_dim) projection matrix P̂.
    """
    # Center activations
    mean = benign_activations.mean(axis=0, keepdims=True)
    centered = benign_activations - mean

    # Covariance and SVD
    cov = centered.T @ centered / len(centered)
    U, S, Vt = np.linalg.svd(cov, full_matrices=True)

    # Find null space: eigenvalues below threshold
    max_eigenval = S[0]
    null_mask = S < (rank_threshold * max_eigenval)
    null_vectors = U[:, null_mask]  # (hidden_dim, n_null)

    # Projection matrix: P̂ = U_null · U_null^T
    P_hat = null_vectors @ null_vectors.T

    return P_hat


def compute_delta_matrix(
    malicious_activations: np.ndarray,
    refusal_directions: np.ndarray,
    P_hat: np.ndarray,
    alpha: float = 0.01,
) -> np.ndarray:
    """Compute the learned Δ̃* matrix via regularized least squares.

    Δ̃* = argmin ||Δ̃ · P̂ · H_m - R||² + α ||Δ̃ · P̂||²

    Args:
        malicious_activations: (n_samples, hidden_dim) malicious activations.
        refusal_directions: (n_samples, hidden_dim) target refusal vectors.
        P_hat: (hidden_dim, hidden_dim) null-space projection.
        alpha: Regularization strength.

    Returns:
        (hidden_dim, hidden_dim) transformation matrix Δ̃*.
    """
    H_m = malicious_activations.T  # (d, n)
    R = refusal_directions.T  # (d, n)

    # Projected malicious activations
    PH = P_hat @ H_m  # (d, n)

    # Closed-form: Δ̃* = R · (PH)^T · (PH · (PH)^T + α · P̂)^(-1)
    PHPH_T = PH @ PH.T  # (d, d)
    regularized = PHPH_T + alpha * P_hat  # (d, d)

    # Solve via pseudoinverse for numerical stability
    Delta = R @ PH.T @ np.linalg.pinv(regularized)

    return Delta


def save_alphasteer_matrices(
    output_dir: Path,
    delta: np.ndarray,
    projection: np.ndarray,
    layer: int,
) -> None:
    """Save Δ̃* and P̂ matrices for a layer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(delta).float(), output_dir / f"delta_layer{layer}.pt")
    torch.save(torch.from_numpy(projection).float(), output_dir / f"projection_layer{layer}.pt")
