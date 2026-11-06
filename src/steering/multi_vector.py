"""Multi-vector composition for simultaneous multi-behavior steering.

Naive stacking of steering vectors fails when vectors are correlated
(e.g., refusal and anti-sycophancy vectors share variance). This module
provides orthogonalization to ensure independent steering directions.
"""

import torch
from torch import Tensor


def orthogonalize_vectors(vectors: list[Tensor]) -> list[Tensor]:
    """Gram-Schmidt orthogonalization of steering vectors.

    Orthogonalizes each vector against all preceding vectors, preserving
    the original magnitude. The first vector is unchanged; subsequent
    vectors have their projections onto prior vectors removed.

    Args:
        vectors: List of 1-D tensors (hidden_dim,) to orthogonalize.

    Returns:
        List of orthogonalized tensors with original magnitudes preserved.
        Vectors that become near-zero after orthogonalization (i.e., were
        nearly parallel to a prior vector) are excluded with a warning.
    """
    ortho = []
    for i, v in enumerate(vectors):
        v_orth = v.clone().float()
        original_norm = torch.norm(v.float())

        for u in ortho:
            # Remove projection of v onto u
            proj = (torch.dot(v_orth, u) / torch.dot(u, u)) * u
            v_orth = v_orth - proj

        residual_norm = torch.norm(v_orth)
        if residual_norm < 1e-6:
            print(f"  Warning: vector {i} is nearly parallel to prior vectors "
                  f"(residual norm {residual_norm:.2e}), skipping.")
            continue

        # Rescale to original magnitude
        v_orth = v_orth / residual_norm * original_norm
        ortho.append(v_orth)

    return ortho


def compute_cosine_matrix(vectors: list[Tensor]) -> Tensor:
    """Compute pairwise cosine similarity matrix for diagnostics.

    Args:
        vectors: List of 1-D tensors.

    Returns:
        (n, n) tensor of cosine similarities.
    """
    n = len(vectors)
    matrix = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            cos = torch.dot(vectors[i].float(), vectors[j].float()) / (
                torch.norm(vectors[i].float()) * torch.norm(vectors[j].float()) + 1e-8
            )
            matrix[i, j] = cos
    return matrix
