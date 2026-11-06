"""Tests for steering vector operations."""

import torch
import numpy as np
import pytest

from src.steering.multi_vector import orthogonalize_vectors, compute_cosine_matrix


class TestOrthogonalizeVectors:
    """Test Gram-Schmidt orthogonalization of steering vectors."""

    def test_two_orthogonal_vectors_unchanged(self):
        """Vectors that are already orthogonal should be unchanged (up to sign)."""
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, 1.0, 0.0])
        result = orthogonalize_vectors([v1, v2])
        assert len(result) == 2
        assert torch.allclose(result[0], v1)
        assert torch.allclose(result[1], v2)

    def test_two_correlated_vectors(self):
        """Correlated vectors should become orthogonal."""
        v1 = torch.tensor([1.0, 1.0, 0.0])
        v2 = torch.tensor([1.0, 0.0, 0.0])
        result = orthogonalize_vectors([v1, v2])
        assert len(result) == 2
        cos = torch.dot(result[0], result[1]) / (
            torch.norm(result[0]) * torch.norm(result[1])
        )
        assert abs(cos.item()) < 1e-5, f"Vectors not orthogonal: cosine={cos:.6f}"

    def test_preserves_magnitude(self):
        """Original magnitudes should be preserved after orthogonalization."""
        v1 = torch.tensor([3.0, 4.0, 0.0])  # norm = 5
        v2 = torch.tensor([1.0, 1.0, 1.0])  # norm = sqrt(3)
        result = orthogonalize_vectors([v1, v2])
        assert abs(torch.norm(result[0]).item() - 5.0) < 1e-5
        assert abs(torch.norm(result[1]).item() - np.sqrt(3)) < 1e-4

    def test_parallel_vectors_removed(self):
        """A vector parallel to a prior vector should be dropped."""
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([2.0, 0.0, 0.0])  # parallel to v1
        result = orthogonalize_vectors([v1, v2])
        assert len(result) == 1

    def test_three_vectors(self):
        """Three vectors should all be mutually orthogonal."""
        v1 = torch.tensor([1.0, 1.0, 0.0, 0.0])
        v2 = torch.tensor([1.0, 0.0, 1.0, 0.0])
        v3 = torch.tensor([1.0, 0.0, 0.0, 1.0])
        result = orthogonalize_vectors([v1, v2, v3])
        assert len(result) == 3
        cos_matrix = compute_cosine_matrix(result)
        for i in range(3):
            for j in range(i + 1, 3):
                assert abs(cos_matrix[i, j].item()) < 1e-5, \
                    f"Vectors {i},{j} not orthogonal: cos={cos_matrix[i, j]:.6f}"

    def test_high_dimensional(self):
        """Works on high-dimensional vectors (like real hidden dims)."""
        torch.manual_seed(42)
        dim = 4096
        v1 = torch.randn(dim)
        v2 = torch.randn(dim)
        v3 = torch.randn(dim)
        result = orthogonalize_vectors([v1, v2, v3])
        assert len(result) == 3
        cos_matrix = compute_cosine_matrix(result)
        for i in range(3):
            for j in range(i + 1, 3):
                assert abs(cos_matrix[i, j].item()) < 1e-4

    def test_empty_input(self):
        result = orthogonalize_vectors([])
        assert result == []

    def test_single_vector(self):
        v = torch.tensor([1.0, 2.0, 3.0])
        result = orthogonalize_vectors([v])
        assert len(result) == 1
        assert torch.allclose(result[0], v)


class TestCosineMatrix:
    """Test cosine similarity matrix computation."""

    def test_identity_on_orthogonal(self):
        vectors = [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
        ]
        matrix = compute_cosine_matrix(vectors)
        assert matrix.shape == (3, 3)
        # Diagonal should be 1.0
        for i in range(3):
            assert abs(matrix[i, i].item() - 1.0) < 1e-5
        # Off-diagonal should be 0.0
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert abs(matrix[i, j].item()) < 1e-5

    def test_parallel_vectors(self):
        vectors = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([3.0, 0.0]),
        ]
        matrix = compute_cosine_matrix(vectors)
        assert abs(matrix[0, 1].item() - 1.0) < 1e-5
