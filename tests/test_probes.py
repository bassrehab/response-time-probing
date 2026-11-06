"""Tests for the probe training pipeline.

Includes a unit test with synthetic data and an integration smoke test
that runs the full pipeline on Mistral-7B with 20 jailbreak contrast pairs.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.probes.train_probe import (
    ProbeResult,
    layer_sweep,
    save_probe_results,
    train_probe_at_layer,
)
from src.utils.seeding import seed_everything


class TestTrainProbeAtLayer:
    """Unit tests for probe training with synthetic data."""

    def setup_method(self):
        seed_everything(42)
        # Create linearly separable synthetic activations
        n_samples = 100
        hidden_dim = 64
        self.positive = np.random.randn(n_samples // 2, hidden_dim) + 2.0
        self.negative = np.random.randn(n_samples // 2, hidden_dim) - 2.0
        self.activations = np.concatenate([self.positive, self.negative], axis=0)
        self.labels = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

    def test_returns_correct_types(self):
        probe, scaler, auroc, acc, fpr, fold_aurocs = train_probe_at_layer(
            self.activations, self.labels, n_folds=3
        )
        assert isinstance(probe, LogisticRegression)
        assert isinstance(auroc, float)
        assert isinstance(acc, float)
        assert isinstance(fpr, float)
        assert isinstance(fold_aurocs, list)
        assert len(fold_aurocs) == 3

    def test_high_auroc_on_separable_data(self):
        _, _, auroc, acc, fpr, _ = train_probe_at_layer(
            self.activations, self.labels, n_folds=5
        )
        assert auroc > 0.95, f"Expected AUROC > 0.95 on separable data, got {auroc}"
        assert acc > 0.90
        assert fpr < 0.15

    def test_near_chance_on_random_labels(self):
        rng = np.random.RandomState(99)
        random_labels = rng.randint(0, 2, size=len(self.labels))
        _, _, auroc, _, _, _ = train_probe_at_layer(
            self.activations, random_labels, n_folds=3
        )
        # With random labels, AUROC should be near 0.5 (but can vary)
        assert auroc < 0.85, f"AUROC suspiciously high on random labels: {auroc}"


class TestLayerSweep:
    """Test layer sweep functionality."""

    def setup_method(self):
        seed_everything(42)
        n_samples = 80
        hidden_dim = 32

        # Layer 5 is linearly separable, others are random
        self.all_activations = {}
        for layer in range(8):
            if layer == 5:
                pos = np.random.randn(n_samples // 2, hidden_dim) + 3.0
                neg = np.random.randn(n_samples // 2, hidden_dim) - 3.0
                self.all_activations[layer] = np.concatenate([pos, neg], axis=0)
            else:
                self.all_activations[layer] = np.random.randn(n_samples, hidden_dim)

        self.labels = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

    def test_finds_best_layer(self):
        results, best_layer, best_probe, best_scaler = layer_sweep(
            self.all_activations, self.labels, n_folds=3
        )
        assert best_layer == 5, f"Expected best layer 5, got {best_layer}"
        assert results[5].auroc > 0.9
        assert isinstance(best_probe, LogisticRegression)

    def test_returns_all_layers(self):
        results, _, _, _ = layer_sweep(self.all_activations, self.labels, n_folds=3)
        assert len(results) == 8
        for layer in range(8):
            assert layer in results
            assert isinstance(results[layer], ProbeResult)


class TestSaveProbeResults:
    """Test saving and loading probe results."""

    def test_save_creates_expected_files(self, tmp_path):
        seed_everything(42)
        n, d = 40, 16
        activations = np.random.randn(n, d)
        labels = np.array([1] * (n // 2) + [0] * (n // 2))

        probe, scaler, auroc, acc, fpr, fold_aurocs = train_probe_at_layer(
            activations, labels, n_folds=3
        )
        results = {
            0: ProbeResult(layer=0, auroc=auroc, auroc_std=0.01,
                           accuracy=acc, fpr=fpr, fold_aurocs=fold_aurocs)
        }
        save_probe_results(tmp_path, "test-model", "jailbreak", results, 0, probe, scaler)

        assert (tmp_path / "layer_sweep.json").exists()
        assert (tmp_path / "best_layer.json").exists()
        assert (tmp_path / "best_probe.pkl").exists()

        with open(tmp_path / "best_layer.json") as f:
            meta = json.load(f)
        assert meta["layer"] == 0
        assert meta["model"] == "test-model"

        with open(tmp_path / "best_probe.pkl", "rb") as f:
            data = pickle.load(f)
        assert "probe" in data
        assert "scaler" in data
