"""Train linear probes on activations at specified layers.

Supports full layer sweep with 5-fold cross-validation, returning AUROC per layer.
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    """Result of training a probe at a single layer."""

    layer: int
    auroc: float
    auroc_std: float
    accuracy: float
    fpr: float  # False positive rate at threshold=0.5
    fold_aurocs: list[float]


@dataclass
class LayerSweepResult:
    """Results of sweeping all layers."""

    model_name: str
    probe_type: str
    best_layer: int
    best_auroc: float
    per_layer: list[ProbeResult]


def train_probe_at_layer(
    activations: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[LogisticRegression, StandardScaler, float, float, float, list[float]]:
    """Train a LogisticRegression probe with cross-validation.

    Args:
        activations: (n_samples, hidden_dim) array of activations.
        labels: (n_samples,) binary labels (1=positive/unsafe, 0=negative/safe).
        n_folds: Number of cross-validation folds.
        seed: Random seed.

    Returns:
        (probe, scaler, mean_auroc, mean_accuracy, mean_fpr, fold_aurocs)
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aurocs = []
    fold_accs = []
    fold_fprs = []

    for train_idx, val_idx in skf.split(activations, labels):
        X_train, X_val = activations[train_idx], activations[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        probe = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
            C=1.0,
        )
        probe.fit(X_train, y_train)

        probs = probe.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)

        auroc = roc_auc_score(y_val, probs)
        acc = (preds == y_val).mean()

        # FPR: fraction of negatives (label=0) predicted as positive
        neg_mask = y_val == 0
        if neg_mask.sum() > 0:
            fpr = preds[neg_mask].mean()
        else:
            fpr = 0.0

        fold_aurocs.append(auroc)
        fold_accs.append(acc)
        fold_fprs.append(fpr)

    # Train final probe on all data for saving
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(activations)
    probe_final = LogisticRegression(
        max_iter=1000, solver="lbfgs", random_state=seed, C=1.0
    )
    probe_final.fit(X_scaled, labels)

    return (
        probe_final,
        scaler_final,
        float(np.mean(fold_aurocs)),
        float(np.mean(fold_accs)),
        float(np.mean(fold_fprs)),
        [float(a) for a in fold_aurocs],
    )


def layer_sweep(
    all_activations: dict[int, np.ndarray],
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[dict[int, ProbeResult], int, LogisticRegression, StandardScaler]:
    """Train probes at every layer and find the best one.

    Args:
        all_activations: {layer_idx: (n_samples, hidden_dim)} activations per layer.
        labels: (n_samples,) binary labels.
        n_folds: CV folds.
        seed: Random seed.

    Returns:
        (results_per_layer, best_layer, best_probe, best_scaler)
    """
    results: dict[int, ProbeResult] = {}
    layer_probes: dict[int, tuple[LogisticRegression, StandardScaler]] = {}

    for layer_idx in sorted(all_activations.keys()):
        acts = all_activations[layer_idx]
        probe, scaler, auroc, acc, fpr, fold_aurocs = train_probe_at_layer(
            acts, labels, n_folds=n_folds, seed=seed
        )
        results[layer_idx] = ProbeResult(
            layer=layer_idx,
            auroc=auroc,
            auroc_std=float(np.std(fold_aurocs)),
            accuracy=acc,
            fpr=fpr,
            fold_aurocs=fold_aurocs,
        )
        layer_probes[layer_idx] = (probe, scaler)
        print(f"  Layer {layer_idx:3d}: AUROC={auroc:.4f} (±{np.std(fold_aurocs):.4f}), "
              f"Acc={acc:.4f}, FPR={fpr:.4f}")

    # Select best layer with robust tiebreaking.
    # When multiple layers achieve ceiling AUROC (within 0.005), the first-past
    # selection is fragile across hardware (MPS vs CUDA produce slightly different
    # floating point, leading to different layer choices). We break ties by:
    #   1. Lowest FPR (fewer false positives on deployment benign queries)
    #   2. Highest accuracy (better separation at the 0.5 decision boundary)
    #   3. Earliest layer (fewer parameters activated = faster, more interpretable)
    best_layer = max(
        results.keys(),
        key=lambda l: (
            results[l].auroc,         # Primary: highest AUROC
            -results[l].fpr,          # Tiebreak 1: lowest FPR (negate for max)
            results[l].accuracy,      # Tiebreak 2: highest accuracy
            -l,                       # Tiebreak 3: latest layer with best metrics
        ),
    )
    best_probe, best_scaler = layer_probes[best_layer]

    return results, best_layer, best_probe, best_scaler


def save_probe_results(
    output_dir: Path,
    model_name: str,
    probe_type: str,
    results: dict[int, ProbeResult],
    best_layer: int,
    best_probe: LogisticRegression,
    best_scaler: StandardScaler,
) -> None:
    """Save probe training results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Layer sweep results
    sweep_data = {
        layer: {
            "layer": r.layer,
            "auroc": r.auroc,
            "auroc_std": r.auroc_std,
            "accuracy": r.accuracy,
            "fpr": r.fpr,
            "fold_aurocs": r.fold_aurocs,
        }
        for layer, r in results.items()
    }
    with open(output_dir / "layer_sweep.json", "w") as f:
        json.dump(sweep_data, f, indent=2)

    # Best layer metadata
    best = results[best_layer]
    best_info = {
        "layer": best_layer,
        "auroc": best.auroc,
        "auroc_std": best.auroc_std,
        "accuracy": best.accuracy,
        "fpr": best.fpr,
        "model": model_name,
        "probe_type": probe_type,
    }
    with open(output_dir / "best_layer.json", "w") as f:
        json.dump(best_info, f, indent=2)

    # Save trained probe and scaler
    with open(output_dir / "best_probe.pkl", "wb") as f:
        pickle.dump({"probe": best_probe, "scaler": best_scaler}, f)

    print(f"  Saved results to {output_dir}")
