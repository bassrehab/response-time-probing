"""Truncated Polynomial Classifier (TPC) probe training.

Based on "Beyond Linear Probes: Dynamic Safety Monitoring for Language Models"
(Oldfield et al., ICLR 2026).

TPCs extend linear probes with polynomial features:
  degree 1: linear probe (same as LogisticRegression)
  degree 2: linear + quadratic interactions via CP factorization
  degree 3+: higher-order (not implemented — diminishing returns)

The CP factorization makes degree-2 tractable: instead of storing D×D weights,
we use rank-R factors: W² = Σ λᵣ · uᵣ ⊗ uᵣ, so parameters scale as O(R·D)
instead of O(D²).

For our use case, we compare degree-2 TPC against our linear probe at the
best jailbreak probe layer. Two outcomes:
  - If marginal AUROC improvement (<2%): "linear probes suffice" → keep simple
  - If significant improvement (>5%): upgrade probe → re-run experiments
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
class TPCResult:
    """Result of training a TPC probe."""

    degree: int
    rank: int
    auroc: float
    auroc_std: float
    accuracy: float
    fpr: float
    fold_aurocs: list[float]
    n_features: int  # Total features after polynomial expansion


def _polynomial_features_cp(X: np.ndarray, rank: int, seed: int = 42) -> np.ndarray:
    """Compute approximate degree-2 polynomial features via CP factorization.

    Instead of computing all D×D quadratic interactions (O(D²) features),
    we project onto R random directions and square them: (X @ u_r)².
    This gives R additional features capturing quadratic structure.

    Combined feature vector: [X, (X @ u₁)², (X @ u₂)², ..., (X @ uᵣ)²]
    Total features: D + R

    Args:
        X: (n_samples, D) input activations.
        rank: Number of random projection directions.
        seed: Random seed for projection directions.

    Returns:
        (n_samples, D + rank) feature matrix.
    """
    rng = np.random.RandomState(seed)
    D = X.shape[1]

    # Random projection directions (unit vectors)
    U = rng.randn(D, rank)
    U = U / np.linalg.norm(U, axis=0, keepdims=True)

    # Quadratic features: (X @ u_r)² for each direction
    projections = X @ U  # (n_samples, rank)
    quad_features = projections ** 2  # (n_samples, rank)

    return np.hstack([X, quad_features])


def train_tpc_at_layer(
    activations: np.ndarray,
    labels: np.ndarray,
    degree: int = 2,
    rank: int = 64,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[LogisticRegression, StandardScaler, np.ndarray, TPCResult]:
    """Train a TPC probe with cross-validation.

    Args:
        activations: (n_samples, hidden_dim) activations.
        labels: (n_samples,) binary labels.
        degree: Polynomial degree (1 = linear, 2 = quadratic).
        rank: CP rank for quadratic features.
        n_folds: CV folds.
        seed: Random seed.

    Returns:
        (probe, scaler, projection_matrix, result)
    """
    if degree == 1:
        # Just a linear probe — no polynomial features
        features = activations
        U = None
    elif degree == 2:
        features = _polynomial_features_cp(activations, rank, seed)
        # Store projection matrix for inference
        rng = np.random.RandomState(seed)
        D = activations.shape[1]
        U = rng.randn(D, rank)
        U = U / np.linalg.norm(U, axis=0, keepdims=True)
    else:
        raise ValueError(f"Degree {degree} not implemented. Use 1 or 2.")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aurocs = []
    fold_accs = []
    fold_fprs = []

    for train_idx, val_idx in skf.split(features, labels):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        probe = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed, C=1.0)
        probe.fit(X_train, y_train)

        probs = probe.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)

        auroc = roc_auc_score(y_val, probs)
        acc = (preds == y_val).mean()
        neg_mask = y_val == 0
        fpr = preds[neg_mask].mean() if neg_mask.sum() > 0 else 0.0

        fold_aurocs.append(auroc)
        fold_accs.append(acc)
        fold_fprs.append(fpr)

    # Train final on all data
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(features)
    probe_final = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed, C=1.0)
    probe_final.fit(X_scaled, labels)

    result = TPCResult(
        degree=degree,
        rank=rank if degree >= 2 else 0,
        auroc=float(np.mean(fold_aurocs)),
        auroc_std=float(np.std(fold_aurocs)),
        accuracy=float(np.mean(fold_accs)),
        fpr=float(np.mean(fold_fprs)),
        fold_aurocs=[float(a) for a in fold_aurocs],
        n_features=features.shape[1],
    )

    return probe_final, scaler_final, U, result


def compare_linear_vs_tpc(
    activations: np.ndarray,
    labels: np.ndarray,
    ranks: list[int] | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Compare linear probe vs TPC at various ranks.

    Returns dict with comparison results for reporting.
    """
    if ranks is None:
        ranks = [16, 32, 64, 128]

    results = {}

    # Linear baseline (degree 1)
    _, _, _, linear_result = train_tpc_at_layer(
        activations, labels, degree=1, n_folds=n_folds, seed=seed
    )
    results["linear"] = {
        "auroc": linear_result.auroc,
        "auroc_std": linear_result.auroc_std,
        "accuracy": linear_result.accuracy,
        "fpr": linear_result.fpr,
        "n_features": linear_result.n_features,
    }
    print(f"  Linear probe:     AUROC={linear_result.auroc:.4f} (±{linear_result.auroc_std:.4f}), "
          f"features={linear_result.n_features}")

    # TPC degree-2 at various ranks
    for rank in ranks:
        _, _, _, tpc_result = train_tpc_at_layer(
            activations, labels, degree=2, rank=rank, n_folds=n_folds, seed=seed
        )
        results[f"tpc_d2_r{rank}"] = {
            "auroc": tpc_result.auroc,
            "auroc_std": tpc_result.auroc_std,
            "accuracy": tpc_result.accuracy,
            "fpr": tpc_result.fpr,
            "n_features": tpc_result.n_features,
        }
        delta = (tpc_result.auroc - linear_result.auroc) * 100
        print(f"  TPC d=2 r={rank:3d}:   AUROC={tpc_result.auroc:.4f} (±{tpc_result.auroc_std:.4f}), "
              f"features={tpc_result.n_features}, Δ={delta:+.2f}pp")

    # Summary
    best_tpc_key = max(
        [k for k in results if k.startswith("tpc_")],
        key=lambda k: results[k]["auroc"],
    )
    best_tpc = results[best_tpc_key]
    improvement = (best_tpc["auroc"] - results["linear"]["auroc"]) * 100

    results["summary"] = {
        "best_tpc": best_tpc_key,
        "improvement_pp": improvement,
        "recommendation": (
            "upgrade_to_tpc" if improvement > 5.0
            else "marginal_tpc" if improvement > 2.0
            else "linear_sufficient"
        ),
    }

    return results
