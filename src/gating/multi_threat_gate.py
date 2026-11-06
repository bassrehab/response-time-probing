"""Multi-threat probe-gated steering.

Composes multiple probe-gate-steer triples to defend against several
threat types simultaneously (e.g., jailbreak + sycophancy + sandbagging).

Three composition strategies:
  - "naive": Stack vectors without modification. Expected to fail when
    vectors share variance (confirmed by prior cross-model results).
  - "orthogonal": Gram-Schmidt orthogonalize vectors before injection.
    All vectors injected at the same layer. Removes cross-interference.
  - "layer_separated": Each behavior uses a different steering layer.
    No orthogonalization needed since vectors operate independently.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.gating.controllers import BangBangController, ProportionalController
from src.models.hooks import ActivationHook
from src.steering.multi_vector import orthogonalize_vectors, compute_cosine_matrix


@dataclass
class ThreatConfig:
    """Configuration for a single threat in a multi-threat system."""

    name: str                     # e.g., "jailbreak", "sycophancy"
    probe_path: Path              # Path to best_probe.pkl
    steering_vector_path: Path    # Path to vector_layer{N}.pt
    probe_layer: int
    steering_layer: int
    controller_type: str = "proportional"
    threshold: float = 0.5
    max_strength: float = 1.0
    gain: float = 2.0


class MultiThreatGate:
    """Compose multiple probe-gate-steer triples for multi-behavior defense.

    Each threat has its own probe (detecting that specific behavior) and
    steering vector (correcting that specific behavior). The composition
    strategy determines how multiple steering vectors interact.
    """

    def __init__(
        self,
        model: nn.Module,
        configs: list[ThreatConfig],
        strategy: str = "orthogonal",
    ) -> None:
        self.model = model
        self.strategy = strategy
        self.configs = configs
        self._threats: list[_ThreatInstance] = []
        self._handles: list = []

        # Load all probes and vectors
        probes = []
        scalers = []
        vectors = []
        controllers = []

        for cfg in configs:
            with open(cfg.probe_path, "rb") as f:
                data = pickle.load(f)
            probes.append(data["probe"])
            scalers.append(data["scaler"])
            vectors.append(torch.load(cfg.steering_vector_path, weights_only=True))

            if cfg.controller_type == "bang_bang":
                controllers.append(BangBangController(
                    threshold=cfg.threshold, on_strength=cfg.max_strength
                ))
            else:
                controllers.append(ProportionalController(
                    threshold=cfg.threshold, max_strength=cfg.max_strength,
                    gain=cfg.gain
                ))

        # Apply composition strategy to vectors
        if strategy == "orthogonal":
            print(f"  Pre-orthogonalization cosine matrix:")
            cos_before = compute_cosine_matrix(vectors)
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    print(f"    {configs[i].name} × {configs[j].name}: "
                          f"{cos_before[i, j]:.4f}")

            vectors = orthogonalize_vectors(vectors)

            print(f"  Post-orthogonalization cosine matrix:")
            cos_after = compute_cosine_matrix(vectors)
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    print(f"    {configs[i].name} × {configs[j].name}: "
                          f"{cos_after[i, j]:.4f}")

        elif strategy == "layer_separated":
            # Spread steering layers across model depth
            total_layers = len(ActivationHook._resolve_layer(model, 0).weight
                               if hasattr(ActivationHook._resolve_layer(model, 0), 'weight')
                               else [None] * 32)  # fallback
            # Use the model's actual layers
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                total_layers = len(model.model.layers)
            else:
                total_layers = 32

            spread = np.linspace(
                total_layers * 0.3, total_layers * 0.7,
                len(configs), dtype=int
            )
            for i, cfg in enumerate(configs):
                cfg.steering_layer = int(spread[i])
            print(f"  Layer-separated steering layers: "
                  f"{[cfg.steering_layer for cfg in configs]}")

        elif strategy == "naive":
            print("  Using naive vector stacking (no orthogonalization)")

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Build threat instances and register hooks
        for i, cfg in enumerate(configs):
            vec = vectors[i] if i < len(vectors) else None
            if vec is None:
                print(f"  Warning: vector for {cfg.name} was removed during "
                      f"orthogonalization (near-parallel), skipping.")
                continue

            threat = _ThreatInstance(
                name=cfg.name,
                probe=probes[i],
                scaler=scalers[i],
                vector=vec,
                controller=controllers[i],
                probe_layer=cfg.probe_layer,
                steering_layer=cfg.steering_layer,
            )
            self._threats.append(threat)

        # Register hooks
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register probe and steering hooks for all threats."""
        for threat in self._threats:
            # Probe hook
            probe_module = ActivationHook._resolve_layer(self.model, threat.probe_layer)
            handle = probe_module.register_forward_hook(threat.make_probe_hook())
            self._handles.append(handle)

            # Steering hook
            steer_module = ActivationHook._resolve_layer(self.model, threat.steering_layer)
            handle = steer_module.register_forward_hook(threat.make_steer_hook())
            self._handles.append(handle)

    def reset(self) -> None:
        """Reset all threat states between generations."""
        for threat in self._threats:
            threat.reset()

    def get_scores(self) -> dict[str, float]:
        """Get current probe scores for all threats."""
        return {t.name: t.last_score for t in self._threats}

    def get_strengths(self) -> dict[str, float]:
        """Get current steering strengths for all threats."""
        return {t.name: t.current_strength for t in self._threats}

    def get_score_histories(self) -> dict[str, list[float]]:
        """Get full score histories for all threats."""
        return {t.name: list(t.score_history) for t in self._threats}

    def remove(self) -> None:
        """Remove all hooks from the model."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._threats.clear()


@dataclass
class _ThreatInstance:
    """Internal state for a single threat in the multi-threat system."""

    name: str
    probe: LogisticRegression
    scaler: StandardScaler
    vector: torch.Tensor
    controller: object  # BangBangController or ProportionalController
    probe_layer: int
    steering_layer: int
    last_score: float = 0.0
    current_strength: float = 0.0
    score_history: list = None

    def __post_init__(self):
        if self.score_history is None:
            self.score_history = []

    def make_probe_hook(self):
        """Return a forward hook function for probing."""
        threat = self

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            activation = hidden[:, -1, :].detach().cpu().float().numpy()
            activation_scaled = threat.scaler.transform(activation)
            score = float(threat.probe.predict_proba(activation_scaled)[0, 1])
            threat.last_score = score
            threat.score_history.append(score)
            threat.current_strength = threat.controller(score)

        return hook_fn

    def make_steer_hook(self):
        """Return a forward hook function for steering."""
        threat = self

        def hook_fn(module, input, output):
            if threat.current_strength == 0.0:
                return output
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            sv = threat.vector.to(hidden.device, dtype=hidden.dtype)
            hidden[:, -1, :] += threat.current_strength * sv
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        return hook_fn

    def reset(self):
        self.last_score = 0.0
        self.current_strength = 0.0
        self.score_history = []
