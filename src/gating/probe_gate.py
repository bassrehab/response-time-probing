"""Core probe-gated steering system.

Probe detects → Controller modulates → Steering corrects.

Flow per forward pass:
  1. Extract activation at probe_layer
  2. Run probe → get score in [0, 1] (P(unsafe))
  3. Controller maps score → steering_strength
  4. Inject steering_vector * strength at steering_layer

TIMING NOTE — READ CAREFULLY:
During a single forward pass, transformer layers execute sequentially (0→N).

If probe_layer < steering_layer (e.g., Mistral: probe@6, steer@14), the
probe hook fires FIRST, then the steering hook fires later in the same pass.
This gives us "same_token" gating — probe detects, then steering corrects
within the same forward pass. This is the ideal configuration.

If probe_layer > steering_layer (e.g., hypothetical probe@30, steer@14),
the steering hook fires BEFORE the probe hook. This means we steer based on
the PREVIOUS token's probe score ("next_token" mode). For autoregressive
generation this is acceptable — the probe detects the threat from prompt
tokens before generation starts, and subsequent tokens are steered based
on the prior token's score.

For the paper: report which timing mode each model uses. All current configs
use same_token (probe@6-7, steer@14).
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn, Tensor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.gating.controllers import BangBangController, ProportionalController
from src.models.hooks import ActivationHook


@dataclass
class GateConfig:
    """Configuration for a probe-gated steering instance."""

    probe_path: Path              # Path to best_probe.pkl
    steering_vector_path: Path    # Path to vector_layer{N}.pt
    probe_layer: int              # Layer to extract activations for probing
    steering_layer: int           # Layer to inject steering vector
    controller_type: str = "proportional"  # "bang_bang" | "proportional"
    threshold: float = 0.5
    max_strength: float = 1.0
    gain: float = 2.0             # For proportional controller


class ProbeGatedSteering:
    """Core system: probe detects → controller modulates → steering corrects.

    Registers forward hooks on the model for both probing and steering.
    During generation, the probe continuously monitors activations and the
    controller dynamically adjusts steering strength.
    """

    def __init__(self, model: nn.Module, config: GateConfig) -> None:
        self.model = model
        self.config = config

        # Load probe + scaler
        with open(config.probe_path, "rb") as f:
            probe_data = pickle.load(f)
        self.probe: LogisticRegression = probe_data["probe"]
        self.scaler: StandardScaler = probe_data["scaler"]

        # Load steering vector
        self.steering_vector: Tensor = torch.load(
            config.steering_vector_path, weights_only=True
        )

        # Set up controller
        if config.controller_type == "bang_bang":
            self.controller = BangBangController(
                threshold=config.threshold,
                on_strength=config.max_strength,
            )
        elif config.controller_type == "proportional":
            self.controller = ProportionalController(
                threshold=config.threshold,
                max_strength=config.max_strength,
                gain=config.gain,
            )
        else:
            raise ValueError(f"Unknown controller type: {config.controller_type}")

        # State
        self._current_strength: float = 0.0
        self._last_score: float = 0.0
        self._score_history: list[float] = []

        # Register hooks
        self._register_hooks()

        # Determine timing mode based on layer ordering
        # same_token: probe fires first (probe_layer < steering_layer) — ideal
        # next_token: steer fires first (probe_layer > steering_layer) — one token lag
        if config.probe_layer < config.steering_layer:
            self._timing_mode = "same_token"
        else:
            self._timing_mode = "next_token"
        print(f"  ProbeGatedSteering: probe@{config.probe_layer} → steer@{config.steering_layer} "
              f"(timing: {self._timing_mode})")

    def _register_hooks(self) -> None:
        probe_layer = ActivationHook._resolve_layer(self.model, self.config.probe_layer)
        self._probe_handle = probe_layer.register_forward_hook(self._probe_hook)

        steer_layer = ActivationHook._resolve_layer(self.model, self.config.steering_layer)
        self._steer_handle = steer_layer.register_forward_hook(self._steer_hook)

    def _probe_hook(self, module: nn.Module, input: tuple, output: tuple | Tensor) -> None:
        """Extract activation, run probe, update steering strength.

        In same_token mode (probe_layer < steering_layer), this fires before
        _steer_hook, so the steering strength is updated before injection.
        In next_token mode (probe_layer > steering_layer), this fires after
        _steer_hook, so steering uses the previous token's probe score.
        """
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Last token activation
        activation = hidden[:, -1, :].detach().cpu().float().numpy()

        # Scale and predict
        activation_scaled = self.scaler.transform(activation)
        score = float(self.probe.predict_proba(activation_scaled)[0, 1])

        self._last_score = score
        self._score_history.append(score)
        self._current_strength = self.controller(score)

    def _steer_hook(self, module: nn.Module, input: tuple, output: tuple | Tensor) -> tuple | Tensor:
        """Inject steering vector scaled by controller output."""
        if self._current_strength == 0.0:
            return output

        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        sv = self.steering_vector.to(hidden.device, dtype=hidden.dtype)
        hidden[:, -1, :] += self._current_strength * sv

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    @property
    def last_score(self) -> float:
        """Most recent probe score."""
        return self._last_score

    @property
    def current_strength(self) -> float:
        """Current steering strength."""
        return self._current_strength

    @property
    def score_history(self) -> list[float]:
        """Full history of probe scores during generation."""
        return self._score_history

    @property
    def timing_mode(self) -> str:
        """Whether gating is same-token or next-token."""
        return self._timing_mode

    def reset(self) -> None:
        """Reset state between generations."""
        self._current_strength = 0.0
        self._last_score = 0.0
        self._score_history = []

    def remove(self) -> None:
        """Remove all hooks from the model."""
        self._probe_handle.remove()
        self._steer_handle.remove()
