"""Static CAA steering — always-on, no gating.

Injects the steering vector at every forward pass regardless of input.
This is the simplest activation steering defense and serves as a key
comparison point: probe-gated should match or exceed this on safety
while having better utility (no steering on benign inputs).
"""

import torch
from torch import nn

from src.models.hooks import ActivationHook
from src.steering.apply_steering import SteeringHook


class StaticSteering:
    """Always-on steering vector injection.

    Unlike probe-gated steering, this applies the steering vector at constant
    strength to every generation, regardless of whether the input is harmful.
    Expected tradeoff: better safety than no defense, but worse utility
    (steers benign prompts too).
    """

    def __init__(
        self,
        model: nn.Module,
        steering_vector_path: str,
        steering_layer: int,
        strength: float = 1.0,
    ) -> None:
        self.vector = torch.load(steering_vector_path, weights_only=True)
        self.hook = SteeringHook(model, steering_layer, self.vector, strength=strength)
        self.strength = strength

    def set_strength(self, strength: float) -> None:
        self.strength = strength
        self.hook.set_strength(strength)

    def remove(self) -> None:
        self.hook.remove()
