"""Hook-based activation steering injection."""

import torch
from torch import nn, Tensor

from src.models.hooks import ActivationHook


class SteeringHook:
    """Injects a steering vector into a model's residual stream at a specified layer.

    The vector is scaled by `strength` at injection time. Modifies the last token's
    activation in-place during the forward pass.

    Usage:
        hook = SteeringHook(model, layer_idx=14, vector=sv, strength=0.8)
        model.generate(...)  # steering is applied during generation
        hook.set_strength(0.0)  # disable
        hook.remove()
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        vector: Tensor,
        strength: float = 1.0,
    ) -> None:
        self.layer_idx = layer_idx
        self.vector = vector
        self.strength = strength

        layer = ActivationHook._resolve_layer(model, layer_idx)
        self._handle = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module: nn.Module, input: tuple, output: tuple | Tensor) -> tuple | Tensor:
        if self.strength == 0.0:
            return output

        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        sv = self.vector.to(hidden.device, dtype=hidden.dtype)
        hidden[:, -1, :] += self.strength * sv

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def set_strength(self, strength: float) -> None:
        self.strength = strength

    def remove(self) -> None:
        self._handle.remove()
