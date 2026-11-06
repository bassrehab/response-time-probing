"""Activation extraction hooks for transformer models.

Captures residual stream activations at any specified layer during forward passes.
"""

import torch
from torch import nn, Tensor


class ActivationHook:
    """Captures residual stream activations at a specified transformer layer.

    Usage:
        hook = ActivationHook(model, layer_idx=14)
        with torch.no_grad():
            model(**inputs)
        activations = hook.get()       # (batch, seq_len, hidden_dim)
        last_token = hook.get_last()   # (batch, hidden_dim)
        hook.remove()
    """

    def __init__(self, model: nn.Module, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        self._activation: Tensor | None = None
        layer = self._resolve_layer(model, layer_idx)
        self._handle = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module: nn.Module, input: tuple, output: tuple | Tensor) -> None:
        # Most HF models return a tuple; the hidden state is the first element.
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        self._activation = hidden.detach()

    def get(self) -> Tensor:
        """Return full activation tensor (batch, seq_len, hidden_dim)."""
        if self._activation is None:
            raise RuntimeError("No activation captured yet. Run a forward pass first.")
        return self._activation

    def get_last(self) -> Tensor:
        """Return last-token activation (batch, hidden_dim)."""
        return self.get()[:, -1, :]

    def clear(self) -> None:
        self._activation = None

    def remove(self) -> None:
        """Remove the hook from the model."""
        self._handle.remove()
        self._activation = None

    @staticmethod
    def _resolve_layer(model: nn.Module, layer_idx: int) -> nn.Module:
        """Resolve a transformer layer by index across architectures."""
        # HF models wrap the transformer in model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
            # Gemma 3+ multimodal: model.model.language_model.layers
            layers = model.model.language_model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
        else:
            raise ValueError(
                f"Cannot resolve layers for {type(model).__name__}. "
                "Expected model.model.layers, model.model.language_model.layers, or model.transformer.h"
            )
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(
                f"Layer {layer_idx} out of range [0, {len(layers)}) for {type(model).__name__}"
            )
        return layers[layer_idx]


class MultiLayerHook:
    """Captures activations at multiple layers simultaneously.

    Usage:
        hook = MultiLayerHook(model, layer_indices=[4, 14, 30])
        with torch.no_grad():
            model(**inputs)
        acts = hook.get_all()  # dict[int, Tensor]
        hook.remove()
    """

    def __init__(self, model: nn.Module, layer_indices: list[int]) -> None:
        self._hooks = {idx: ActivationHook(model, idx) for idx in layer_indices}

    def get_all(self) -> dict[int, Tensor]:
        """Return {layer_idx: activation_tensor} for all hooked layers."""
        return {idx: hook.get() for idx, hook in self._hooks.items()}

    def get_all_last(self) -> dict[int, Tensor]:
        """Return {layer_idx: last_token_activation} for all hooked layers."""
        return {idx: hook.get_last() for idx, hook in self._hooks.items()}

    def clear(self) -> None:
        for hook in self._hooks.values():
            hook.clear()

    def remove(self) -> None:
        for hook in self._hooks.values():
            hook.remove()
        self._hooks.clear()
