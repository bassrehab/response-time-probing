"""Unified model loading for HuggingFace models.

Loads models in bfloat16 with automatic device placement (MPS / CUDA / CPU).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

from src.models.configs import ModelConfig, get_config

# transformers >= 4.48 uses 'dtype', older uses 'torch_dtype'
_DTYPE_KWARG = "dtype" if hasattr(transformers, "__version__") and tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 48) else "torch_dtype"


def get_device() -> str:
    """Detect best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(
    model_name: str,
    *,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, ModelConfig]:
    """Load a HuggingFace model and tokenizer.

    Args:
        model_name: Short name (e.g. "mistral-7b") matching MODEL_CONFIGS keys.
        device: Override device. Auto-detected if None.
        dtype: Model dtype. Defaults to bfloat16.

    Returns:
        (model, tokenizer, config) tuple.
    """
    config = get_config(model_name)
    device = device or get_device()

    tokenizer = AutoTokenizer.from_pretrained(config.hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # MPS doesn't support device_map="auto", so load to CPU then move
    if device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_id,
            **{_DTYPE_KWARG: dtype},
            low_cpu_mem_usage=True,
        ).to(device)
    elif device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_id,
            **{_DTYPE_KWARG: dtype},
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_id,
            **{_DTYPE_KWARG: dtype},
            low_cpu_mem_usage=True,
        )

    model.eval()
    return model, tokenizer, config
