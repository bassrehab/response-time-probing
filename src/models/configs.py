"""Per-model configurations for probing and steering experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    hf_id: str
    ollama_id: str
    num_layers: int
    hidden_dim: int
    optimal_probe_layer: int
    optimal_steering_layer: int
    steering_strength_default: float
    dtype: str
    max_seq_len: int


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "mistral-7b": ModelConfig(
        hf_id="mistralai/Mistral-7B-Instruct-v0.3",
        ollama_id="mistral:7b-instruct-v0.3-q8_0",
        num_layers=32,
        hidden_dim=4096,
        optimal_probe_layer=30,
        optimal_steering_layer=14,
        steering_strength_default=0.8,
        dtype="bfloat16",
        max_seq_len=4096,
    ),
    "gemma2-9b": ModelConfig(
        hf_id="google/gemma-2-9b-it",
        ollama_id="gemma2:9b-instruct-q8_0",
        num_layers=42,
        hidden_dim=3584,
        optimal_probe_layer=4,
        optimal_steering_layer=14,
        steering_strength_default=0.7,
        dtype="bfloat16",
        max_seq_len=4096,
    ),
    "qwen25-7b": ModelConfig(
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        ollama_id="qwen2.5:7b-instruct-q8_0",
        num_layers=28,
        hidden_dim=3584,
        optimal_probe_layer=12,
        optimal_steering_layer=14,
        steering_strength_default=0.8,
        dtype="bfloat16",
        max_seq_len=4096,
    ),
    "llama31-8b": ModelConfig(
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        ollama_id="llama3.1:8b-instruct-q8_0",
        num_layers=32,
        hidden_dim=4096,
        optimal_probe_layer=16,
        optimal_steering_layer=14,
        steering_strength_default=0.8,
        dtype="bfloat16",
        max_seq_len=4096,
    ),
    "qwen3-8b": ModelConfig(
        hf_id="Qwen/Qwen3-8B",
        ollama_id="qwen3:8b-instruct-q8_0",
        num_layers=36,
        hidden_dim=4096,
        optimal_probe_layer=1,
        optimal_steering_layer=14,
        steering_strength_default=0.8,
        dtype="bfloat16",
        max_seq_len=4096,
    ),
    "gemma3-12b": ModelConfig(
        hf_id="google/gemma-3-12b-it",
        ollama_id="gemma3:12b-instruct-q8_0",
        num_layers=48,
        hidden_dim=3840,
        optimal_probe_layer=4,
        optimal_steering_layer=14,
        steering_strength_default=0.7,
        dtype="bfloat16",
        max_seq_len=4096,
    ),
    "gemma4-31b": ModelConfig(
        hf_id="google/gemma-4-31b-it",
        ollama_id="gemma4:31b-instruct-q8_0",
        num_layers=64,
        hidden_dim=5376,
        optimal_probe_layer=1,
        optimal_steering_layer=14,
        steering_strength_default=0.7,
        dtype="bfloat16",
        max_seq_len=4096,
    ),
}


def get_config(model_name: str) -> ModelConfig:
    """Get model config by short name. Raises KeyError if not found."""
    if model_name not in MODEL_CONFIGS:
        raise KeyError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_name]
