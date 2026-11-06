#!/bin/bash
set -euo pipefail

# ============================================================================
# GPU Setup Script — Fresh A100 Instance
# Run: bash scripts/gpu_setup.sh
# Time: ~15 minutes
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

LOG_FILE="$PROJECT_ROOT/results/gpu_setup.log"
mkdir -p results
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "GPU SETUP — $(date)"
echo "Project root: $PROJECT_ROOT"
echo "============================================================"

# --- Step 1: System check ---
echo ""
echo "[1/7] Checking system..."
if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Is this a GPU instance?"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "CUDA version: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'nvcc not found')"
python3 --version

# --- Step 2: Install Python dependencies ---
echo ""
echo "[2/7] Installing Python dependencies..."
pip install -e ".[dev]" --quiet 2>&1 | tail -5
pip install sentencepiece protobuf nanogcg --quiet 2>&1 | tail -3

# Verify torch sees CUDA
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available to PyTorch!'
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# --- Step 3: Clone and install AlphaSteer ---
echo ""
echo "[3/7] Setting up AlphaSteer baseline..."
mkdir -p external
if [ ! -d "external/AlphaSteer" ]; then
    git clone https://github.com/AlphaLab-USTC/AlphaSteer.git external/AlphaSteer
    cd external/AlphaSteer
    pip install -e . --quiet 2>&1 | tail -3 || echo "WARNING: AlphaSteer pip install failed — may need manual setup"
    cd "$PROJECT_ROOT"
else
    echo "  AlphaSteer already cloned"
fi

# Check for pre-computed matrices
if [ -d "external/AlphaSteer/data/steering_matrix" ]; then
    echo "  Pre-computed AlphaSteer matrices found"
    ls external/AlphaSteer/data/steering_matrix/
else
    echo "  No pre-computed matrices — will train from scratch (Queue 1)"
fi

# --- Step 4: Download models ---
echo ""
echo "[4/7] Pre-downloading HuggingFace models..."
python3 -c "
from huggingface_hub import snapshot_download
import os

models = [
    'mistralai/Mistral-7B-Instruct-v0.3',
    'google/gemma-2-9b-it',
    'Qwen/Qwen2.5-7B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct',
]

for model_id in models:
    name = model_id.split('/')[-1]
    print(f'  Downloading {name}...')
    try:
        snapshot_download(model_id, ignore_patterns=['*.gguf', '*.ggml'])
        print(f'    OK')
    except Exception as e:
        print(f'    WARNING: {e}')
        print(f'    May need to accept license at https://huggingface.co/{model_id}')
"

# --- Step 5: Ensure datasets exist ---
echo ""
echo "[5/7] Checking datasets..."
if [ ! -f "data/contrast_pairs/jailbreak_pairs.json" ]; then
    echo "  Generating contrast pairs..."
    python3 scripts/02_generate_contrast_pairs.py --probe_type jailbreak --n_pairs 40
else
    echo "  Contrast pairs found"
fi

# Verify attack data
python3 -c "
from src.attacks.registry import load_attack_set
from src.eval.openended_eval import get_all_queries

attacks = load_attack_set('prefilling', n_attacks=20, include_benign=True)
print(f'  Attack set: {len(attacks)} prompts')
queries = get_all_queries()
print(f'  Open-ended queries: {len(queries)} prompts')
"

# --- Step 6: Smoke test ---
echo ""
echo "[6/7] Running smoke test (1 prompt, GPU inference)..."
python3 -c "
import torch
from src.models.loader import load_model

model_name = 'mistral-7b'
print(f'  Loading {model_name} on CUDA...')
model, tokenizer, config = load_model(model_name, device='cuda')

messages = [{'role': 'user', 'content': 'Say hello in one word.'}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors='pt').to('cuda')

import time
t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
elapsed = time.time() - t0

response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f'  Response: {response.strip()}')
print(f'  Latency: {elapsed:.2f}s')
print(f'  SMOKE TEST PASSED')
del model
torch.cuda.empty_cache()
"

# --- Step 7: Print run plan ---
echo ""
echo "[7/7] Run plan and cost estimate..."
echo ""
echo "============================================================"
echo "  SETUP COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "  Run plan (sequential, resumable):"
echo "  Queue 1: AlphaSteer on Mistral-7B        ~3-5 hrs   CRITICAL"
echo "  Queue 2: AlphaSteer on 3 remaining models ~6-9 hrs"
echo "  Queue 3: CAST baseline on all 4 models    ~2-3 hrs"
echo "  Queue 4: GCG suffix generation            ~8-12 hrs"
echo "  Queue 5: Expanded eval with GCG           ~4-6 hrs"
echo "  Queue 6: Ablation study                   ~2-3 hrs"
echo ""
echo "  Estimated total: 25-38 hours"
echo "  Cost (spot \$1.10/hr):  \$28-42"
echo "  Cost (on-demand \$3.67/hr): \$92-139"
echo ""
echo "  Start with: bash scripts/gpu_run_all.sh"
echo "  Monitor with: bash scripts/gpu_monitor.sh"
echo "============================================================"
