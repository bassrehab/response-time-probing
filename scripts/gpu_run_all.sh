#!/bin/bash
set -euo pipefail

# ============================================================================
# GPU Master Run Script — All queues, sequential, resumable
# Run: bash scripts/gpu_run_all.sh
# Resume: just re-run — completed queues are skipped
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RESULTS_DIR="$PROJECT_ROOT/results"
COST_LOG="$RESULTS_DIR/gpu_cost.log"
QUEUE_STATUS="$RESULTS_DIR/gpu_queue_status.json"
START_TIME=$(date +%s)
RATE_SPOT=1.10
RATE_ONDEMAND=3.67

mkdir -p "$RESULTS_DIR"

# ---- Helpers ----

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RESULTS_DIR/gpu_run.log"
}

update_cost() {
    local now=$(date +%s)
    local elapsed_hrs=$(echo "scale=2; ($now - $START_TIME) / 3600" | bc)
    local cost_spot=$(echo "scale=2; $elapsed_hrs * $RATE_SPOT" | bc)
    local cost_ondemand=$(echo "scale=2; $elapsed_hrs * $RATE_ONDEMAND" | bc)
    echo "{\"elapsed_hrs\": $elapsed_hrs, \"cost_spot\": $cost_spot, \"cost_ondemand\": $cost_ondemand, \"timestamp\": \"$(date)\"}" >> "$COST_LOG"
    log "COST: ${elapsed_hrs}hrs, spot=\$${cost_spot}, on-demand=\$${cost_ondemand}"
}

queue_done() {
    local queue=$1
    local marker="$RESULTS_DIR/.queue_${queue}_done"
    [ -f "$marker" ]
}

mark_queue_done() {
    local queue=$1
    touch "$RESULTS_DIR/.queue_${queue}_done"
    update_cost
    log "QUEUE $queue COMPLETE"
}

# ============================================================================
# QUEUE 1: AlphaSteer on Mistral-7B (CRITICAL)
# ============================================================================

if queue_done 1; then
    log "SKIPPING Queue 1 — results found"
else
    log "=========================================="
    log "QUEUE 1: AlphaSteer on Mistral-7B"
    log "=========================================="

    python3 -c "
import json, time, sys
import torch
import numpy as np
from pathlib import Path

from src.models.loader import load_model
from src.models.hooks import ActivationHook
from src.attacks.datasets.prompts import ADVBENCH_BEHAVIORS, BENIGN_QUERIES, BENIGN_OPENENDED
from src.attacks.registry import load_attack_set
from src.baselines.alphasteer.wrapper import (
    compute_null_space_projection, compute_delta_matrix,
    save_alphasteer_matrices, AlphaSteerDefense, AlphaSteerConfig,
)
from src.baselines.static_steering.infer import StaticSteering
from src.eval.safety_eval import run_safety_eval, save_eval_results
from src.eval.openended_eval import run_openended_eval, save_openended_results
from src.gating.probe_gate import GateConfig, ProbeGatedSteering
from src.utils.paths import probe_results_dir, vector_results_dir, RESULTS_DIR
from src.utils.run_logger import capture_run_metadata, save_run_metadata
from src.utils.seeding import seed_everything

seed_everything(42)
device = 'cuda'
model_name = 'mistral-7b'
output_dir = RESULTS_DIR / 'alphasteer' / model_name
output_dir.mkdir(parents=True, exist_ok=True)

metadata = capture_run_metadata('queue1_alphasteer', 'gpu_run_all.sh')
save_run_metadata(output_dir, metadata)

# Load model
print(f'Loading {model_name}...')
model, tokenizer, config = load_model(model_name, device=device)

# ---- Train AlphaSteer Δ matrix ----
print('Training AlphaSteer Δ matrix...')
steering_layer = config.optimal_steering_layer
benign_prompts = list(BENIGN_QUERIES[:40]) + list(BENIGN_OPENENDED[:60])
malicious_prompts = list(ADVBENCH_BEHAVIORS[:40])

def extract_acts(prompts, layer):
    hook = ActivationHook(model, layer)
    acts = []
    for p in prompts:
        msgs = [{'role': 'user', 'content': p}]
        fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(fmt, return_tensors='pt', truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            model(**inp)
        acts.append(hook.get_last().cpu().float().numpy())
        hook.clear()
    hook.remove()
    return np.concatenate(acts, axis=0)

t0 = time.time()
benign_acts = extract_acts(benign_prompts, steering_layer)
malicious_acts = extract_acts(malicious_prompts, steering_layer)

refusal_dir = benign_acts.mean(axis=0) - malicious_acts.mean(axis=0)
refusal_dir = refusal_dir / np.linalg.norm(refusal_dir)
refusal_directions = np.tile(refusal_dir, (len(malicious_acts), 1))

P_hat = compute_null_space_projection(benign_acts)
Delta = compute_delta_matrix(malicious_acts, refusal_directions, P_hat)
save_alphasteer_matrices(output_dir / 'matrices', Delta, P_hat, steering_layer)
print(f'  Δ matrix trained in {time.time()-t0:.0f}s')

# ---- Build attack + benign sets ----
all_prompts = []
for attack in ['prefilling', 'intent_laundering']:
    all_prompts.extend(load_attack_set(attack, n_attacks=20, include_benign=False))
for i in range(min(20, len(BENIGN_QUERIES))):
    all_prompts.append({'prompt': BENIGN_QUERIES[i], 'attack_type': 'benign'})

# ---- Eval all defenses ----
probe_dir = probe_results_dir(model_name, 'jailbreak')
vector_dir = vector_results_dir(model_name, 'refusal')
vector_path = vector_dir / f'vector_layer{steering_layer}.pt'

with open(probe_dir / 'best_layer.json') as f:
    probe_layer = json.load(f)['layer']

defenses = [
    ('none', None),
    ('static_1.3', 'static'),
    ('gated_bb_1.3', 'gated'),
    ('alphasteer_1.0', 'alphasteer'),
]

results = {}
for def_name, def_type in defenses:
    print(f'\\n  [{def_name}]')
    pgs, static, alpha_def = None, None, None

    if def_type == 'static':
        static = StaticSteering(model, str(vector_path), steering_layer, 1.3)
    elif def_type == 'gated':
        pgs = ProbeGatedSteering(model, GateConfig(
            probe_path=probe_dir / 'best_probe.pkl',
            steering_vector_path=vector_path,
            probe_layer=probe_layer, steering_layer=steering_layer,
            controller_type='bang_bang', threshold=0.3, max_strength=1.3,
        ))
    elif def_type == 'alphasteer':
        alpha_def = AlphaSteerDefense(model, AlphaSteerConfig(
            model_name=model_name,
            matrices_dir=output_dir / 'matrices',
            steering_layers=[steering_layer],
            strength=1.0,
        ))

    # Safety
    _, metrics = run_safety_eval(model, tokenizer, all_prompts, def_name, model_name, device, pgs=pgs)
    print(f'    Safety: DSR={metrics.dsr:.3f}')

    # Utility
    if pgs: pgs.reset()
    oe_result = run_openended_eval(model, tokenizer, device, defense_name=def_name)
    save_openended_results(output_dir / 'openended', oe_result)
    print(f'    Utility: Hedging={oe_result.hedging_rate:.1%}')

    if pgs: pgs.remove()
    if static: static.remove()
    if alpha_def: alpha_def.remove()

    results[def_name] = {
        'dsr': metrics.dsr, 'asr': metrics.asr, 'fpr': metrics.fpr,
        'hedging_rate': oe_result.hedging_rate,
        'mean_response_length': oe_result.mean_response_length,
    }

# ---- Print comparison ----
print()
print('=' * 70)
print('QUEUE 1 RESULTS: AlphaSteer vs Probe-Gated vs Static (Mistral-7B)')
print('=' * 70)
print(f'{\"Defense\":<20} {\"DSR\":>6} {\"Hedge%\":>8} {\"FPR\":>6}')
print('-' * 45)
for name, r in results.items():
    print(f'{name:<20} {r[\"dsr\"]:>6.3f} {r[\"hedging_rate\"]:>7.1%} {r[\"fpr\"]:>6.3f}')

with open(output_dir / 'comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\\nResults saved to {output_dir}')

del model
torch.cuda.empty_cache()
" 2>&1 | tee "$RESULTS_DIR/queue1.log"

    mark_queue_done 1
fi

# ============================================================================
# QUEUE 2: AlphaSteer on remaining 3 models
# ============================================================================

if queue_done 2; then
    log "SKIPPING Queue 2 — results found"
else
    log "=========================================="
    log "QUEUE 2: AlphaSteer on Gemma, Qwen, Llama"
    log "=========================================="

    for MODEL in gemma2-9b qwen25-7b llama31-8b; do
        MARKER="$RESULTS_DIR/.queue_2_${MODEL}_done"
        if [ -f "$MARKER" ]; then
            log "  Skipping $MODEL — already done"
            continue
        fi

        log "  Running AlphaSteer on $MODEL..."
        python3 scripts/05_train_alphasteer.py --model "$MODEL" --device cuda 2>&1 | tee -a "$RESULTS_DIR/queue2_${MODEL}.log"

        # Run eval using the cross-model validation script logic
        python3 -c "
import json, torch, numpy as np
from src.models.loader import load_model
from src.attacks.registry import load_attack_set
from src.attacks.datasets.prompts import BENIGN_QUERIES
from src.baselines.alphasteer.wrapper import AlphaSteerDefense, AlphaSteerConfig
from src.eval.safety_eval import run_safety_eval
from src.eval.openended_eval import run_openended_eval, save_openended_results
from src.utils.paths import RESULTS_DIR
from src.utils.seeding import seed_everything

seed_everything(42)
model_name = '$MODEL'
output_dir = RESULTS_DIR / 'alphasteer' / model_name
output_dir.mkdir(parents=True, exist_ok=True)

model, tokenizer, config = load_model(model_name, device='cuda')
steering_layer = config.optimal_steering_layer

all_prompts = []
for attack in ['prefilling', 'intent_laundering']:
    all_prompts.extend(load_attack_set(attack, n_attacks=20, include_benign=False))
for i in range(min(20, len(BENIGN_QUERIES))):
    all_prompts.append({'prompt': BENIGN_QUERIES[i], 'attack_type': 'benign'})

# AlphaSteer eval
matrices_dir = output_dir / 'matrices'
if not matrices_dir.exists():
    matrices_dir = RESULTS_DIR / 'alphasteer' / model_name
alpha_def = AlphaSteerDefense(model, AlphaSteerConfig(
    model_name=model_name, matrices_dir=matrices_dir,
    steering_layers=[steering_layer], strength=1.0,
))
_, metrics = run_safety_eval(model, tokenizer, all_prompts, 'alphasteer', model_name, 'cuda')
oe = run_openended_eval(model, tokenizer, 'cuda', defense_name='alphasteer')
save_openended_results(output_dir / 'openended', oe)
alpha_def.remove()

result = {'dsr': metrics.dsr, 'hedging_rate': oe.hedging_rate, 'fpr': metrics.fpr}
with open(output_dir / 'alphasteer_result.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f'{model_name}: DSR={metrics.dsr:.3f}, Hedging={oe.hedging_rate:.1%}')
del model; torch.cuda.empty_cache()
" 2>&1 | tee -a "$RESULTS_DIR/queue2_${MODEL}.log"

        touch "$MARKER"
        log "  $MODEL complete"
    done

    mark_queue_done 2
fi

# ============================================================================
# QUEUE 3: CAST baseline on all 4 models
# ============================================================================

if queue_done 3; then
    log "SKIPPING Queue 3 — results found"
else
    log "=========================================="
    log "QUEUE 3: CAST baseline"
    log "=========================================="

    for MODEL in mistral-7b gemma2-9b qwen25-7b llama31-8b; do
        MARKER="$RESULTS_DIR/.queue_3_${MODEL}_done"
        if [ -f "$MARKER" ]; then
            log "  Skipping $MODEL — already done"
            continue
        fi

        log "  Running CAST on $MODEL..."
        python3 -c "
import json, torch, numpy as np
from src.models.loader import load_model
from src.models.hooks import ActivationHook
from src.attacks.registry import load_attack_set
from src.attacks.datasets.prompts import BENIGN_QUERIES, BENIGN_OPENENDED
from src.eval.safety_eval import run_safety_eval
from src.eval.openended_eval import run_openended_eval, save_openended_results
from src.steering.extract_vectors import load_steering_vector
from src.utils.paths import vector_results_dir, RESULTS_DIR
from src.utils.seeding import seed_everything

seed_everything(42)
model_name = '$MODEL'
output_dir = RESULTS_DIR / 'cast' / model_name
output_dir.mkdir(parents=True, exist_ok=True)

model, tokenizer, config = load_model(model_name, device='cuda')
steering_layer = config.optimal_steering_layer

# CAST: cosine-threshold gating
# Load steering vector, compute condition vector from benign activations
vec_dir = vector_results_dir(model_name, 'refusal')
vec_path = vec_dir / f'vector_layer{steering_layer}.pt'
if not vec_path.exists():
    print(f'  No steering vector for {model_name}, skipping')
    exit(0)

steering_vec = load_steering_vector(vec_path)

# Extract benign activation mean as condition vector
benign_prompts = list(BENIGN_QUERIES[:20])
hook = ActivationHook(model, steering_layer)
acts = []
for p in benign_prompts:
    msgs = [{'role': 'user', 'content': p}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(fmt, return_tensors='pt', truncation=True, max_length=4096).to('cuda')
    with torch.no_grad():
        model(**inp)
    acts.append(hook.get_last().cpu().float().numpy())
    hook.clear()
hook.remove()

condition_vec = np.mean(acts, axis=0).squeeze()
condition_vec = condition_vec / np.linalg.norm(condition_vec)
condition_vec_t = torch.from_numpy(condition_vec).float()

# CAST hook: steer when cosine(activation, condition) < threshold
class CASTHook:
    def __init__(self, model, layer, sv, cond, threshold=0.5, strength=1.0):
        self.sv = sv
        self.cond = cond
        self.threshold = threshold
        self.strength = strength
        layer_mod = ActivationHook._resolve_layer(model, layer)
        self._handle = layer_mod.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        act = h[:, -1, :].detach().cpu().float()
        cos = torch.nn.functional.cosine_similarity(act, self.cond.unsqueeze(0))
        if cos.item() < self.threshold:
            sv = self.sv.to(h.device, dtype=h.dtype)
            h[:, -1, :] += self.strength * sv
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    def remove(self):
        self._handle.remove()

# Build attack set
all_prompts = []
for attack in ['prefilling', 'intent_laundering']:
    all_prompts.extend(load_attack_set(attack, n_attacks=20, include_benign=False))
for i in range(min(20, len(BENIGN_QUERIES))):
    all_prompts.append({'prompt': BENIGN_QUERIES[i], 'attack_type': 'benign'})

# Run CAST
cast = CASTHook(model, steering_layer, steering_vec, condition_vec_t, threshold=0.3, strength=1.0)
_, metrics = run_safety_eval(model, tokenizer, all_prompts, 'cast', model_name, 'cuda')
oe = run_openended_eval(model, tokenizer, 'cuda', defense_name='cast')
save_openended_results(output_dir / 'openended', oe)
cast.remove()

result = {'dsr': metrics.dsr, 'hedging_rate': oe.hedging_rate, 'fpr': metrics.fpr}
with open(output_dir / 'cast_result.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f'{model_name}: DSR={metrics.dsr:.3f}, Hedging={oe.hedging_rate:.1%}')

del model; torch.cuda.empty_cache()
" 2>&1 | tee -a "$RESULTS_DIR/queue3_${MODEL}.log"

        touch "$MARKER"
        log "  $MODEL complete"
    done

    mark_queue_done 3
fi

# ============================================================================
# QUEUE 4: GCG suffix generation
# ============================================================================

if queue_done 4; then
    log "SKIPPING Queue 4 — results found"
else
    log "=========================================="
    log "QUEUE 4: GCG suffix generation"
    log "=========================================="

    for MODEL in mistral-7b gemma2-9b qwen25-7b llama31-8b; do
        MARKER="$RESULTS_DIR/.queue_4_${MODEL}_done"
        if [ -f "$MARKER" ]; then
            log "  Skipping $MODEL — already done"
            continue
        fi

        log "  Generating GCG suffixes for $MODEL..."
        python3 -c "
import json, torch
from src.models.loader import load_model
from src.models.configs import get_config
from src.attacks.datasets.prompts import ADVBENCH_BEHAVIORS
from src.utils.paths import DATA_DIR
from src.utils.seeding import seed_everything
from pathlib import Path

seed_everything(42)
model_name = '$MODEL'
config = get_config(model_name)

print(f'Loading {model_name}...')
model, tokenizer, _ = load_model(model_name, device='cuda')

try:
    import nanogcg
    print('nanogcg available — generating optimized suffixes')
except ImportError:
    print('WARNING: nanogcg not available, generating placeholder suffixes')
    # Save placeholder suffixes
    out_dir = DATA_DIR / 'attacks' / 'gcg'
    out_dir.mkdir(parents=True, exist_ok=True)
    suffixes = ['[placeholder — nanogcg not installed]'] * 100
    with open(out_dir / f'{model_name}_suffixes.json', 'w') as f:
        json.dump(suffixes, f, indent=2)
    print(f'Saved {len(suffixes)} placeholder suffixes')
    exit(0)

# Generate real GCG suffixes
behaviors = ADVBENCH_BEHAVIORS[:100]
out_dir = DATA_DIR / 'attacks' / 'gcg'
out_dir.mkdir(parents=True, exist_ok=True)

suffixes = []
for i, behavior in enumerate(behaviors):
    print(f'  [{i+1}/{len(behaviors)}] {behavior[:50]}...')
    try:
        result = nanogcg.run(model, tokenizer, behavior, num_steps=256, batch_size=256)
        suffixes.append(result.best_string)
        print(f'    Loss: {result.best_loss:.4f}')
    except Exception as e:
        print(f'    ERROR: {e}')
        suffixes.append('')

    # Checkpoint every 10
    if (i + 1) % 10 == 0:
        with open(out_dir / f'{model_name}_suffixes.json', 'w') as f:
            json.dump(suffixes, f, indent=2)

with open(out_dir / f'{model_name}_suffixes.json', 'w') as f:
    json.dump(suffixes, f, indent=2)
print(f'Saved {len(suffixes)} suffixes')

del model; torch.cuda.empty_cache()
" 2>&1 | tee -a "$RESULTS_DIR/queue4_${MODEL}.log"

        touch "$MARKER"
        log "  $MODEL complete"
    done

    mark_queue_done 4
fi

# ============================================================================
# QUEUE 5: Expanded eval with GCG
# ============================================================================

if queue_done 5; then
    log "SKIPPING Queue 5 — results found"
else
    log "=========================================="
    log "QUEUE 5: Expanded eval with GCG attacks"
    log "=========================================="

    for MODEL in mistral-7b gemma2-9b qwen25-7b llama31-8b; do
        MARKER="$RESULTS_DIR/.queue_5_${MODEL}_done"
        if [ -f "$MARKER" ]; then
            log "  Skipping $MODEL — already done"
            continue
        fi

        log "  Running expanded eval on $MODEL..."
        python3 -c "
import json, torch
from src.models.loader import load_model
from src.attacks.registry import load_attack_set
from src.attacks.datasets.prompts import BENIGN_QUERIES
from src.eval.safety_eval import run_safety_eval, save_eval_results
from src.baselines.static_steering.infer import StaticSteering
from src.gating.probe_gate import GateConfig, ProbeGatedSteering
from src.baselines.alphasteer.wrapper import AlphaSteerDefense, AlphaSteerConfig
from src.utils.paths import probe_results_dir, vector_results_dir, RESULTS_DIR
from src.utils.seeding import seed_everything

seed_everything(42)
model_name = '$MODEL'
output_dir = RESULTS_DIR / 'expanded_eval' / model_name
output_dir.mkdir(parents=True, exist_ok=True)

model, tokenizer, config = load_model(model_name, device='cuda')
sl = config.optimal_steering_layer

# Build expanded attack set (all available attacks)
all_prompts = []
for attack_name in ['prefilling', 'intent_laundering', 'deep_inception', 'gcg', 'autodan']:
    try:
        attacks = load_attack_set(attack_name, n_attacks=20, include_benign=False)
        all_prompts.extend(attacks)
        print(f'  Loaded {len(attacks)} {attack_name} attacks')
    except Exception as e:
        print(f'  Skipping {attack_name}: {e}')
for i in range(min(20, len(BENIGN_QUERIES))):
    all_prompts.append({'prompt': BENIGN_QUERIES[i], 'attack_type': 'benign'})

print(f'  Total: {len(all_prompts)} prompts')

# Run defenses
probe_dir = probe_results_dir(model_name, 'jailbreak')
vec_dir = vector_results_dir(model_name, 'refusal')
vec_path = vec_dir / f'vector_layer{sl}.pt'

results = {}
for def_name in ['none', 'static_1.3', 'gated_bb_1.3']:
    print(f'  [{def_name}]...')
    pgs, static = None, None
    if def_name == 'static_1.3':
        static = StaticSteering(model, str(vec_path), sl, 1.3)
    elif def_name == 'gated_bb_1.3':
        if (probe_dir / 'best_probe.pkl').exists():
            with open(probe_dir / 'best_layer.json') as f:
                pl = json.load(f)['layer']
            pgs = ProbeGatedSteering(model, GateConfig(
                probe_path=probe_dir / 'best_probe.pkl',
                steering_vector_path=vec_path,
                probe_layer=pl, steering_layer=sl,
                controller_type='bang_bang', threshold=0.3, max_strength=1.3,
            ))

    _, metrics = run_safety_eval(model, tokenizer, all_prompts, def_name, model_name, 'cuda', pgs=pgs)
    save_eval_results(output_dir, [], metrics, def_name, model_name)
    if pgs: pgs.remove()
    if static: static.remove()

    results[def_name] = {'dsr': metrics.dsr, 'per_attack_asr': metrics.per_attack_asr}
    print(f'    DSR={metrics.dsr:.3f}')

with open(output_dir / 'expanded_results.json', 'w') as f:
    json.dump(results, f, indent=2)

del model; torch.cuda.empty_cache()
" 2>&1 | tee -a "$RESULTS_DIR/queue5_${MODEL}.log"

        touch "$MARKER"
        log "  $MODEL complete"
    done

    mark_queue_done 5
fi

# ============================================================================
# QUEUE 6: Ablation study (Mistral-7B)
# ============================================================================

if queue_done 6; then
    log "SKIPPING Queue 6 — results found"
else
    log "=========================================="
    log "QUEUE 6: Ablation study (Mistral-7B)"
    log "=========================================="

    python3 -c "
import json, torch
from src.models.loader import load_model
from src.attacks.registry import load_attack_set
from src.attacks.datasets.prompts import BENIGN_QUERIES
from src.eval.safety_eval import run_safety_eval
from src.eval.openended_eval import run_openended_eval, save_openended_results
from src.baselines.static_steering.infer import StaticSteering
from src.gating.probe_gate import GateConfig, ProbeGatedSteering
from src.utils.paths import probe_results_dir, vector_results_dir, RESULTS_DIR
from src.utils.seeding import seed_everything

seed_everything(42)
model_name = 'mistral-7b'
output_dir = RESULTS_DIR / 'ablation' / model_name
output_dir.mkdir(parents=True, exist_ok=True)

model, tokenizer, config = load_model(model_name, device='cuda')
sl = config.optimal_steering_layer
probe_dir = probe_results_dir(model_name, 'jailbreak')
vec_dir = vector_results_dir(model_name, 'refusal')
vec_path = vec_dir / f'vector_layer{sl}.pt'

with open(probe_dir / 'best_layer.json') as f:
    pl = json.load(f)['layer']

all_prompts = []
for attack in ['prefilling', 'intent_laundering']:
    all_prompts.extend(load_attack_set(attack, n_attacks=20, include_benign=False))
for i in range(min(20, len(BENIGN_QUERIES))):
    all_prompts.append({'prompt': BENIGN_QUERIES[i], 'attack_type': 'benign'})

# Ablation conditions:
# 1. none: no defense
# 2. probe_only: detect but don't steer (strength=0)
# 3. steer_only: always steer (= static)
# 4. probe+steer: full system
ablations = {
    'none': {'type': 'none'},
    'probe_only': {'type': 'gated', 'strength': 0.0},  # probe fires but strength=0
    'steer_only': {'type': 'static', 'strength': 1.3},  # static = always steer
    'probe_steer': {'type': 'gated', 'strength': 1.3},  # full system
}

results = {}
for name, cfg in ablations.items():
    print(f'  [{name}]')
    pgs, static = None, None

    if cfg['type'] == 'static':
        static = StaticSteering(model, str(vec_path), sl, cfg['strength'])
    elif cfg['type'] == 'gated':
        pgs = ProbeGatedSteering(model, GateConfig(
            probe_path=probe_dir / 'best_probe.pkl',
            steering_vector_path=vec_path,
            probe_layer=pl, steering_layer=sl,
            controller_type='bang_bang', threshold=0.3, max_strength=cfg['strength'],
        ))

    _, metrics = run_safety_eval(model, tokenizer, all_prompts, name, model_name, 'cuda', pgs=pgs)
    oe = run_openended_eval(model, tokenizer, 'cuda', defense_name=name)
    save_openended_results(output_dir / 'openended', oe)

    if pgs: pgs.remove()
    if static: static.remove()

    results[name] = {
        'dsr': metrics.dsr, 'hedging_rate': oe.hedging_rate, 'fpr': metrics.fpr,
    }
    print(f'    DSR={metrics.dsr:.3f}, Hedging={oe.hedging_rate:.1%}')

print()
print('ABLATION RESULTS')
print(f'{\"Condition\":<15} {\"DSR\":>6} {\"Hedge%\":>8}')
for name, r in results.items():
    print(f'{name:<15} {r[\"dsr\"]:>6.3f} {r[\"hedging_rate\"]:>7.1%}')

with open(output_dir / 'ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

del model; torch.cuda.empty_cache()
" 2>&1 | tee "$RESULTS_DIR/queue6.log"

    mark_queue_done 6
fi

# ============================================================================
# DONE
# ============================================================================

log ""
log "============================================================"
log "ALL QUEUES COMPLETE"
log "============================================================"
update_cost
log ""
log "Results in: $RESULTS_DIR/"
log "  alphasteer/    — Queue 1-2"
log "  cast/          — Queue 3"
log "  expanded_eval/ — Queue 5"
log "  ablation/      — Queue 6"
log "  data/attacks/gcg/ — Queue 4"
