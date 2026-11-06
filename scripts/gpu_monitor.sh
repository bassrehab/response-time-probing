#!/bin/bash

# ============================================================================
# GPU Monitor — run in a second terminal
# Usage: bash scripts/gpu_monitor.sh
# ============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results"
RATE_SPOT=1.10
RATE_ONDEMAND=3.67

# Find start time from first cost log entry or queue marker
START_TIME=""
if [ -f "$RESULTS_DIR/gpu_cost.log" ]; then
    START_TIME=$(head -1 "$RESULTS_DIR/gpu_cost.log" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    from datetime import datetime
    # Approximate start from elapsed
    print(d.get('timestamp', ''))
except: pass
" 2>/dev/null)
fi

while true; do
    clear
    echo "============================================================"
    echo "  GPU MONITOR — $(date)"
    echo "============================================================"
    echo ""

    # GPU status
    echo "  GPU:"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | while read line; do
        echo "    $line"
    done
    echo ""

    # Queue status
    echo "  QUEUES:"
    for q in 1 2 3 4 5 6; do
        case $q in
            1) desc="AlphaSteer Mistral (CRITICAL)" ;;
            2) desc="AlphaSteer Gemma/Qwen/Llama" ;;
            3) desc="CAST all models" ;;
            4) desc="GCG suffix generation" ;;
            5) desc="Expanded eval with GCG" ;;
            6) desc="Ablation study" ;;
        esac

        if [ -f "$RESULTS_DIR/.queue_${q}_done" ]; then
            echo "    Queue $q: DONE    — $desc"
        elif [ -f "$RESULTS_DIR/queue${q}.log" ] || ls "$RESULTS_DIR/queue${q}_"*.log 2>/dev/null | head -1 > /dev/null 2>&1; then
            echo "    Queue $q: RUNNING — $desc"
        else
            echo "    Queue $q: PENDING — $desc"
        fi
    done
    echo ""

    # Per-model sub-queue status for queues 2-5
    for q in 2 3 4 5; do
        has_any=false
        for model in mistral-7b gemma2-9b qwen25-7b llama31-8b; do
            if [ -f "$RESULTS_DIR/.queue_${q}_${model}_done" ]; then
                has_any=true
            fi
        done
        if $has_any && ! [ -f "$RESULTS_DIR/.queue_${q}_done" ]; then
            echo "    Queue $q sub-models:"
            for model in mistral-7b gemma2-9b qwen25-7b llama31-8b; do
                if [ -f "$RESULTS_DIR/.queue_${q}_${model}_done" ]; then
                    echo "      $model: DONE"
                else
                    echo "      $model: pending"
                fi
            done
            echo ""
        fi
    done

    # Cost estimate
    if [ -f "$RESULTS_DIR/gpu_cost.log" ]; then
        LATEST=$(tail -1 "$RESULTS_DIR/gpu_cost.log" 2>/dev/null)
        if [ -n "$LATEST" ]; then
            echo "  COST:"
            echo "$LATEST" | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    print(f'    Elapsed: {d[\"elapsed_hrs\"]:.1f} hours')
    print(f'    Spot:    \${d[\"cost_spot\"]:.2f}')
    print(f'    On-demand: \${d[\"cost_ondemand\"]:.2f}')
except: pass
"
        fi
    else
        echo "  COST: (waiting for first queue to complete)"
    fi
    echo ""

    # Latest log activity
    echo "  LATEST LOG:"
    LATEST_LOG=$(ls -t "$RESULTS_DIR"/queue*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "    $(basename "$LATEST_LOG"):"
        tail -3 "$LATEST_LOG" 2>/dev/null | while read line; do
            echo "      $line"
        done
    else
        echo "    (no logs yet)"
    fi
    echo ""

    # Running processes
    echo "  PROCESSES:"
    PROCS=$(ps aux | grep "[p]ython.*scripts/" | grep -v monitor | awk '{print $11, $12}' | head -5)
    if [ -n "$PROCS" ]; then
        echo "$PROCS" | while read line; do
            echo "    $line"
        done
    else
        echo "    (none running)"
    fi

    echo ""
    echo "  Refresh in 30s... (Ctrl+C to stop)"
    sleep 30
done
