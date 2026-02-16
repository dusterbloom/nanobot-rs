#!/bin/bash
# setup.sh - Launch two llama-server instances for local-only inference
#
# VRAM Budget: 24GB (RTX3090)
# Allocation:
#   - Main (8080): Ministral-8B (~5GB model + 1.5GB KV) - Main agent
#   - Subagent (8083): Ministral-3B (~2GB model + 0.5GB KV) - Task execution
#   Total: ~9GB, headroom: ~15GB
#
# Optimizations:
#   - Q8 KV cache (50% memory savings)
#   - Repetition penalty (prevent loops)
#   - Generation limits (prevent runaway)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PID_DIR="$SCRIPT_DIR/pids"

LLAMA_SERVER="/home/peppi/llama.cpp/build/bin/llama-server"
MODELS_DIR="/home/peppi/models"

mkdir -p "$LOG_DIR" "$PID_DIR"

# Model configurations: model_path:port:context_size
declare -A MODELS=(
    ["main"]="$MODELS_DIR/Ministral-8B-Instruct-Q4_K_M.gguf:8080:16384"
    ["subagent"]="$MODELS_DIR/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf:8083:8192"
)

start_server() {
    local name="$1"
    local config="${MODELS[$name]}"
    local model="${config%%:*}"
    local rest="${config#*:}"
    local port="${rest%%:*}"
    local ctx="${rest#*:}"
    local pidfile="$PID_DIR/${name}.pid"
    local logfile="$LOG_DIR/${name}.log"

    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "[$name] Already running on port $port (PID $(cat $pidfile))"
        return 0
    fi

    if [ ! -f "$model" ]; then
        echo "[$name] ERROR: Model not found: $model"
        return 1
    fi

    echo "[$name] Starting on port $port (ctx=$ctx)..."
    echo "  Model: $model"

    # Optimized settings for small models:
    # - Q8 KV cache: 50% memory savings
    # - repeat-penalty 1.1: prevent repetition loops
    # - repeat-last-n 128: check more context for repeats
    # - n-predict 2048: limit generation length
    # - temp 0.7, top-p 0.9: balanced sampling
    env -i HOME="$HOME" PATH="$PATH" \
        LD_LIBRARY_PATH="/home/peppi/llama.cpp/build/bin:/usr/local/cuda/lib64" \
        CUDA_VISIBLE_DEVICES=0 \
        "$LLAMA_SERVER" \
        --model "$model" \
        --port "$port" \
        --n-gpu-layers 99 \
        --ctx-size "$ctx" \
        --threads 4 \
        --flash-attn on \
        --cache-type-k q8_0 \
        --cache-type-v q8_0 \
        --repeat-penalty 1.1 \
        --repeat-last-n 128 \
        --n-predict 2048 \
        --temp 0.7 \
        --top-p 0.9 \
        > "$logfile" 2>&1 &

    echo $! > "$pidfile"
    echo "  PID: $(cat $pidfile)"
}

wait_for_server() {
    local name="$1"
    local port="${2:-8080}"
    local max_wait=90
    
    echo "[$name] Waiting for server..."
    for i in $(seq 1 $max_wait); do
        if curl -s "http://localhost:$port/health" 2>/dev/null | grep -q "ok"; then
            echo "[$name] Ready!"
            return 0
        fi
        sleep 1
    done
    echo "[$name] WARNING: Server not ready after ${max_wait}s"
    return 1
}

stop_servers() {
    echo "Stopping all servers..."
    for name in "${!MODELS[@]}"; do
        local pidfile="$PID_DIR/${name}.pid"
        if [ -f "$pidfile" ]; then
            local pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid"
                echo "[$name] Stopped (PID $pid)"
            fi
            rm -f "$pidfile"
        fi
    done
}

status() {
    echo "Server Status:"
    echo "==============="
    for name in "${!MODELS[@]}"; do
        local config="${MODELS[$name]}"
        local port="${config#*:}"
        port="${port%%:*}"
        local pidfile="$PID_DIR/${name}.pid"
        
        printf "%-12s " "[$name]"
        if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
            echo "RUNNING (PID $(cat $pidfile), port $port)"
        else
            echo "STOPPED"
        fi
    done
}

case "${1:-}" in
    start)
        echo "Starting local-only llama-server cluster..."
        echo ""
        start_server "subagent"
        sleep 2
        start_server "main"
        echo ""
        echo "Waiting for all servers..."
        wait_for_server "subagent" 8083
        wait_for_server "main" 8080
        echo ""
        status
        ;;
    stop)
        stop_servers
        ;;
    restart)
        stop_servers
        sleep 2
        $0 start
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo ""
        echo "Model allocation (24GB VRAM, Q8 KV cache):"
        echo "  Main (8080):       Ministral-8B      (~6.5GB, 16K ctx)"
        echo "  Subagent (8083):   Ministral-3-3B    (~2.5GB, 8K ctx)"
        exit 1
        ;;
esac
