#!/data/data/com.termux/files/usr/bin/bash
# GreyFox-CLI setup script (Termux/Android)
# Installs deps, builds llama.cpp, and launches a llama-server + the CLI.
#
# You choose the model. Point MODEL_URL/MODEL_FILE at any tool-calling-capable
# chat GGUF (e.g. a Qwen2.5-Instruct, Llama-3.x-Instruct, or similar small
# quant sized for your device's RAM). This script does not hardcode a model.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config -- edit these
# ---------------------------------------------------------------------------
BASE_DIR="$HOME/greyfox-cli"
LLAMACPP_DIR="$BASE_DIR/llama.cpp"
MODELS_DIR="$BASE_DIR/models"
VENV_DIR="$BASE_DIR/venv"
LOG_DIR="$BASE_DIR/logs"

LLAMACPP_PIN="b7500"   # pin to a known-working tag/commit; bump deliberately.
                        # MUST be recent enough to support streaming responses
                        # together with tool calling (older builds -- roughly
                        # anything around the original b3600 pin -- reject that
                        # combination with a 500 "Cannot use tools with stream").
                        # greyfox_cli.py detects that specific error and falls
                        # back to non-streaming automatically, so an old pin
                        # degrades rather than breaks, but bump this when you can.

MODEL_URL="${MODEL_URL:-}"        # e.g. https://huggingface.co/<repo>/resolve/main/<file>.gguf
MODEL_FILE="${MODEL_FILE:-model.gguf}"

SERVER_PORT="${SERVER_PORT:-8080}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo -e "\e[1;34m[greyfox]\e[0m $*"; }
warn() { echo -e "\e[1;33m[greyfox]\e[0m $*"; }
err()  { echo -e "\e[1;31m[greyfox]\e[0m $*" >&2; }

pick_ram_tier() {
    # returns a context size based on available RAM
    local mem_kb
    mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local mem_gb=$((mem_kb / 1024 / 1024))
    if [ "$mem_gb" -ge 7 ]; then
        echo 16384
    elif [ "$mem_gb" -ge 5 ]; then
        echo 10240
    else
        echo 6144
    fi
}

print_battery_hint() {
    local model
    model=$(getprop ro.product.manufacturer 2>/dev/null || echo "unknown")
    log "Battery optimization can kill the background server. Manufacturer detected: $model"
    cat <<'EOF'
  Xiaomi/Redmi (MIUI/HyperOS): Security app -> Battery -> Termux -> No restrictions;
                                also disable "MIUI Optimization" in Developer Options.
  Samsung (One UI):            Device Care -> Battery -> Termux -> add to "Never sleeping apps".
  Huawei (EMUI/HarmonyOS):     Battery settings -> Protected Apps -> enable for Termux.
  Other:                       look for a per-app battery/background restriction setting.
EOF
}

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

install_packages() {
    log "Installing Termux packages..."
    pkg install -y git cmake clang python build-essential libandroid-spawn >/dev/null
}

build_llamacpp() {
    mkdir -p "$BASE_DIR"
    if [ -d "$LLAMACPP_DIR" ] && [ -f "$LLAMACPP_DIR/build/bin/llama-server" ] && [ "${1:-}" != "--force" ]; then
        log "llama.cpp already built, skipping (use --force-rebuild to redo)."
        return
    fi
    if [ ! -d "$LLAMACPP_DIR" ]; then
        log "Cloning llama.cpp..."
        git clone https://github.com/ggml-org/llama.cpp "$LLAMACPP_DIR"
    fi
    cd "$LLAMACPP_DIR"
    git fetch --tags
    git checkout "$LLAMACPP_PIN"
    log "Building llama.cpp (this takes a while on-device)..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_LLAMAFILE=OFF
    cmake --build build --config Release -j"$(nproc)"
    cd "$BASE_DIR"
}

download_model() {
    mkdir -p "$MODELS_DIR"
    local dest="$MODELS_DIR/$MODEL_FILE"
    if [ -f "$dest" ]; then
        log "Model already present at $dest, skipping download."
        return
    fi
    if [ -z "$MODEL_URL" ]; then
        err "MODEL_URL is not set. Export MODEL_URL (and optionally MODEL_FILE) to a GGUF"
        err "chat model before running setup.sh, e.g.:"
        err "  MODEL_URL=https://huggingface.co/.../resolve/main/model.gguf bash setup.sh"
        exit 1
    fi
    log "Downloading model to $dest ..."
    curl -L -C - -o "$dest" "$MODEL_URL"
}

setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        log "Creating Python venv..."
        python -m venv "$VENV_DIR"
    fi
    "$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
    # greyfox_cli.py only uses the stdlib (including concurrent.futures,
    # threading, tempfile -- all standard since Python 3.2+), so nothing
    # else needs installing here regardless of Termux's Python version.
}

install_cli() {
    cp "$(dirname "$0")/greyfox_cli.py" "$BASE_DIR/greyfox_cli.py"
}

start_server() {
    mkdir -p "$LOG_DIR"
    if pgrep -f "llama-server" >/dev/null 2>&1; then
        log "llama-server already running."
        return
    fi
    local ctx
    ctx=$(pick_ram_tier)
    log "Starting llama-server (context=$ctx, port=$SERVER_PORT)..."
    nohup "$LLAMACPP_DIR/build/bin/llama-server" \
        -m "$MODELS_DIR/$MODEL_FILE" \
        -c "$ctx" \
        --port "$SERVER_PORT" \
        --host 127.0.0.1 \
        --jinja \
        > "$LOG_DIR/llama-server.log" 2>&1 &
    echo $! > "$BASE_DIR/server.pid"
    log "Waiting for server to come up..."
    for _ in $(seq 1 30); do
        if curl -s "http://127.0.0.1:$SERVER_PORT/health" >/dev/null 2>&1; then
            log "Server is up."
            return
        fi
        sleep 1
    done
    err "Server did not come up in time. Check $LOG_DIR/llama-server.log"
    tail -n 30 "$LOG_DIR/llama-server.log" >&2
    exit 1
}

stop_server() {
    if [ -f "$BASE_DIR/server.pid" ]; then
        kill "$(cat "$BASE_DIR/server.pid")" 2>/dev/null || true
        rm -f "$BASE_DIR/server.pid"
        log "Server stopped."
    else
        pkill -f "llama-server" 2>/dev/null || true
        log "No tracked server.pid; sent pkill for llama-server anyway."
    fi
}

install_autostart() {
    mkdir -p "$HOME/.termux/boot" 2>/dev/null || true
    local hook="$HOME/.shortcuts/greyfox-autostart.sh"
    mkdir -p "$HOME/.shortcuts"
    cat > "$hook" <<EOF
#!/data/data/com.termux/files/usr/bin/bash
bash "$BASE_DIR/setup.sh" --run-only
EOF
    chmod +x "$hook"
    log "Autostart hook installed at $hook"
}

disable_autostart() {
    rm -f "$HOME/.shortcuts/greyfox-autostart.sh"
    log "Autostart hook removed."
}

selftest() {
    log "Running selftest..."
    start_server
    local base="http://127.0.0.1:$SERVER_PORT"
    local resp

    resp=$(curl -s "$base/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"Say OK and nothing else."}]}')
    echo "$resp" | grep -q "OK" && log "chat completion: PASS" || warn "chat completion: check output above"

    resp=$(curl -s "$base/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"stream": true, "messages":[{"role":"user","content":"count to 3"}]}')
    echo "$resp" | grep -q "^data:" && log "streaming (no tools): PASS" || warn "streaming (no tools): check output above"

    # this is the combination greyfox_cli.py actually uses on every turn --
    # older llama-server builds reject it with a 500 "Cannot use tools with stream"
    resp=$(curl -s -w '\nHTTP_STATUS:%{http_code}' "$base/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"stream": true, "tools": [{"type":"function","function":{"name":"noop","description":"no-op","parameters":{"type":"object","properties":{}}}}], "messages":[{"role":"user","content":"hi"}]}')
    if echo "$resp" | grep -q "HTTP_STATUS:200" && ! echo "$resp" | grep -qi "cannot use tools with stream"; then
        log "streaming + tool calling: PASS"
    else
        warn "streaming + tool calling: NOT SUPPORTED by this llama-server build."
        warn "  greyfox_cli.py will still work -- it detects this and falls back to"
        warn "  non-streaming automatically -- but replies won't stream token-by-token."
        warn "  Bump LLAMACPP_PIN in setup.sh to a more recent tag and run"
        warn "  './setup.sh --force-rebuild' to fix this."
    fi

    resp=$(curl -s "$base/props")
    if echo "$resp" | grep -q "n_ctx"; then
        log "/props (context-size detection): PASS"
    else
        warn "/props doesn't report n_ctx -- greyfox_cli.py will fall back to a"
        warn "  RAM-tier estimate for context-usage tracking / compaction timing."
    fi
}

launch_cli() {
    "$VENV_DIR/bin/python" "$BASE_DIR/greyfox_cli.py"
}

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

FORCE_REBUILD=0
SETUP_ONLY=0
RUN_ONLY=0
NO_AUTOSTART=0

for arg in "$@"; do
    case "$arg" in
        --setup-only) SETUP_ONLY=1 ;;
        --run-only) RUN_ONLY=1 ;;
        --force-rebuild) FORCE_REBUILD=1 ;;
        --selftest) selftest; exit 0 ;;
        --stop) stop_server; exit 0 ;;
        --no-autostart) NO_AUTOSTART=1 ;;
        --disable-autostart) disable_autostart; exit 0 ;;
        --version) echo "GreyFox-CLI 0.2.0"; exit 0 ;;
        *) err "unknown flag: $arg"; exit 1 ;;
    esac
done

if [ "$RUN_ONLY" -eq 1 ]; then
    start_server
    launch_cli
    exit 0
fi

install_packages
if [ "$FORCE_REBUILD" -eq 1 ]; then
    build_llamacpp --force
else
    build_llamacpp
fi
download_model
setup_venv
install_cli
print_battery_hint
if [ "$NO_AUTOSTART" -eq 0 ]; then
    install_autostart
fi

if [ "$SETUP_ONLY" -eq 1 ]; then
    log "Setup complete (--setup-only given, not launching)."
    exit 0
fi

start_server
launch_cli
