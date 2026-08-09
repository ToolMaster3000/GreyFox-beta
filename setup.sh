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
CONFIG_FILE="$BASE_DIR/config.env"   # persists MODEL_FILE/MODEL_URL/SERVER_PORT
                                      # across invocations that don't set them
                                      # (see save_config below)

LLAMACPP_PIN="b7501"   # pin to a known-working tag/commit; bump deliberately.
                        # MUST be recent enough to support streaming responses
                        # together with tool calling -- that landed in
                        # llama.cpp at tag b5478 (2025-05-25, ggml-org/llama.cpp
                        # PR #12379, fixed up in b5495). b7501 (2025-12-21) is
                        # comfortably past that. Older builds -- including the
                        # b3600 this used to be pinned to -- reject stream+tools
                        # outright with a 500 "Cannot use tools with stream".
                        # greyfox_cli.py detects that specific error and falls
                        # back to non-streaming automatically, so an old pin
                        # degrades rather than breaks, but bump this when you can.
                        # (Double-check any pin you choose actually exists --
                        # `git ls-remote --tags https://github.com/ggml-org/llama.cpp
                        # | grep <tag>` -- llama.cpp's build-number tags have gaps.)

# MODEL_URL/MODEL_FILE/SERVER_PORT: an explicitly-exported env var for *this*
# invocation always wins. Otherwise fall back to whatever was saved from a
# previous full setup (see save_config). This matters because autostart's
# boot hook and `--run-only` never re-export these -- without persisting them,
# a custom MODEL_FILE would silently revert to the "model.gguf" default (and
# fail to find your actual model) on every reboot. Falls back to hardcoded
# defaults if neither an env var nor a saved config exists yet.
_explicit_model_url="${MODEL_URL:-}"
_explicit_model_file="${MODEL_FILE:-}"
_explicit_server_port="${SERVER_PORT:-}"
if [ -f "$CONFIG_FILE" ]; then
    # shellcheck disable=SC1090
    . "$CONFIG_FILE"
fi
MODEL_URL="${_explicit_model_url:-${MODEL_URL:-}}"
MODEL_FILE="${_explicit_model_file:-${MODEL_FILE:-model.gguf}}"
SERVER_PORT="${_explicit_server_port:-${SERVER_PORT:-8080}}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo -e "\e[1;34m[greyfox]\e[0m $*"; }
warn() { echo -e "\e[1;33m[greyfox]\e[0m $*"; }
err()  { echo -e "\e[1;31m[greyfox]\e[0m $*" >&2; }

pick_ram_tier() {
    # returns a context size based on available RAM
    local mem_kb
    mem_kb=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
    mem_kb="${mem_kb:-0}"   # if /proc/meminfo is missing/unparseable, don't
                            # let an empty value blow up the arithmetic below
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
    if [ -d "$LLAMACPP_DIR" ] && [ ! -d "$LLAMACPP_DIR/.git" ]; then
        # a directory here that isn't actually a git checkout (e.g. a
        # previous clone that got interrupted) would otherwise make the
        # `git fetch` below fail with a cryptic "not a git repository"
        warn "$LLAMACPP_DIR exists but isn't a git checkout -- removing and re-cloning."
        rm -rf "$LLAMACPP_DIR"
    fi
    if [ ! -d "$LLAMACPP_DIR" ]; then
        log "Cloning llama.cpp..."
        git clone https://github.com/ggml-org/llama.cpp "$LLAMACPP_DIR"
    fi
    cd "$LLAMACPP_DIR"
    git fetch --tags
    if ! git rev-parse --verify -q "$LLAMACPP_PIN" >/dev/null; then
        err "LLAMACPP_PIN=\"$LLAMACPP_PIN\" doesn't exist in ggml-org/llama.cpp."
        err "llama.cpp's build-number tags (bNNNN) have gaps -- check what actually"
        err "exists before picking one, e.g.:"
        err "  git ls-remote --tags https://github.com/ggml-org/llama.cpp | grep b7501"
        err "or browse https://github.com/ggml-org/llama.cpp/tags"
        exit 1
    fi
    git checkout "$LLAMACPP_PIN"
    log "Building llama.cpp (this takes a while on-device)..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_LLAMAFILE=OFF
    cmake --build build --config Release -j"$(nproc 2>/dev/null || echo 4)"
    cd "$BASE_DIR"
}

download_model() {
    mkdir -p "$MODELS_DIR"
    local dest="$MODELS_DIR/$MODEL_FILE"
    if [ -f "$dest" ]; then
        if [ -s "$dest" ] && [ "$(head -c 4 "$dest" 2>/dev/null)" = "GGUF" ]; then
            log "Model already present at $dest, skipping download."
            return
        fi
        warn "Existing file at $dest doesn't look like a valid GGUF (empty, or an"
        warn "  interrupted previous download) -- removing and re-downloading."
        rm -f "$dest"
    fi
    if [ -z "$MODEL_URL" ]; then
        err "MODEL_URL is not set. Export MODEL_URL (and optionally MODEL_FILE) to a GGUF"
        err "chat model before running setup.sh, e.g.:"
        err "  MODEL_URL=https://huggingface.co/.../resolve/main/model.gguf bash setup.sh"
        exit 1
    fi
    log "Downloading model to $dest ..."
    # -f: fail on HTTP errors instead of silently writing the error page to
    # $dest and reporting success (curl's default behavior for e.g. a 404).
    if ! curl -fL -C - -o "$dest" "$MODEL_URL"; then
        err "Model download failed. Check MODEL_URL and your connection."
        rm -f "$dest"
        exit 1
    fi
    # -f catches most HTTP-error cases, but a redirect to an HTML
    # login/consent page (common on gated HF repos without a token) can still
    # come back as 200 -- catch that by checking the GGUF magic bytes.
    if [ ! -s "$dest" ] || [ "$(head -c 4 "$dest")" != "GGUF" ]; then
        err "Downloaded file at $dest doesn't look like a valid GGUF model."
        err "MODEL_URL may be wrong, gated (needs an auth token), or pointing at an"
        err "HTML page instead of the raw file."
        rm -f "$dest"
        exit 1
    fi
    log "Model downloaded and verified."
}

save_config() {
    # Persists the resolved MODEL_URL/MODEL_FILE/SERVER_PORT so that a later
    # invocation with no env vars set (autostart's boot hook, `--run-only`)
    # still finds the right model instead of silently falling back to the
    # "model.gguf" default.
    mkdir -p "$BASE_DIR"
    cat > "$CONFIG_FILE" <<EOF
MODEL_URL="$MODEL_URL"
MODEL_FILE="$MODEL_FILE"
SERVER_PORT="$SERVER_PORT"
EOF
}

setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        log "Creating Python venv..."
        local py_bin
        py_bin="$(command -v python3 || command -v python || true)"
        if [ -z "$py_bin" ]; then
            err "No python3/python found on PATH -- did install_packages run first?"
            exit 1
        fi
        "$py_bin" -m venv "$VENV_DIR"
    fi
    # non-fatal: a flaky connection here shouldn't nuke a setup that already
    # got through the (much more expensive) llama.cpp build. Nothing else
    # needs installing regardless -- see comment below.
    "$VENV_DIR/bin/pip" install --upgrade pip >/dev/null 2>&1 \
        || warn "pip self-upgrade failed (non-fatal, continuing)"
    # greyfox_cli.py only uses the stdlib (including concurrent.futures,
    # threading, tempfile -- all standard since Python 3.2+), so nothing
    # else needs installing here regardless of Termux's Python version.
}

install_cli() {
    # Resolve the directory setup.sh itself lives in (handles both
    # `bash setup.sh` and `bash /path/to/setup.sh`; falls back to $PWD if
    # dirname can't be resolved, e.g. when piped via `curl ... | bash`).
    local script_dir
    script_dir="$(cd "$(dirname "$0")" 2>/dev/null && pwd)"
    [ -z "$script_dir" ] && script_dir="$PWD"

    local src=""
    if [ -f "$script_dir/greyfox_cli.py" ]; then
        src="$script_dir/greyfox_cli.py"
    elif [ -f "$PWD/greyfox_cli.py" ]; then
        src="$PWD/greyfox_cli.py"
    fi

    if [ -z "$src" ]; then
        err "Can't find greyfox_cli.py -- looked in:"
        err "  $script_dir/greyfox_cli.py"
        err "  $PWD/greyfox_cli.py"
        err ""
        err "setup.sh expects greyfox_cli.py to sit in the same directory as"
        err "itself. If you 'git clone'd the repo this should already be true --"
        err "make sure you're running setup.sh from inside the cloned folder"
        err "(e.g. 'cd ~/greyfox-src && bash setup.sh'), not a copy of setup.sh"
        err "elsewhere. If you downloaded the files individually rather than"
        err "cloning, put setup.sh and greyfox_cli.py in the same folder first."
        exit 1
    fi
    cp "$src" "$BASE_DIR/greyfox_cli.py"
    log "Installed greyfox_cli.py from $src"

    # Also copy setup.sh itself into $BASE_DIR: install_autostart()'s boot
    # hook and --run-only both invoke "$BASE_DIR/setup.sh", not the copy in
    # wherever the repo was originally cloned to. Without this, the very
    # first autostart/reboot after setup fails the same way, just for
    # setup.sh instead of greyfox_cli.py.
    local self_path="$script_dir/setup.sh"
    if [ ! -f "$self_path" ]; then
        # fall back to whatever $0 actually points at, in case the running
        # script isn't literally named setup.sh
        self_path="$0"
    fi
    if [ -f "$self_path" ]; then
        cp "$self_path" "$BASE_DIR/setup.sh"
        chmod +x "$BASE_DIR/setup.sh"
        log "Installed setup.sh to $BASE_DIR/setup.sh (used by autostart/--run-only)"
    else
        warn "Could not locate setup.sh itself to copy into $BASE_DIR -- autostart"
        warn "  and 'bash setup.sh --run-only' from $BASE_DIR may not work until you"
        warn "  re-run full setup from the cloned repo."
    fi
}

start_server() {
    mkdir -p "$LOG_DIR"
    if pgrep -f "llama-server" >/dev/null 2>&1; then
        log "llama-server already running."
        return
    fi
    if [ ! -x "$LLAMACPP_DIR/build/bin/llama-server" ]; then
        err "llama-server binary not found at $LLAMACPP_DIR/build/bin/llama-server."
        err "Run full setup first: bash setup.sh --setup-only"
        exit 1
    fi
    if [ ! -f "$MODELS_DIR/$MODEL_FILE" ]; then
        err "Model file not found at $MODELS_DIR/$MODEL_FILE."
        err "MODEL_FILE=\"$MODEL_FILE\" -- if that doesn't match what you actually"
        err "downloaded, re-run with the right one, e.g.:"
        err "  MODEL_FILE=your-model.gguf bash setup.sh --setup-only"
        exit 1
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
    log "Waiting for server to come up (a large model on a phone CPU can take a while)..."
    local startup_timeout="${SERVER_STARTUP_TIMEOUT:-120}"
    local i
    for i in $(seq 1 "$startup_timeout"); do
        if curl -s "http://127.0.0.1:$SERVER_PORT/health" >/dev/null 2>&1; then
            log "Server is up."
            return
        fi
        # fail fast instead of waiting out the full timeout if the process
        # already died (bad model file, OOM kill, port already in use, etc.)
        if ! kill -0 "$(cat "$BASE_DIR/server.pid" 2>/dev/null)" 2>/dev/null; then
            err "llama-server exited before coming up. Check $LOG_DIR/llama-server.log"
            tail -n 30 "$LOG_DIR/llama-server.log" >&2
            exit 1
        fi
        sleep 1
    done
    err "Server did not come up within ${startup_timeout}s. Check $LOG_DIR/llama-server.log"
    err "(if it's just a slow model load, try again with a longer SERVER_STARTUP_TIMEOUT)"
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
save_config
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
