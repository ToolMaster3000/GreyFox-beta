#!/usr/bin/env python3
"""
GreyFox-CLI
A local, on-device chat CLI for Android/Termux running any tool-calling-capable
GGUF model via llama.cpp, with a JSON-schema-constrained tool-calling loop.

Same tool surface as the project this is trimmed down from (web search, HTTP
requests, sandboxed file access, Python execution, keyword notes search, and
Termux:API integration) but WITHOUT the long-context/autonomous-task
reliability machinery (no scratchpad, invariants file, diff checkpointing,
subtask graph, context compaction, complexity classifier, reflection passes,
fact ledger, playbook cache, etc). /auto multi-turn autonomy is kept, in a
much simpler form: it just repeats the normal round loop across turns until
the model signals it's done, calls request_user_input, or a safety cap of
turns is hit.

Model choice is up to you -- point MODEL/server at whatever chat-template +
tool-calling-capable GGUF you like. This script doesn't hardcode a model.
"""

import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

HOME = Path.home()
BASE_DIR = HOME / "greyfox-cli"
WORKSPACE = BASE_DIR / "workspace"
SESSIONS_DIR = BASE_DIR / "sessions"

LLAMA_SERVER_URL = os.environ.get("GREYFOX_SERVER_URL", "http://127.0.0.1:8080")

MAX_TOOL_ROUNDS_PER_TURN = 8       # tool calls allowed within a single user turn
MAX_AUTO_TURNS = 8                  # safety cap for /auto, mirrors upstream default
RUN_PYTHON_TIMEOUT = 15             # seconds
HTTP_TIMEOUT = 20                   # seconds
TOOL_OUTPUT_CHAR_CAP = 4000         # truncate huge tool output before it hits context

DEFAULT_SYSTEM_PROMPT = """You are GreyFox, a concise on-device assistant running locally on an \
Android phone via llama.cpp. You have access to tools (see schema). Call a tool \
only when it's actually needed; otherwise just answer directly. When you are \
confident you have a complete final answer, respond with plain text and no tool \
calls. If you are running as part of an autonomous /auto task and you believe the \
goal is fully accomplished, end your final message with the exact line \
TASK_COMPLETE on its own. If you genuinely cannot proceed without the user \
(missing credential, an irreversible/destructive action, or a choice only they \
can make), call the request_user_input tool rather than guessing -- but for \
ordinary ambiguity, state a reasonable assumption and keep going."""

# --------------------------------------------------------------------------
# Tool schema (used both to build the constraining JSON schema sent to
# llama.cpp's server and to dispatch calls locally)
# --------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web (DuckDuckGo HTML) and return top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "http_request",
            "description": "Make an arbitrary HTTP request to a URL you construct.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]},
                    "url": {"type": "string"},
                    "headers": {"type": "string", "description": "Optional raw header lines, one per line, e.g. 'Content-Type: application/json'"},
                    "body": {"type": "string", "description": "Optional request body"},
                },
                "required": ["method", "url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the sandboxed workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (overwrite) a text file in the sandboxed workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files in a directory within the sandboxed workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative path, '.' for workspace root"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": f"Run a Python snippet in a sandboxed subprocess ({RUN_PYTHON_TIMEOUT}s timeout).",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Keyword-ranked (TF-IDF-ish) search over text files in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_api",
            "description": "Clipboard read/write or send a notification via the Termux:API app.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["clipboard_get", "clipboard_set", "notify"]},
                    "text": {"type": "string", "description": "Text for clipboard_set or notify"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_user_input",
            "description": "Stop and explicitly ask the user something you genuinely cannot proceed without "
                            "(a missing credential, a choice only they can make, or a destructive action). "
                            "Do not use this for ordinary ambiguity -- state an assumption and continue instead.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


# --------------------------------------------------------------------------
# Sandbox helpers
# --------------------------------------------------------------------------

class SandboxError(Exception):
    pass


def safe_path(rel_path: str) -> Path:
    """Resolve a path strictly inside WORKSPACE; reject traversal / absolute paths."""
    rel_path = rel_path or "."
    candidate = (WORKSPACE / rel_path).resolve()
    try:
        candidate.relative_to(WORKSPACE.resolve())
    except ValueError:
        raise SandboxError(f"path '{rel_path}' escapes the sandboxed workspace")
    return candidate


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------

def tool_web_search(args):
    query = args.get("query", "")
    q = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={q}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (GreyFox-CLI)"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"[web_search error] {e}"

    results = []
    for m in re.finditer(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>',
        html, re.S,
    ):
        link, title, snippet = m.groups()
        clean = lambda s: re.sub(r"<[^>]+>", "", s).strip()
        results.append(f"- {clean(title)}\n  {link}\n  {clean(snippet)}")
        if len(results) >= 5:
            break
    if not results:
        return "No results parsed (page structure may have changed, or no results found)."
    return "\n".join(results)


def tool_http_request(args):
    method = args.get("method", "GET").upper()
    url = args.get("url", "")
    headers = {}
    if args.get("headers"):
        for line in args["headers"].splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip()] = v.strip()
    data = args.get("body")
    data_bytes = data.encode("utf-8") if data else None
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return f"status={resp.status}\n{body}"
    except urllib.error.HTTPError as e:
        return f"status={e.code}\n{e.read().decode('utf-8', errors='replace')}"
    except Exception as e:
        return f"[http_request error] {e}"


def tool_read_file(args):
    try:
        p = safe_path(args.get("path", ""))
        if not p.is_file():
            return f"[read_file error] not a file: {args.get('path')}"
        return p.read_text(encoding="utf-8", errors="replace")
    except SandboxError as e:
        return f"[read_file error] {e}"
    except Exception as e:
        return f"[read_file error] {e}"


def tool_write_file(args):
    try:
        p = safe_path(args.get("path", ""))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args.get("content", ""), encoding="utf-8")
        return f"wrote {len(args.get('content', ''))} chars to {args.get('path')}"
    except SandboxError as e:
        return f"[write_file error] {e}"
    except Exception as e:
        return f"[write_file error] {e}"


def tool_list_directory(args):
    try:
        p = safe_path(args.get("path", "."))
        if not p.is_dir():
            return f"[list_directory error] not a directory: {args.get('path', '.')}"
        entries = sorted(os.listdir(p))
        return "\n".join(entries) if entries else "(empty)"
    except SandboxError as e:
        return f"[list_directory error] {e}"


def tool_run_python(args):
    code = args.get("code", "")
    script_path = WORKSPACE / "_tmp_run.py"
    try:
        script_path.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(WORKSPACE),
            capture_output=True,
            text=True,
            timeout=RUN_PYTHON_TIMEOUT,
        )
        out = proc.stdout
        if proc.stderr:
            out += f"\n[stderr]\n{proc.stderr}"
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[run_python error] timed out after {RUN_PYTHON_TIMEOUT}s"
    except Exception as e:
        return f"[run_python error] {e}"
    finally:
        script_path.unlink(missing_ok=True)


def tool_search_notes(args):
    query = args.get("query", "").lower()
    terms = [t for t in re.split(r"\W+", query) if t]
    if not terms:
        return "empty query"
    scored = []
    for path in WORKSPACE.rglob("*"):
        if path.is_file() and path.suffix in (".txt", ".md", ".json", ".log"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                continue
            score = sum(text.count(t) for t in terms)
            if score > 0:
                scored.append((score, path.relative_to(WORKSPACE)))
    scored.sort(reverse=True)
    if not scored:
        return "no matches"
    return "\n".join(f"{score}\t{path}" for score, path in scored[:10])


def tool_termux_api(args):
    action = args.get("action")
    try:
        if action == "clipboard_get":
            out = subprocess.run(["termux-clipboard-get"], capture_output=True, text=True, timeout=5)
            return out.stdout.strip()
        elif action == "clipboard_set":
            subprocess.run(["termux-clipboard-set"], input=args.get("text", ""), text=True, timeout=5)
            return "clipboard set"
        elif action == "notify":
            subprocess.run(["termux-notification", "--content", args.get("text", "")], timeout=5)
            return "notification sent"
        else:
            return f"[termux_api error] unknown action {action}"
    except FileNotFoundError:
        return "[termux_api error] Termux:API not installed"
    except Exception as e:
        return f"[termux_api error] {e}"


def tool_request_user_input(args):
    # Handled specially by the round loop (it halts the loop); this
    # implementation is only reached if something calls it directly.
    return f"[request_user_input] {args.get('question', '')}"


TOOL_IMPL = {
    "web_search": tool_web_search,
    "http_request": tool_http_request,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "list_directory": tool_list_directory,
    "run_python": tool_run_python,
    "search_notes": tool_search_notes,
    "termux_api": tool_termux_api,
    "request_user_input": tool_request_user_input,
}


def dispatch_tool(name, args):
    if name not in TOOL_IMPL:
        return f"[error] unknown tool '{name}'"
    try:
        result = TOOL_IMPL[name](args or {})
    except Exception as e:
        result = f"[tool error] {e}"
    result = str(result)
    if len(result) > TOOL_OUTPUT_CHAR_CAP:
        result = result[:TOOL_OUTPUT_CHAR_CAP] + f"\n...[truncated, {len(result) - TOOL_OUTPUT_CHAR_CAP} more chars]"
    return result


# --------------------------------------------------------------------------
# llama.cpp server client (OpenAI-compatible /v1/chat/completions, which
# llama-server exposes with native `tools` support -- this is what gives us
# structurally-valid tool calls without hand-rolling a GBNF grammar)
# --------------------------------------------------------------------------

def call_model(messages, tools=True, temperature=0.3):
    payload = {
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
    }
    if tools:
        payload["tools"] = TOOLS
        payload["tool_choice"] = "auto"

    req = urllib.request.Request(
        f"{LLAMA_SERVER_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach llama-server at {LLAMA_SERVER_URL} ({e}). "
            f"Is it running? (see setup.sh)"
        )
    choice = data["choices"][0]["message"]
    return choice


# --------------------------------------------------------------------------
# Round loop: one user turn, up to MAX_TOOL_ROUNDS_PER_TURN tool calls
# --------------------------------------------------------------------------

def run_turn(messages, spinner_label="thinking", quiet=False):
    """Runs tool-calling rounds until the model returns a plain text answer,
    a request_user_input call, or the round cap is hit. Mutates `messages`
    in place. Returns (final_text, stopped_reason)."""
    for round_i in range(MAX_TOOL_ROUNDS_PER_TURN):
        if not quiet:
            print(f"  ...{spinner_label} (round {round_i + 1}/{MAX_TOOL_ROUNDS_PER_TURN})", file=sys.stderr)
        choice = call_model(messages)
        tool_calls = choice.get("tool_calls") or []

        if not tool_calls:
            text = choice.get("content") or ""
            messages.append({"role": "assistant", "content": text})
            return text, "answered"

        # record the assistant's tool-call message, then execute each call
        messages.append({
            "role": "assistant",
            "content": choice.get("content") or "",
            "tool_calls": tool_calls,
        })

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "request_user_input":
                question = args.get("question", "(no question given)")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": "halted: waiting on user",
                })
                return question, "needs_user"

            if not quiet:
                print(f"  -> {name}({json.dumps(args)[:120]})", file=sys.stderr)
            result = dispatch_tool(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": result,
            })

    return "(round cap reached without a final answer this turn)", "round_cap"


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------

def save_session(name, messages):
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / f"{name}.json"
    path.write_text(json.dumps(messages, indent=2), encoding="utf-8")
    return path


def load_session(name):
    path = SESSIONS_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_sessions():
    if not SESSIONS_DIR.exists():
        return []
    return sorted(p.stem for p in SESSIONS_DIR.glob("*.json"))


# --------------------------------------------------------------------------
# /auto : simplified multi-turn autonomy
# --------------------------------------------------------------------------

def run_auto(messages, goal):
    """Repeats normal turns, feeding the model's own progress back to itself,
    until it emits TASK_COMPLETE, calls request_user_input, or MAX_AUTO_TURNS
    is hit. No todo.md, no checkpointing, no reflection pass -- just the loop."""
    messages.append({
        "role": "user",
        "content": (
            f"Autonomous task goal: {goal}\n\n"
            "Work toward this goal. You may need several of your own turns; "
            "each time you reply, either continue making progress or, if the "
            "goal is fully done, end your message with the line TASK_COMPLETE."
        ),
    })

    for turn_i in range(MAX_AUTO_TURNS):
        print(f"\n[/auto turn {turn_i + 1}/{MAX_AUTO_TURNS}]", file=sys.stderr)
        text, reason = run_turn(messages, spinner_label=f"auto turn {turn_i + 1}")
        print(text)

        if reason == "needs_user":
            print(f"\n[/auto stopped -- model needs input: {text}]")
            return

        if "TASK_COMPLETE" in text:
            print(f"\n[/auto finished in {turn_i + 1} turn(s)]")
            return

        # nudge it to keep going next turn
        messages.append({
            "role": "user",
            "content": "Continue toward the goal. If it's actually complete, "
                        "say so and end with TASK_COMPLETE.",
        })

    print(f"\n[/auto stopped: hit the {MAX_AUTO_TURNS}-turn safety cap]")


# --------------------------------------------------------------------------
# REPL
# --------------------------------------------------------------------------

VERSION = "0.1.0"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Original mascot -- plain ASCII, not derived from any other CLI's brand art.
MASCOT = [
    " /\\_/\\ ",
    "( o.o )",
    " > ^ < ",
]
MASCOT_COLOR = (100, 116, 139)  # slate grey

# Original gradient: slate grey -> silver -> ember orange (grey fox coat ->
# tail highlight). Not taken from any other project's theme.
GRADIENT = [(100, 116, 139), (203, 213, 225), (249, 115, 22)]

WORDMARK = " ".join("GREYFOX")


def _lerp(a, b, t):
    return round(a + (b - a) * t)


def _gradient_color(t, stops):
    """t in [0,1]; interpolates across a list of (r,g,b) stops."""
    if len(stops) == 1:
        return stops[0]
    seg = t * (len(stops) - 1)
    i = min(int(seg), len(stops) - 2)
    local_t = seg - i
    r0, g0, b0 = stops[i]
    r1, g1, b1 = stops[i + 1]
    return (_lerp(r0, r1, local_t), _lerp(g0, g1, local_t), _lerp(b0, b1, local_t))


def _gradient_text(text, stops, bold=False):
    if not text:
        return ""
    out = []
    n = max(len(text) - 1, 1)
    for i, ch in enumerate(text):
        r, g, b = _gradient_color(i / n, stops)
        prefix = BOLD if bold else ""
        out.append(f"{prefix}\033[38;2;{r};{g};{b}m{ch}{RESET}")
    return "".join(out)


def _gradient_rule(width, stops):
    return _gradient_text("─" * width, stops)


def _detect_ram_mb():
    """Reads total RAM directly from /proc/meminfo, same technique setup.sh
    uses for its own RAM-tiered context sizing -- purely informational here."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return None
    return None


def _ram_tier_ctx_size(ram_mb):
    """Mirrors setup.sh's pick_ram_tier() so the banner's displayed context
    size matches what the server was actually started with, without the
    caller needing to pass anything in."""
    if ram_mb is None:
        return None
    if ram_mb >= 7000:
        return 16384
    elif ram_mb >= 5500:
        return 10240
    else:
        return 6144


def _detect_model_label():
    """Asks the running llama-server what model it's actually serving via
    its OpenAI-compatible /v1/models endpoint. Falls back to a generic
    label if the server isn't up yet or doesn't respond in time -- the
    banner should never block startup waiting on this."""
    try:
        req = urllib.request.Request(f"{LLAMA_SERVER_URL}/v1/models")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        model_id = data.get("data", [{}])[0].get("id")
        if model_id:
            return os.path.basename(model_id)
    except Exception:
        pass
    return "local GGUF via llama.cpp"


def print_banner():
    """Boxed header: mascot, gradient wordmark, rule lines, info line --
    same overall structure as the project this is descended from, but with
    an original mascot and an original color scheme (not the other
    project's copied Claude Code mascot / Gemini CLI gradient). Fully
    self-contained: detects RAM tier and the actually-running model on its
    own, nothing needs to be passed in."""
    ctx_size = _ram_tier_ctx_size(_detect_ram_mb())
    model_label = _detect_model_label()

    mascot_lines = [f"\033[1;38;2;{MASCOT_COLOR[0]};{MASCOT_COLOR[1]};{MASCOT_COLOR[2]}m{line}{RESET}" for line in MASCOT]
    wordmark_line = _gradient_text(WORDMARK, GRADIENT, bold=True)
    rule_width = len(WORDMARK) + 2
    rule = _gradient_rule(rule_width, GRADIENT)

    info_bits = [f"{DIM}{model_label}{RESET}"]
    if ctx_size:
        info_bits.append(f"{DIM}context {RESET}{ctx_size} tokens")
    info_bits.append(f"{DIM}v{VERSION}{RESET}")
    info_line = f"  {DIM}·{RESET}  ".join(info_bits)

    border = f"\033[1;38;2;{MASCOT_COLOR[0]};{MASCOT_COLOR[1]};{MASCOT_COLOR[2]}m"
    print(f"{border}╭{'─' * (rule_width + 4)}╮{RESET}")
    for line in mascot_lines:
        print(f"{border}│{RESET}  {line}  {border}│{RESET}")
    print(f"{border}│{RESET}  {rule}  {border}│{RESET}")
    print(f"{border}│{RESET}  {wordmark_line}  {border}│{RESET}")
    print(f"{border}│{RESET}  {rule}  {border}│{RESET}")
    print(f"{border}╰{'─' * (rule_width + 4)}╯{RESET}")
    print()
    print(info_line)
    print(f"{DIM}› type your message   /help commands   /quit exit{RESET}")
    print(f"{DIM}tools: search, http, files, python, notes, termux -- used automatically when needed{RESET}")
    print()


HELP_TEXT = """\
Commands:
  /help                   show this list
  /reset                  clear conversation, keep system prompt
  /system <prompt>        replace the system prompt
  /save [name]            save the current conversation
  /load <name>            load a saved conversation
  /sessions               list saved conversations
  /regenerate             re-run the last user prompt for a fresh answer
  /auto <goal>            run turns autonomously (up to {max_auto} turns) until
                          the model says TASK_COMPLETE, needs your input, or the
                          safety cap is hit
  /stats                  show basic stats about the last response
  /quit                   exit
""".format(max_auto=MAX_AUTO_TURNS)


def new_conversation(system_prompt=DEFAULT_SYSTEM_PROMPT):
    return [{"role": "system", "content": system_prompt}]


def last_user_message(messages):
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return None


def main():
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    messages = new_conversation()
    last_stats = {}

    print_banner()
    print("type /help for commands, or just start chatting.")
    print(f"(talking to llama-server at {LLAMA_SERVER_URL})\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith("/"):
            parts = shlex.split(line)
            cmd = parts[0]

            if cmd == "/help":
                print(HELP_TEXT)

            elif cmd == "/reset":
                messages = new_conversation(messages[0]["content"])
                print("conversation cleared.")

            elif cmd == "/system":
                new_prompt = line[len("/system "):].strip()
                if not new_prompt:
                    print("usage: /system <prompt>")
                else:
                    messages[0] = {"role": "system", "content": new_prompt}
                    print("system prompt updated.")

            elif cmd == "/save":
                name = parts[1] if len(parts) > 1 else time.strftime("%Y%m%d-%H%M%S")
                path = save_session(name, messages)
                print(f"saved to {path}")

            elif cmd == "/load":
                if len(parts) < 2:
                    print("usage: /load <name>")
                else:
                    loaded = load_session(parts[1])
                    if loaded is None:
                        print(f"no session named '{parts[1]}'")
                    else:
                        messages = loaded
                        print(f"loaded '{parts[1]}' ({len(messages)} messages).")

            elif cmd == "/sessions":
                names = list_sessions()
                print("\n".join(names) if names else "(none saved yet)")

            elif cmd == "/regenerate":
                prev = last_user_message(messages)
                if prev is None:
                    print("nothing to regenerate yet.")
                else:
                    # drop everything after the last user message and retry
                    idx = max(i for i, m in enumerate(messages) if m["role"] == "user")
                    messages = messages[: idx + 1]
                    t0 = time.time()
                    text, reason = run_turn(messages)
                    last_stats = {"elapsed": time.time() - t0, "reason": reason}
                    print(text)

            elif cmd == "/auto":
                goal = line[len("/auto "):].strip()
                if not goal:
                    print("usage: /auto <goal>")
                else:
                    run_auto(messages, goal)

            elif cmd == "/stats":
                print(json.dumps(last_stats, indent=2) if last_stats else "(no stats yet)")

            elif cmd == "/quit":
                break

            else:
                print(f"unknown command '{cmd}', try /help")
            continue

        # plain chat turn
        messages.append({"role": "user", "content": line})
        t0 = time.time()
        try:
            text, reason = run_turn(messages)
        except RuntimeError as e:
            print(f"[error] {e}")
            messages.pop()
            continue
        last_stats = {"elapsed": round(time.time() - t0, 1), "reason": reason}
        print(text)
        if reason == "needs_user":
            print("(model is waiting on you -- see request above)")


if __name__ == "__main__":
    main()
