#!/usr/bin/env python3
"""
GreyFox-CLI
A local, on-device chat CLI for Android/Termux running any tool-calling-capable
GGUF model via llama.cpp, with a JSON-schema-constrained tool-calling loop.

Same tool surface as the project this is trimmed down from (web search, HTTP
requests, sandboxed file access, Python execution, keyword notes search, and
Termux:API integration), now with a slimmed-down version of the long-context/
autonomous-task reliability machinery re-added: an invariants.md file that's
always reinjected, context compaction once the transcript gets big, a small
structured fact ledger, concurrent dispatch of independent tool calls, and
failure escalation instead of blind retries. /auto multi-turn autonomy stays
simple: it repeats the normal round loop across turns until the model signals
it's done, calls request_user_input, or a safety cap of turns is hit.

Model choice is up to you -- point MODEL/server at whatever chat-template +
tool-calling-capable GGUF you like. This script doesn't hardcode a model.
"""

import atexit
import concurrent.futures
import difflib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import readline  # noqa: F401  -- importing enables arrow-key history/editing for input()
except ImportError:
    readline = None  # not available on some minimal builds; input() just degrades to no history

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

HOME = Path.home()
BASE_DIR = HOME / "greyfox-cli"
WORKSPACE = BASE_DIR / "workspace"
SESSIONS_DIR = BASE_DIR / "sessions"
HISTORY_FILE = BASE_DIR / "history"  # readline input history, persisted across sessions

LLAMA_SERVER_URL = os.environ.get("GREYFOX_SERVER_URL", "http://127.0.0.1:8080")

MAX_TOOL_ROUNDS_PER_TURN = 8       # tool calls allowed within a single user turn
MAX_AUTO_TURNS = 8                  # safety cap for /auto, mirrors upstream default
RUN_PYTHON_TIMEOUT = 15             # seconds
HTTP_TIMEOUT = 20                   # seconds
TOOL_OUTPUT_CHAR_CAP = 4000         # truncate huge tool output before it hits context

# -- context compaction -------------------------------------------------------
INVARIANTS_PATH = BASE_DIR / "invariants.md"
FACTS_PATH = BASE_DIR / "facts.json"
DEFAULT_CTX_SIZE = 8192            # fallback if we can't detect the server's n_ctx
COMPACT_TRIGGER_FRACTION = 0.70    # compact once estimated usage crosses this
COMPACT_KEEP_RECENT = 10           # most recent messages kept verbatim, always
CHARS_PER_TOKEN_ESTIMATE = 4       # rough char/token ratio, no local tokenizer

# -- /auto plan tracking --------------------------------------------------------
# Regenerated at the start of every /auto run (see capture_auto_plan). Kept
# outside the compactable history and outside invariants.md -- these are
# scoped to the current run, not permanent facts about the user/project.
PLAN_PATH = BASE_DIR / "plan.md"
AUTO_GOAL_PATH = BASE_DIR / "auto_goal.md"  # "definition of done" for the current run

# -- failure escalation --------------------------------------------------------
FAILURE_ESCALATION_THRESHOLD = 2   # consecutive same-tool failures before a nudge

# -- concurrent dispatch --------------------------------------------------------
TOOL_THREAD_POOL_SIZE = 4
IO_BOUND_TOOLS = {"web_search", "http_request"}  # benefit most from parallelism

# per-tool label used by the phase spinner / batched progress lines
PHASE_LABELS = {
    "web_search": "searching",
    "http_request": "http request",
    "read_file": "reading file",
    "write_file": "writing file",
    "list_directory": "listing directory",
    "run_python": "running python",
    "search_notes": "searching notes",
    "termux_api": "termux api",
    "record_fact": "recording fact",
    "query_facts": "querying facts",
    "update_plan": "updating plan",
    "request_user_input": "waiting on user",
}

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
            "name": "record_fact",
            "description": "Record a short durable fact/decision in the structured fact ledger "
                            "(separate from conversation history, so it survives compaction and "
                            "/auto turns). Use for things worth remembering across a long task: "
                            "decisions made, IDs, paths, results of subtasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Short label, e.g. 'db_password_location'"},
                    "value": {"type": "string", "description": "The fact itself, kept short"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_facts",
            "description": "Search the structured fact ledger recorded via record_fact. "
                            "Use an empty query to list everything recorded so far.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Substring to match against keys/values; empty lists all"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": "Write or update the ordered step-by-step plan for the current /auto run. "
                            "Call this once near the start of a multi-step task with the FULL ordered "
                            "list of steps, and again any time a step's status changes or the plan "
                            "needs to be revised -- always pass the complete list, not just the change, "
                            "since this replaces the whole plan. Mark a step's 'done' true once it's "
                            "actually finished, not when you start it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Short description of this step"},
                                "done": {"type": "boolean", "description": "Whether this step is complete"},
                            },
                            "required": ["text", "done"],
                        },
                    },
                },
                "required": ["steps"],
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
    # unique filename (not a fixed "_tmp_run.py") so concurrent run_python
    # calls in the same round don't stomp on each other
    fd, tmp_name = tempfile.mkstemp(prefix="_tmp_run_", suffix=".py", dir=str(WORKSPACE))
    os.close(fd)
    script_path = Path(tmp_name)
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


def _load_facts():
    if not FACTS_PATH.exists():
        return []
    try:
        return json.loads(FACTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_facts(facts):
    FACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FACTS_PATH.write_text(json.dumps(facts, indent=2), encoding="utf-8")


def tool_record_fact(args):
    key = (args.get("key") or "").strip()
    value = (args.get("value") or "").strip()
    if not key:
        return "[record_fact error] 'key' is required"
    facts = _load_facts()
    facts.append({"key": key, "value": value, "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
    _save_facts(facts)
    return f"recorded fact '{key}' ({len(facts)} facts total)"


def tool_query_facts(args):
    query = (args.get("query") or "").strip().lower()
    facts = _load_facts()
    if not facts:
        return "no facts recorded yet"
    if query:
        facts = [f for f in facts if query in f["key"].lower() or query in f["value"].lower()]
        if not facts:
            return f"no facts match '{query}'"
    lines = [f"[{f['ts']}] {f['key']}: {f['value']}" for f in facts[-50:]]
    return "\n".join(lines)


def _render_plan(steps):
    lines = ["# Current plan (this /auto run)"]
    for s in steps:
        box = "x" if s.get("done") else " "
        lines.append(f"- [{box}] {s.get('text', '')}")
    return "\n".join(lines)


def tool_update_plan(args):
    steps = args.get("steps")
    if not isinstance(steps, list) or not steps:
        return "[update_plan error] 'steps' must be a non-empty list of {text, done} objects"
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(_render_plan(steps), encoding="utf-8")
    done = sum(1 for s in steps if s.get("done"))
    return f"plan updated: {done}/{len(steps)} steps done"


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
    "record_fact": tool_record_fact,
    "query_facts": tool_query_facts,
    "update_plan": tool_update_plan,
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


_TOOL_ERROR_RE = re.compile(r"^\[[^\]]*error", re.IGNORECASE)


def is_tool_error(result):
    return bool(_TOOL_ERROR_RE.match(result or ""))


# per-tool consecutive-failure counter, used for failure escalation below.
# Reset on /reset and whenever a tool call for that name succeeds.
TOOL_FAIL_COUNTS = {}


def note_tool_outcome(name, result, messages):
    """Track consecutive failures per tool. Blindly retrying a broken call is
    how small models grind a task into the ground -- after
    FAILURE_ESCALATION_THRESHOLD failures in a row for the same tool, nudge
    the model toward a different approach instead of letting it keep hammering
    the same call. Re-nudges every couple of failures after that if it keeps
    going."""
    if is_tool_error(result):
        TOOL_FAIL_COUNTS[name] = TOOL_FAIL_COUNTS.get(name, 0) + 1
        count = TOOL_FAIL_COUNTS[name]
        if count >= FAILURE_ESCALATION_THRESHOLD and (count - FAILURE_ESCALATION_THRESHOLD) % 2 == 0:
            messages.append({
                "role": "system",
                "content": (
                    f"Note: '{name}' has now failed {count} times in a row with similar-looking "
                    "errors. Stop retrying the same call with the same arguments -- try different "
                    "arguments, a different tool, or call request_user_input if you're genuinely stuck."
                ),
            })
    else:
        TOOL_FAIL_COUNTS[name] = 0


# --------------------------------------------------------------------------
# llama.cpp server client (OpenAI-compatible /v1/chat/completions, which
# llama-server exposes with native `tools` support -- this is what gives us
# structurally-valid tool calls without hand-rolling a GBNF grammar)
# --------------------------------------------------------------------------

def call_model(messages, tools=True, temperature=0.3):
    """Non-streaming call. Used for internal/background calls that shouldn't
    print anything (context-compaction summaries, etc) -- the user-facing
    path is call_model_streaming, below."""
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


class StreamingToolsUnsupported(RuntimeError):
    """Raised when llama-server rejects stream+tools together (older builds:
    'Cannot use tools with stream'). Caught by run_turn to fall back to a
    non-streaming call for the rest of the session."""


def call_model_streaming(messages, tools=True, temperature=0.3, print_tokens=True,
                          on_first_chunk=None, on_first_content=None, renderer=None):
    """Streams /v1/chat/completions (llama-server's OpenAI-compatible SSE
    stream) and prints content token-by-token as it arrives instead of
    blocking until the full reply is generated -- the difference between a
    long silent wait and visible progress on a 3-8 tok/s phone CPU.

    If a `renderer` (MarkdownStreamRenderer) is given, content tokens are fed
    through it instead of printed raw, so headers/bold/code get styled live.
    `on_first_content` fires once, right before the first real content
    character is emitted (used to print the assistant chat-chrome prefix
    lazily -- only once we know this round actually has visible text, not
    just tool calls).

    Tool-call deltas arrive incrementally too (indexed, OpenAI-style) and are
    accumulated here rather than printed. Returns (choice, stats) where choice
    has the same shape as call_model()'s return value and stats carries
    elapsed/tokens/tok-per-sec for the live readout.
    """
    payload = {
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "stream": True,
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

    content_parts = []
    tool_calls_acc = {}
    t0 = time.time()
    first_token_time = None
    chunk_seen = False
    content_started = False

    try:
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    if not chunk_seen:
                        chunk_seen = True
                        if on_first_chunk:
                            on_first_chunk()
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    token_text = delta.get("content")
                    if token_text:
                        if first_token_time is None:
                            first_token_time = time.time()
                        content_parts.append(token_text)
                        if print_tokens:
                            if not content_started:
                                content_started = True
                                if on_first_content:
                                    on_first_content()
                            if renderer is not None:
                                renderer.feed(token_text)
                            else:
                                print(token_text, end="", flush=True)
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = tool_calls_acc.setdefault(
                            idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if tools and "cannot use tools with stream" in body.lower():
                # older llama-server builds (pre streaming+tool_calls support)
                # reject this combination outright rather than degrading gracefully
                raise StreamingToolsUnsupported(
                    "This llama-server build doesn't support streaming together with tool "
                    "calling. Update llama.cpp (bump LLAMACPP_PIN in setup.sh and "
                    "--force-rebuild) for streamed output; falling back to non-streaming "
                    "for now."
                )
            raise RuntimeError(f"llama-server returned HTTP {e.code}: {body[:300]}")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach llama-server at {LLAMA_SERVER_URL} ({e}). "
                f"Is it running? (see setup.sh)"
            )
    finally:
        # if the connection dies mid-response, an open bold/code-block style
        # would otherwise leave the terminal stuck in that state
        if renderer is not None:
            renderer.finish()

    elapsed = time.time() - t0
    content = "".join(content_parts)
    tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
    tokens_est = max(len(content) // CHARS_PER_TOKEN_ESTIMATE, 0)
    gen_elapsed = elapsed - (first_token_time - t0) if first_token_time else elapsed
    tok_per_sec = (tokens_est / gen_elapsed) if gen_elapsed > 0.05 and tokens_est else 0.0

    choice = {"content": content, "tool_calls": tool_calls}
    stats = {"elapsed": round(elapsed, 2), "tokens_est": tokens_est, "tok_per_sec": round(tok_per_sec, 1)}
    return choice, stats


# --------------------------------------------------------------------------
# Context management: invariants (always reinjected) + compaction (older
# turns summarized once the transcript gets big)
# --------------------------------------------------------------------------

CTX_SIZE = DEFAULT_CTX_SIZE  # refined at startup by resolve_ctx_size()
STREAMING_SUPPORTED = True  # flips to False for the rest of the session if the
                             # server rejects stream+tools (see StreamingToolsUnsupported)


def detect_server_ctx_size():
    """Ask llama-server's /props endpoint for the context size it was
    actually started with, if it exposes one."""
    try:
        req = urllib.request.Request(f"{LLAMA_SERVER_URL}/props")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        n_ctx = (data.get("default_generation_settings") or {}).get("n_ctx") or data.get("n_ctx")
        if n_ctx:
            return int(n_ctx)
    except Exception:
        pass
    return None


def resolve_ctx_size():
    return detect_server_ctx_size() or _ram_tier_ctx_size(_detect_ram_mb()) or DEFAULT_CTX_SIZE


def estimate_tokens(messages):
    """Rough char/4 estimate -- there's no local tokenizer available, but this
    is close enough to trigger compaction at roughly the right point."""
    total_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    return total_chars // CHARS_PER_TOKEN_ESTIMATE


def load_invariants():
    """invariants.md: small set of load-bearing facts/constraints the model
    must never lose, kept OUT of the compactable message history and
    reinjected fresh on every call so compaction can never drop it."""
    if not INVARIANTS_PATH.exists():
        INVARIANTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        INVARIANTS_PATH.write_text(
            "# Invariants\n"
            "# Load-bearing facts/constraints the model must never lose or contradict.\n"
            "# One per line (blank lines and '#' comments are ignored). This file is\n"
            "# reinjected fresh every turn and is never summarized away by compaction.\n"
            "#\n"
            "# Example:\n"
            "# - Never delete or overwrite files outside the workspace sandbox.\n",
            encoding="utf-8",
        )
        return ""
    lines = [
        line for line in INVARIANTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return "\n".join(lines).strip()


def invariants_message():
    text = load_invariants()
    if not text:
        return None
    return {"role": "system", "content": f"Invariants (always true -- do not lose or contradict these):\n{text}"}


def plan_message():
    """Reinjected fresh every call, same as invariants -- the plan is
    control-flow-critical for a long /auto run and must survive compaction."""
    if not PLAN_PATH.exists():
        return None
    text = PLAN_PATH.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return {
        "role": "system",
        "content": f"{text}\n\n(Call update_plan with the full step list whenever a step's status "
                   f"changes, or to revise the plan -- always pass the complete list.)",
    }


def definition_of_done_message():
    """The explicit success criteria captured once at the start of an /auto
    run (see capture_auto_plan). Reinjected every turn so 'am I actually
    done?' is checked against a written contract, not just the model's
    in-context judgment after a dozen turns of drift."""
    if not AUTO_GOAL_PATH.exists():
        return None
    text = AUTO_GOAL_PATH.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return {"role": "system", "content": text}


def effective_messages(messages):
    """The list actually sent to the model: stored history plus freshly
    re-read invariants/definition-of-done/plan blocks inserted right after
    the system prompt. Doesn't mutate or grow `messages` itself."""
    extras = [m for m in (invariants_message(), definition_of_done_message(), plan_message()) if m]
    if not extras:
        return messages
    return [messages[0]] + extras + messages[1:]


def _render_for_summary(chunk):
    lines = []
    for m in chunk:
        role = m.get("role")
        content = m.get("content") or ""
        if m.get("tool_calls"):
            calls = ", ".join(
                f"{c.get('function', {}).get('name')}({(c.get('function', {}).get('arguments') or '')[:80]})"
                for c in m["tool_calls"]
            )
            lines.append(f"[{role} called] {calls}")
        elif content:
            lines.append(f"[{role}] {content[:500]}")
    return "\n".join(lines)


def summarize_chunk(chunk):
    """One cheap extra call: compress an old slice of the transcript into
    terse lines, keeping concrete facts/decisions/numbers and dropping
    filler."""
    rendered = _render_for_summary(chunk)
    if not rendered.strip():
        return "(nothing substantive)"
    prompt_messages = [
        {
            "role": "system",
            "content": (
                "Compress the following conversation excerpt into terse factual bullet "
                "points. Preserve concrete facts, numbers, decisions, file paths, URLs, "
                "and outcomes of tool calls. Discard filler and chit-chat. Max 15 short "
                "lines, no preamble."
            ),
        },
        {"role": "user", "content": rendered[:8000]},
    ]
    try:
        choice = call_model(prompt_messages, tools=False, temperature=0.0)
        return (choice.get("content") or "").strip() or "(summary was empty)"
    except Exception:
        return None  # signal failure distinctly -- see maybe_compact


def maybe_compact(messages):
    """At ~COMPACT_TRIGGER_FRACTION of context, summarize older tool-result
    turns into short lines via one cheap extra call. System prompt and the
    most recent COMPACT_KEEP_RECENT messages stay verbatim; invariants live
    outside this entirely (see effective_messages). Mutates `messages` in
    place. Returns True if it compacted."""
    if len(messages) <= COMPACT_KEEP_RECENT + 2:
        return False
    if estimate_tokens(messages) < CTX_SIZE * COMPACT_TRIGGER_FRACTION:
        return False
    system = messages[0]

    # A fixed-size cut can land in the middle of an assistant tool_calls
    # message and its tool-result response(s) -- if the assistant message
    # ends up summarized away while its "tool" result stays in the verbatim
    # "recent" window, that result references a tool_call_id nothing in the
    # transcript declares anymore, which servers reject as a malformed
    # conversation on the very next call. Grow the kept window backward
    # until it doesn't start on an orphaned "tool" message.
    split = len(messages) - COMPACT_KEEP_RECENT
    while split < len(messages) and messages[split].get("role") == "tool":
        split += 1
    if split >= len(messages) or split <= 1:
        return False  # nothing sensible left to compact this round

    recent = messages[split:]
    old = messages[1:split]
    if not old:
        return False
    summary = summarize_chunk(old)
    if summary is None:
        # the summarization call itself failed (e.g. transient network issue)
        # -- skip compacting this round rather than deleting real history for
        # nothing; estimate_tokens will still be over threshold next round,
        # so this just retries rather than losing data permanently.
        return False
    compacted = {
        "role": "system",
        "content": f"[compacted summary of {len(old)} earlier messages, replacing verbatim history]\n{summary}",
    }
    messages[:] = [system, compacted] + recent
    return True


# --------------------------------------------------------------------------
# TUI: phase spinner
# --------------------------------------------------------------------------

class Spinner:
    """Background-thread spinner for a single in-flight operation (model
    "thinking" before the first streamed token, or a lone tool call). Shows a
    phase label (searching / reading file / running python / ...) rather than
    a generic "thinking" line, so a long tool chain is easy to follow."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    COLOR_BY_PHASE = {
        "searching": "\033[38;2;96;165;250m",       # blue
        "http request": "\033[38;2;96;165;250m",
        "reading file": "\033[38;2;250;204;21m",      # yellow
        "writing file": "\033[38;2;250;204;21m",
        "listing directory": "\033[38;2;250;204;21m",
        "running python": "\033[38;2;74;222;128m",    # green
        "searching notes": "\033[38;2;250;204;21m",
        "termux api": "\033[38;2;192;132;252m",       # purple
        "recording fact": "\033[38;2;192;132;252m",
        "querying facts": "\033[38;2;192;132;252m",
    }

    def __init__(self, label):
        self.label = label
        self.color = self.COLOR_BY_PHASE.get(label, DIM)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        i = 0
        t0 = time.time()
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            elapsed = time.time() - t0
            print(f"\r  {self.color}{frame} {self.label}{RESET} {DIM}({elapsed:.1f}s){RESET}   ",
                  end="", file=sys.stderr, flush=True)
            i += 1
            time.sleep(0.08)

    def stop(self, final=None):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.3)
        print("\r" + " " * 90 + "\r", end="", file=sys.stderr, flush=True)
        if final:
            print(final, file=sys.stderr)


class MarkdownStreamRenderer:
    """Turns a stream of raw text chunks into ANSI-styled terminal output as
    they arrive: '# headers', **bold**, `inline code`, and ```fenced code```
    blocks get styled; everything else passes through unchanged.

    Only active when COLOR_ENABLED -- when color is off this is a pure
    passthrough, so a piped/logged transcript keeps the raw markdown source
    intact instead of losing information to hidden styling characters.

    Built for streaming: feed() can be called repeatedly with arbitrary
    chunk boundaries -- including a chunk that splits "**" or "```" right
    down the middle -- by holding back not-yet-resolvable characters until
    either more data arrives or finish() is called.
    """

    CODE_COLOR = (74, 222, 128)        # matches the "running python" phase color
    INLINE_CODE_COLOR = (96, 165, 250)  # matches the "searching"/http phase color

    def __init__(self, file=None):
        self.file = file or sys.stdout
        self.active = COLOR_ENABLED
        self._buf = ""
        self._at_line_start = True
        self._in_bold = False
        self._in_code_span = False
        self._in_code_block = False
        self._in_header = False
        self._fence_lang_mode = False

    def feed(self, text):
        if not text:
            return
        if not self.active:
            self.file.write(text)
            self.file.flush()
            return
        self._buf += text
        self._drain(final=False)
        self.file.flush()

    def finish(self):
        if self.active:
            self._drain(final=True)
            if self._in_bold or self._in_code_span or self._in_code_block or self._in_header:
                self.file.write(RESET)
                self._in_bold = self._in_code_span = self._in_code_block = self._in_header = False
        self.file.flush()

    def _drain(self, final):
        buf = self._buf
        n = len(buf)
        i = 0
        while i < n:
            if self._at_line_start:
                ch = buf[i]
                if self._in_code_block:
                    # only thing that matters at line-start inside a fence
                    # is watching for the closing ```
                    if ch == "`":
                        if n - i < 3 and not final:
                            break
                        if buf[i:i + 3] == "```":
                            self.file.write(f"{RESET}{DIM}```{RESET}")
                            self._in_code_block = False
                            i += 3
                            self._at_line_start = False
                            continue
                        # a content line that just happens to start with a
                        # backtick -- literal, not a fence, don't toggle
                        # inline-code state (that's meaningless inside a block)
                        self.file.write(ch)
                        i += 1
                        self._at_line_start = False
                        continue
                    self._at_line_start = False
                    continue
                # not in a code block -- check for an opening fence or a header
                if ch == "`":
                    if n - i < 3 and not final:
                        break
                    if buf[i:i + 3] == "```":
                        self.file.write(f"{DIM}```{RESET}")
                        self._in_code_block = True
                        self._fence_lang_mode = True
                        i += 3
                        self._at_line_start = False
                        continue
                    self._at_line_start = False
                    continue  # fall through to normal inline handling below
                if ch == "#":
                    j = i
                    while j < n and buf[j] == "#" and (j - i) < 6:
                        j += 1
                    if j == n:
                        if not final:
                            break  # more '#'s (or the deciding space) might still be coming
                        self.file.write(buf[i:j])
                        i = j
                        self._at_line_start = False
                        continue
                    if buf[j] == " " and (j - i) <= 6:
                        self.file.write(_fg(*ACCENT_COLOR, bold=True))
                        self._in_header = True
                        i = j + 1
                        self._at_line_start = False
                        continue
                    self.file.write(buf[i:j])  # e.g. "#hashtag" or 7+ '#'s -- not a header
                    i = j
                    self._at_line_start = False
                    continue
                self._at_line_start = False
                continue

            if self._fence_lang_mode:
                ch = buf[i]
                if ch == "\n":
                    self._fence_lang_mode = False
                    self.file.write(f"\n{_fg(*self.CODE_COLOR)}")
                    self._at_line_start = True
                else:
                    self.file.write(f"{DIM}{ch}{RESET}")
                i += 1
                continue

            ch = buf[i]
            if self._in_code_block:
                # code content: no markdown interpretation at all, but do
                # still watch for the newline that puts us back at line-start
                self.file.write(ch)
                if ch == "\n":
                    self._at_line_start = True
                i += 1
                continue

            if ch == "\n":
                if self._in_header:
                    self.file.write(RESET)
                    self._in_header = False
                self.file.write("\n")
                self._at_line_start = True
                i += 1
                continue

            if self._in_code_span:
                if ch == "`":
                    self.file.write(f"`{RESET}")
                    self._in_code_span = False
                else:
                    self.file.write(ch)
                i += 1
                continue

            if ch == "*":
                if i + 1 >= n and not final:
                    break  # might become "**" with the next chunk
                if i + 1 < n and buf[i + 1] == "*":
                    self._in_bold = not self._in_bold
                    self.file.write(BOLD if self._in_bold else "\033[22m")
                    i += 2
                else:
                    self.file.write("*")  # lone asterisk -- not bold syntax
                    i += 1
                continue

            if ch == "`":
                self.file.write(f"{_fg(*self.INLINE_CODE_COLOR)}`")
                self._in_code_span = True
                i += 1
                continue

            self.file.write(ch)
            i += 1

        self._buf = buf[i:]


# --------------------------------------------------------------------------
# Round loop: one user turn, up to MAX_TOOL_ROUNDS_PER_TURN tool calls
# --------------------------------------------------------------------------

def _assistant_prefix():
    """Small colored chat-chrome marker printed once, right before the
    model's actual reply text starts (never before a tool-calling round,
    which has no visible text of its own) -- makes it visually obvious where
    the assistant's turn begins, the same way a chat app would."""
    return f"{_fg(*ACCENT_COLOR, bold=True)}●{RESET} "


def _call_non_streaming_as_choice(send_messages, quiet):
    """Fallback path used when streaming+tools isn't supported by the
    server: one blocking call, printed as a single block instead of
    token-by-token, but still functional (and still markdown-styled)."""
    t0 = time.time()
    raw = call_model(send_messages)
    choice = {"content": raw.get("content") or "", "tool_calls": raw.get("tool_calls") or []}
    if not quiet and choice["content"]:
        print(_assistant_prefix(), end="")
        renderer = MarkdownStreamRenderer()
        renderer.feed(choice["content"])
        renderer.finish()
        print()
    tokens_est = max(len(choice["content"]) // CHARS_PER_TOKEN_ESTIMATE, 0)
    elapsed = time.time() - t0
    stats = {
        "elapsed": round(elapsed, 2),
        "tokens_est": tokens_est,
        "tok_per_sec": round(tokens_est / elapsed, 1) if elapsed > 0.05 and tokens_est else 0.0,
    }
    return choice, stats


def run_turn(messages, spinner_label="thinking", quiet=False):
    """Runs tool-calling rounds until the model returns a plain text answer,
    a request_user_input call, or the round cap is hit. Mutates `messages`
    in place. Returns (final_text, stopped_reason, stats)."""
    last_stats = {}
    for round_i in range(MAX_TOOL_ROUNDS_PER_TURN):
        compacted = maybe_compact(messages)
        if compacted and not quiet:
            print(
                f"  {DIM}[context compacted -- was over {int(COMPACT_TRIGGER_FRACTION * 100)}% "
                f"of ~{CTX_SIZE} tokens]{RESET}",
                file=sys.stderr,
            )

        send_messages = effective_messages(messages)

        spinner = None
        first_chunk_seen = threading.Event()
        if not quiet:
            spinner = Spinner(f"{spinner_label} (round {round_i + 1}/{MAX_TOOL_ROUNDS_PER_TURN})")
            spinner.start()

        def _on_first_chunk():
            if spinner and not first_chunk_seen.is_set():
                first_chunk_seen.set()
                spinner.stop()

        def _on_first_content():
            # fires only once real reply text starts arriving (not for
            # tool-calling rounds, which have no visible text) -- the chat
            # chrome marker belongs right before the actual answer, not at
            # the top of every internal tool round
            if not quiet:
                print(_assistant_prefix(), end="", flush=True)

        renderer = MarkdownStreamRenderer() if not quiet else None

        global STREAMING_SUPPORTED
        try:
            if STREAMING_SUPPORTED:
                try:
                    choice, stats = call_model_streaming(
                        send_messages, print_tokens=not quiet,
                        on_first_chunk=_on_first_chunk if spinner else None,
                        on_first_content=_on_first_content,
                        renderer=renderer,
                    )
                except StreamingToolsUnsupported as e:
                    STREAMING_SUPPORTED = False
                    if spinner and not first_chunk_seen.is_set():
                        spinner.stop()
                        spinner = None
                    if not quiet:
                        print_callout("warning", str(e), file=sys.stderr)
                    choice, stats = _call_non_streaming_as_choice(send_messages, quiet)
            else:
                # already fell back earlier this session -- go straight to
                # non-streaming so we're not re-triggering the same 400/500
                choice, stats = _call_non_streaming_as_choice(send_messages, quiet)
        finally:
            if spinner and not first_chunk_seen.is_set():
                spinner.stop()
        last_stats = stats

        tool_calls = choice.get("tool_calls") or []

        if not tool_calls:
            text = choice.get("content") or ""
            messages.append({"role": "assistant", "content": text})
            if not quiet:
                if text:
                    print()  # newline after streamed content
                usage_pct = min(estimate_tokens(messages) / CTX_SIZE * 100, 999) if CTX_SIZE else 0
                print(
                    f"  {DIM}[{stats['tokens_est']} tok · {stats['tok_per_sec']:.1f} tok/s · "
                    f"ctx ~{usage_pct:.0f}% used]{RESET}",
                    file=sys.stderr,
                )
            return text, "answered", last_stats

        messages.append({
            "role": "assistant",
            "content": choice.get("content") or "",
            "tool_calls": tool_calls,
        })

        calls_to_run = []
        request_user_call = None
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls_to_run.append((call, name, args))
            if name == "request_user_input" and request_user_call is None:
                request_user_call = (call, name, args)

        if request_user_call is not None:
            # Every tool_call declared in the assistant message above needs a
            # matching tool-result message, or the next call_model request is
            # a malformed conversation the server will likely reject outright.
            # If request_user_input showed up alongside other tool calls in
            # the same round, stub out results for the others instead of
            # silently dropping them -- we're handing control back to the
            # user, not actually running those calls.
            for call, name, args in calls_to_run:
                if call is request_user_call[0]:
                    content = "halted: waiting on user"
                else:
                    content = "skipped: user input was requested elsewhere in this round"
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": content,
                })
            question = request_user_call[2].get("question", "(no question given)")
            return question, "needs_user", last_stats

        if not calls_to_run:
            continue

        if len(calls_to_run) > 1:
            # Independent tool calls in the same round -- dispatch concurrently
            # on a thread pool. web_search/http_request are I/O-bound so this
            # is a real wall-clock win; the other tools are cheap enough that
            # running them alongside costs nothing. Batched ✓/… progress is
            # printed as each call finishes rather than as one blob at the end.
            if not quiet:
                for _call, name, args in calls_to_run:
                    label = PHASE_LABELS.get(name, name)
                    print(f"  … {label}: {name}({json.dumps(args)[:100]})", file=sys.stderr)
            results_by_id = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(TOOL_THREAD_POOL_SIZE, len(calls_to_run))
            ) as pool:
                future_map = {
                    pool.submit(dispatch_tool, name, args): (call, name)
                    for call, name, args in calls_to_run
                }
                for future in concurrent.futures.as_completed(future_map):
                    call, name = future_map[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = f"[tool error] {e}"
                    results_by_id[call.get("id", "")] = result
                    if not quiet:
                        mark = "✗" if is_tool_error(result) else "✓"
                        print(f"  {mark} {PHASE_LABELS.get(name, name)} done", file=sys.stderr)
            for call, name, _args in calls_to_run:
                result = results_by_id[call.get("id", "")]
                note_tool_outcome(name, result, messages)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": result,
                })
        else:
            call, name, args = calls_to_run[0]
            tool_spinner = None
            if not quiet:
                tool_spinner = Spinner(PHASE_LABELS.get(name, name))
                tool_spinner.start()
            try:
                result = dispatch_tool(name, args)
            finally:
                if tool_spinner:
                    mark = "✗" if is_tool_error(result) else "✓"
                    tool_spinner.stop(
                        final=f"  {mark} {PHASE_LABELS.get(name, name)}: {name}({json.dumps(args)[:100]})"
                    )
            note_tool_outcome(name, result, messages)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": result,
            })

    return "(round cap reached without a final answer this turn)", "round_cap", last_stats


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------

def _safe_session_name(name):
    """Session names become bare filenames under SESSIONS_DIR -- reject
    anything that could do path arithmetic out of that directory (slashes,
    '..', null bytes) rather than silently combining untrusted input into
    a Path and writing wherever that resolves to."""
    name = (name or "").strip()
    if not name or name in (".", "..") or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError(f"'{name}' isn't a valid session name (no slashes or '..')")
    return name


def save_session(name, messages):
    name = _safe_session_name(name)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / f"{name}.json"
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(messages, indent=2), encoding="utf-8")
    tmp_path.replace(path)  # atomic on POSIX -- avoids a half-written/corrupt
                             # file if the process gets killed mid-write
                             # (background kill, OOM, battery optimization are
                             # all realistic on a phone)
    return path


def load_session(name):
    name = _safe_session_name(name)
    path = SESSIONS_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"session '{name}' exists but couldn't be read ({e}) -- the file may be corrupted") from e
    if not isinstance(data, list) or not data or not isinstance(data[0], dict) or data[0].get("role") != "system":
        raise RuntimeError(f"session '{name}' doesn't look like a valid conversation (expected system prompt first)")
    return data


def list_sessions():
    if not SESSIONS_DIR.exists():
        return []
    return sorted(p.stem for p in SESSIONS_DIR.glob("*.json"))


# --------------------------------------------------------------------------
# /auto : simplified multi-turn autonomy
# --------------------------------------------------------------------------

def capture_auto_plan(goal):
    """One cheap upfront call: derive an ordered step plan and an explicit
    'definition of done' from the /auto goal, so the harness has a written
    contract to track progress and check completion against instead of
    relying purely on the model's own judgment call many turns later (which
    is how /auto runs drift or declare victory early). Non-fatal on failure
    -- /auto still works without a captured plan, just without this extra
    scaffolding, and the model can still call update_plan itself."""
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text("", encoding="utf-8")
    AUTO_GOAL_PATH.write_text("", encoding="utf-8")

    prompt_messages = [
        {
            "role": "system",
            "content": (
                "Break the following task into a short ordered list of concrete steps, "
                "and a short explicit 'definition of done' -- the specific, checkable "
                "conditions under which the task is actually complete. Respond with ONLY "
                "JSON, no prose, no code fences, in exactly this shape: "
                '{"steps": ["step one", "step two"], "definition_of_done": ["criterion one"]}. '
                "Keep steps to 3-8 items and criteria to 1-5 items."
            ),
        },
        {"role": "user", "content": goal},
    ]
    try:
        choice = call_model(prompt_messages, tools=False, temperature=0.0)
        raw = (choice.get("content") or "").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        steps = [{"text": str(s).strip(), "done": False} for s in (data.get("steps") or []) if str(s).strip()]
        dod = [str(d).strip() for d in (data.get("definition_of_done") or []) if str(d).strip()]
    except Exception:
        steps, dod = [], []

    if steps:
        PLAN_PATH.write_text(_render_plan(steps), encoding="utf-8")
    if dod:
        AUTO_GOAL_PATH.write_text(
            "Definition of done for the current /auto run (check progress against this "
            "explicitly, not just your own sense of completion):\n" + "\n".join(f"- {d}" for d in dod),
            encoding="utf-8",
        )
    return steps, dod


def _print_plan_status():
    if not PLAN_PATH.exists():
        return
    text = PLAN_PATH.read_text(encoding="utf-8").strip()
    if text:
        print(f"\n{text}", file=sys.stderr)


_PLAN_ITEM_RE = re.compile(r"^- \[( |x)\] (.*)$")


def render_plan_for_display():
    """Nicer, colored rendering of the captured /auto plan + definition of
    done for the interactive /plan command -- checkboxes become a green
    check or a dim circle instead of raw '- [x]'/'- [ ]' markdown text."""
    plan_text = PLAN_PATH.read_text(encoding="utf-8").strip() if PLAN_PATH.exists() else ""
    dod_text = AUTO_GOAL_PATH.read_text(encoding="utf-8").strip() if AUTO_GOAL_PATH.exists() else ""
    if not plan_text and not dod_text:
        return None

    lines = []
    if dod_text:
        dod_lines = dod_text.splitlines()
        lines.append(f"{_fg(*ACCENT_COLOR, bold=True)}Definition of done{RESET}")
        for line in dod_lines[1:]:  # first line is the intro sentence, skip re-printing it
            lines.append(f"  {line}" if line.strip().startswith("-") else f"  {DIM}{line}{RESET}")
        if plan_text:
            lines.append("")
    if plan_text:
        lines.append(f"{_fg(*ACCENT_COLOR, bold=True)}Plan{RESET}")
        for line in plan_text.splitlines():
            if line.startswith("#"):
                continue  # skip the raw markdown header, we print our own above
            m = _PLAN_ITEM_RE.match(line)
            if m:
                done = m.group(1) == "x"
                step_text = m.group(2)
                if done:
                    lines.append(f"  {_fg(74, 222, 128)}✓{RESET} {DIM}{step_text}{RESET}")
                else:
                    lines.append(f"  {DIM}○{RESET} {step_text}")
            elif line.strip():
                lines.append(f"  {line}")
    return "\n".join(lines)


def _summarize_tool_usage(messages, start_idx):
    """Tallies tool calls made during messages[start_idx:] by scanning for
    assistant tool_calls entries, rather than threading a counter through
    run_turn -- keeps run_turn's return signature simple."""
    counts = {}
    for m in messages[start_idx:]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for c in m["tool_calls"]:
                name = c.get("function", {}).get("name", "?")
                counts[name] = counts.get(name, 0) + 1
    return counts


def _print_auto_summary(turns_used, elapsed, total_tokens, tool_counts):
    rows = [
        f"Turns       {turns_used}",
        f"Wall time   {elapsed:.0f}s",
        f"Generated   ~{total_tokens} tok",
    ]
    if tool_counts:
        tool_line = ", ".join(f"{name} x{n}" for name, n in sorted(tool_counts.items(), key=lambda kv: -kv[1]))
        rows.append(f"Tools used  {tool_line}")
    else:
        rows.append("Tools used  none")
    print_box(rows)


def run_auto(messages, goal):
    """Repeats normal turns, feeding the model's own progress back to itself,
    until it emits TASK_COMPLETE, calls request_user_input, or MAX_AUTO_TURNS
    is hit. Captures an upfront plan + definition-of-done (see
    capture_auto_plan) so the model has a written contract to track progress
    and completion against, rather than relying purely on its own in-context
    judgment call many turns in. The model can revise the plan itself via
    update_plan as it goes. Ends every exit path with a run summary and,
    if Termux:API is available, a notification -- these runs can take
    minutes on a slow phone CPU and the user will likely have tabbed away."""
    t0 = time.time()
    total_tokens = 0
    msg_start_idx = len(messages)

    steps, dod = capture_auto_plan(goal)
    if steps:
        print(f"\n[/auto plan -- {len(steps)} step(s)]", file=sys.stderr)
        for s in steps:
            print(f"  - {s['text']}", file=sys.stderr)
    if dod:
        print(f"[/auto definition of done -- {len(dod)} criteria]", file=sys.stderr)
        for d in dod:
            print(f"  - {d}", file=sys.stderr)
    if not steps and not dod:
        print("[/auto: couldn't derive an upfront plan -- continuing without one]", file=sys.stderr)

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
        text, reason, stats = run_turn(messages, spinner_label=f"auto turn {turn_i + 1}")
        total_tokens += stats.get("tokens_est", 0) or 0
        if reason != "answered":
            # "answered" replies are already streamed to stdout live; only
            # round_cap/needs_user messages need an explicit print here.
            print(text)

        if reason == "needs_user":
            print(f"\n[/auto stopped -- model needs input: {text}]")
            _print_plan_status()
            _print_auto_summary(turn_i + 1, time.time() - t0, total_tokens,
                                 _summarize_tool_usage(messages, msg_start_idx))
            send_termux_notification("GreyFox needs input", text[:200])
            return

        if "TASK_COMPLETE" in text:
            print(f"\n[/auto finished in {turn_i + 1} turn(s)]")
            _print_plan_status()
            _print_auto_summary(turn_i + 1, time.time() - t0, total_tokens,
                                 _summarize_tool_usage(messages, msg_start_idx))
            send_termux_notification("GreyFox: task complete", goal[:200])
            return

        # nudge it to keep going next turn
        messages.append({
            "role": "user",
            "content": "Continue toward the goal. If it's actually complete, "
                        "say so and end with TASK_COMPLETE.",
        })

    print(f"\n[/auto stopped: hit the {MAX_AUTO_TURNS}-turn safety cap]")
    _print_plan_status()
    _print_auto_summary(MAX_AUTO_TURNS, time.time() - t0, total_tokens,
                         _summarize_tool_usage(messages, msg_start_idx))
    send_termux_notification("GreyFox: /auto stopped", f"Hit the {MAX_AUTO_TURNS}-turn safety cap")


# --------------------------------------------------------------------------
# REPL
# --------------------------------------------------------------------------

VERSION = "0.2.0"


def _detect_color_support():
    """NO_COLOR (https://no-color.org) always wins if set. Otherwise color
    is on only when stdout is an actual terminal -- piping output to a file,
    or running under some Termux widget contexts, would otherwise dump raw
    escape codes into whatever's reading it."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("GREYFOX_FORCE_COLOR"):
        return True
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


COLOR_ENABLED = True   # real values set by _apply_theme() call below
RESET = BOLD = DIM = ""


def _apply_theme(enabled):
    """Flips every color-emitting global on/off in one place. Called once at
    startup, and again at runtime by /theme plain|fancy."""
    global COLOR_ENABLED, RESET, BOLD, DIM
    COLOR_ENABLED = enabled
    RESET = "\033[0m" if enabled else ""
    BOLD = "\033[1m" if enabled else ""
    DIM = "\033[2m" if enabled else ""


def _fg(r, g, b, bold=False):
    """A truecolor foreground escape, or '' if color is disabled. Every
    dynamically-built color (gradient, mascot, callouts, phase spinner) goes
    through this so /theme plain and NO_COLOR silence all of them at once."""
    if not COLOR_ENABLED:
        return ""
    prefix = "\033[1m" if bold else ""
    return f"{prefix}\033[38;2;{r};{g};{b}m"


def term_width(default=80, cap=100):
    """Live terminal width (not cached -- cheap syscall, and phones get
    rotated / terminals get resized between commands)."""
    try:
        cols = shutil.get_terminal_size((default, 24)).columns
    except Exception:
        cols = default
    return max(20, min(cols, cap))


_apply_theme(_detect_color_support())

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
ACCENT_COLOR = GRADIENT[-1]  # ember orange -- used for chat chrome (prompt/reply markers)

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
        out.append(f"{_fg(r, g, b, bold=bold)}{ch}{RESET}")
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


def _termux_api_available():
    """Termux:API tool calls only work if the companion app + CLI shims are
    installed; this just checks whether the shim binaries are on PATH."""
    return shutil.which("termux-notification") is not None


def send_termux_notification(title, message):
    """Harness-level notification (not a model-initiated tool call) -- used
    to alert the user when a long unattended /auto run finishes or needs
    input, since those can take minutes on a slow phone CPU and the user
    will likely have tabbed away by then."""
    if not _termux_api_available():
        return False
    try:
        subprocess.run(
            ["termux-notification", "--title", title, "--content", message],
            timeout=5, capture_output=True,
        )
        return True
    except Exception:
        return False


def copy_to_clipboard(text):
    if not _termux_api_available():
        return False
    try:
        subprocess.run(["termux-clipboard-set"], input=text, text=True, timeout=5)
        return True
    except Exception:
        return False


def _ram_tier_label(ram_mb):
    if ram_mb is None:
        return "unknown"
    if ram_mb >= 7000:
        return "high"
    elif ram_mb >= 5500:
        return "mid"
    return "low"


def print_box(rows, min_width=0):
    """Generic bordered box (status panel, /auto summaries). Wraps any row
    wider than the terminal instead of overflowing off a narrow phone
    screen, and degrades cleanly when color is off (DIM is '' then, so the
    border characters still print, just uncolored)."""
    avail = max(term_width() - 4, 20)  # leave room for "│ " + " │"
    wrapped = []
    for r in rows:
        if not r:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(r, width=avail) or [""])
    width = max([len(r) for r in wrapped] + [min_width, 1])
    print(f"{DIM}┌{'─' * (width + 2)}┐{RESET}")
    for r in wrapped:
        print(f"{DIM}│{RESET} {r.ljust(width)} {DIM}│{RESET}")
    print(f"{DIM}└{'─' * (width + 2)}┘{RESET}")


def print_status_panel():
    """Compact startup status block: RAM tier, detected context size, and
    Termux:API availability -- all of this was already being computed for the
    banner's info line, just not surfaced together as its own block."""
    ram_mb = _detect_ram_mb()
    ram_str = f"{ram_mb} MB" if ram_mb else "unknown"
    tier = _ram_tier_label(ram_mb)
    termux_ok = _termux_api_available()
    invariants_loaded = bool(load_invariants())
    facts_count = len(_load_facts())

    rows = [
        f"RAM         {ram_str}  (tier: {tier})",
        f"Context     ~{CTX_SIZE} tokens  (compacts at {int(COMPACT_TRIGGER_FRACTION * 100)}%)",
        f"Termux:API  {'available' if termux_ok else 'not installed'}",
        f"Invariants  {'loaded' if invariants_loaded else 'empty (see invariants.md)'}",
        f"Facts       {facts_count} recorded",
    ]
    print_box(rows)
    print()


CALLOUT_STYLES = {
    "error": ((248, 113, 113), "✗"),
    "warning": ((250, 204, 21), "!"),
    "info": ((96, 165, 250), "i"),
}


def print_callout(kind, message, file=None):
    """Styled, color-coded line(s) for errors/warnings shown mid-conversation
    -- distinct from print_box (reserved for the heavier startup/summary
    blocks) so inline failures don't interrupt the chat flow with a big box
    every time a tool call or command fails."""
    file = file or sys.stdout
    color, glyph = CALLOUT_STYLES.get(kind, CALLOUT_STYLES["info"])
    label = kind
    width = max(term_width() - 4, 20)
    lines = textwrap.wrap(str(message), width=width) or [""]
    c = _fg(*color, bold=True)
    print(f"{c}{glyph} {label}{RESET}  {lines[0]}", file=file)
    for extra in lines[1:]:
        print(f"  {extra}", file=file)


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


MINIMAL_MODE = False  # set from --minimal argv flag in main()


def print_banner():
    """Boxed header: mascot, gradient wordmark, rule lines, info line --
    same overall structure as the project this is descended from, but with
    an original mascot and an original color scheme (not the other
    project's copied Claude Code mascot / Gemini CLI gradient). Fully
    self-contained: detects RAM tier and the actually-running model on its
    own, nothing needs to be passed in."""
    model_label = _detect_model_label()

    if MINIMAL_MODE:
        bits = [f"v{VERSION}", model_label]
        if CTX_SIZE:
            bits.append(f"ctx {CTX_SIZE}")
        print(f"{_fg(*ACCENT_COLOR, bold=True)}GreyFox{RESET} {DIM}{' · '.join(bits)}{RESET}")
        print(f"{DIM}/help for commands, /theme fancy for the full banner{RESET}\n")
        return

    ctx_size = CTX_SIZE
    mascot_color = _fg(*MASCOT_COLOR, bold=True)
    mascot_lines = [f"{mascot_color}{line}{RESET}" for line in MASCOT]
    wordmark_line = _gradient_text(WORDMARK, GRADIENT, bold=True)
    rule_width = min(len(WORDMARK) + 2, max(term_width() - 6, 10))
    rule = _gradient_rule(rule_width, GRADIENT)

    info_bits = [f"{DIM}{model_label}{RESET}"]
    if ctx_size:
        info_bits.append(f"{DIM}context {RESET}{ctx_size} tokens")
    info_bits.append(f"{DIM}v{VERSION}{RESET}")
    info_line = f"  {DIM}·{RESET}  ".join(info_bits)

    # the rule is deliberately 2 chars wider than the wordmark itself (a
    # small decorative flourish) -- pad the wordmark row to match, or its
    # right-hand border sits crooked relative to every other row in the box.
    wm_pad = max(rule_width - len(WORDMARK), 0)
    wm_left, wm_right = " " * (wm_pad // 2), " " * (wm_pad - wm_pad // 2)

    print(f"{mascot_color}╭{'─' * (rule_width + 4)}╮{RESET}")
    for line in mascot_lines:
        print(f"{mascot_color}│{RESET}  {line}  {mascot_color}│{RESET}")
    print(f"{mascot_color}│{RESET}  {rule}  {mascot_color}│{RESET}")
    print(f"{mascot_color}│{RESET}  {wm_left}{wordmark_line}{wm_right}  {mascot_color}│{RESET}")
    print(f"{mascot_color}│{RESET}  {rule}  {mascot_color}│{RESET}")
    print(f"{mascot_color}╰{'─' * (rule_width + 4)}╯{RESET}")
    print()
    print(info_line)
    print(f"{DIM}› type your message   /help commands   /quit exit{RESET}")
    for line in textwrap.wrap(
        "tools: search, http, files, python, notes, facts, termux -- used automatically when needed",
        width=max(term_width(), 30),
    ):
        print(f"{DIM}{line}{RESET}")
    print()
    print_status_panel()


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
                          safety cap is hit. Captures an upfront plan + definition
                          of done (see /plan) before starting.
  /plan                   show the current /auto plan and definition of done, if any
  /copy                   copy the last reply to the clipboard (needs Termux:API)
  /theme <fancy|plain>    toggle color/banner on or off for this session
  /stats                  show basic stats about the last response
  /quit                   exit
""".format(max_auto=MAX_AUTO_TURNS)

KNOWN_COMMANDS = [
    "/help", "/reset", "/system", "/save", "/load", "/sessions", "/regenerate",
    "/auto", "/plan", "/copy", "/theme", "/stats", "/quit",
]


def new_conversation(system_prompt=DEFAULT_SYSTEM_PROMPT):
    return [{"role": "system", "content": system_prompt}]


def last_user_message(messages):
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return None


def last_assistant_reply(messages):
    """Most recent plain-text assistant reply -- skips assistant messages
    that were only tool_calls (no visible text) so /copy grabs what the
    user actually saw, not an internal tool-dispatch message."""
    for m in reversed(messages):
        if m["role"] == "assistant" and not m.get("tool_calls") and (m.get("content") or "").strip():
            return m["content"]
    return None


def _build_prompt(messages):
    """Colored, context-aware prompt -- shows roughly how full the context
    window is at a glance, without needing to run /stats every turn."""
    try:
        pct = min(estimate_tokens(effective_messages(messages)) / CTX_SIZE * 100, 99) if CTX_SIZE else 0
    except Exception:
        pct = 0
    return f"{_fg(*MASCOT_COLOR)}[{pct:.0f}%]{RESET} {_fg(*ACCENT_COLOR, bold=True)}❯{RESET} "


def _save_readline_history():
    if readline is None:
        return
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(str(HISTORY_FILE))
    except Exception:
        pass


def main():
    global CTX_SIZE, MINIMAL_MODE
    if "--minimal" in sys.argv:
        MINIMAL_MODE = True

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    if readline is not None:
        try:
            readline.set_history_length(500)
            if HISTORY_FILE.exists():
                readline.read_history_file(str(HISTORY_FILE))
        except Exception:
            pass  # corrupted history file, unsupported platform, etc -- not fatal
        # atexit (not a try/finally around the loop) so history is saved no
        # matter how the process ends -- /quit, EOF, Ctrl+C, or an uncaught
        # exception anywhere else -- without having to re-wrap/re-indent the
        # entire REPL loop below.
        atexit.register(_save_readline_history)

    CTX_SIZE = resolve_ctx_size()  # try the server's real n_ctx, else RAM-tier estimate

    messages = new_conversation()
    last_stats = {}

    print_banner()
    print("type /help for commands, or just start chatting.")
    print(f"(talking to llama-server at {LLAMA_SERVER_URL})\n")

    while True:
        try:
            line = input(_build_prompt(messages)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith("/"):
            try:
                parts = shlex.split(line)
            except ValueError as e:
                # e.g. an unmatched quote -- shlex.split raises, not returns
                print(f"couldn't parse that command ({e}) -- check your quotes.")
                continue
            if not parts:
                continue
            cmd = parts[0]

            if cmd == "/help":
                print(HELP_TEXT)

            elif cmd == "/reset":
                messages = new_conversation(messages[0]["content"])
                TOOL_FAIL_COUNTS.clear()
                for stale in (PLAN_PATH, AUTO_GOAL_PATH):
                    if stale.exists():
                        stale.write_text("", encoding="utf-8")
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
                try:
                    path = save_session(name, messages)
                except (ValueError, OSError) as e:
                    print_callout("error", f"couldn't save: {e}")
                else:
                    print(f"saved to {path}")

            elif cmd == "/load":
                if len(parts) < 2:
                    print("usage: /load <name>")
                else:
                    try:
                        loaded = load_session(parts[1])
                    except (ValueError, RuntimeError) as e:
                        print_callout("error", str(e))
                    else:
                        if loaded is None:
                            print(f"no session named '{parts[1]}'")
                        else:
                            messages = loaded
                            print(f"loaded '{parts[1]}' ({len(messages)} messages).")

            elif cmd == "/sessions":
                names = list_sessions()
                if not names:
                    print("(none saved yet)")
                else:
                    rows = []
                    for name in names:
                        path = SESSIONS_DIR / f"{name}.json"
                        try:
                            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
                        except OSError:
                            mtime = "?"
                        try:
                            data = json.loads(path.read_text(encoding="utf-8"))
                            count = f"{len(data)} msgs" if isinstance(data, list) else "?"
                        except Exception:
                            count = "corrupted"
                        rows.append((name, count, mtime))
                    name_w = max(len(r[0]) for r in rows)
                    count_w = max(len(r[1]) for r in rows)
                    for name, count, mtime in rows:
                        print(f"  {_fg(*ACCENT_COLOR)}{name.ljust(name_w)}{RESET}  "
                              f"{DIM}{count.rjust(count_w)} · {mtime}{RESET}")

            elif cmd == "/regenerate":
                prev = last_user_message(messages)
                if prev is None:
                    print("nothing to regenerate yet.")
                else:
                    # drop everything after the last user message and retry
                    idx = max(i for i, m in enumerate(messages) if m["role"] == "user")
                    messages = messages[: idx + 1]
                    rollback_len = len(messages)
                    t0 = time.time()
                    try:
                        text, reason, stats = run_turn(messages)
                    except Exception as e:
                        print_callout("error", str(e))
                        del messages[rollback_len:]  # discard any partial mutation from the failed turn
                    else:
                        last_stats = {"elapsed": round(time.time() - t0, 2), "reason": reason, **stats}
                        if reason != "answered":
                            print(text)

            elif cmd == "/auto":
                goal = line[len("/auto "):].strip()
                if not goal:
                    print("usage: /auto <goal>")
                else:
                    try:
                        run_auto(messages, goal)
                    except Exception as e:
                        print_callout("error", f"/auto stopped early: {e}")
                        print(f"{DIM}(conversation so far is preserved -- /save to keep it, or /regenerate to retry){RESET}")

            elif cmd == "/stats":
                print(json.dumps(last_stats, indent=2) if last_stats else "(no stats yet)")

            elif cmd == "/plan":
                rendered = render_plan_for_display()
                if rendered is None:
                    print("no plan captured yet -- /auto captures one automatically when it starts.")
                else:
                    print(rendered)

            elif cmd == "/copy":
                reply = last_assistant_reply(messages)
                if reply is None:
                    print("nothing to copy yet.")
                elif not _termux_api_available():
                    print_callout("warning", "Termux:API isn't installed -- can't reach the clipboard.")
                elif copy_to_clipboard(reply):
                    print(f"copied ({len(reply)} chars) to clipboard.")
                else:
                    print_callout("error", "clipboard copy failed.")

            elif cmd == "/theme":
                choice = parts[1].lower() if len(parts) > 1 else ""
                if choice not in ("fancy", "plain"):
                    print("usage: /theme fancy|plain")
                else:
                    _apply_theme(choice == "fancy")
                    print(f"theme set to {choice}.")

            elif cmd == "/quit":
                break

            else:
                suggestion = difflib.get_close_matches(cmd, KNOWN_COMMANDS, n=1, cutoff=0.6)
                if suggestion:
                    print(f"unknown command '{cmd}' -- did you mean {suggestion[0]}?")
                else:
                    print(f"unknown command '{cmd}', try /help")
            continue

        # plain chat turn
        rollback_len = len(messages)
        messages.append({"role": "user", "content": line})
        t0 = time.time()
        try:
            text, reason, stats = run_turn(messages)
        except Exception as e:
            print_callout("error", str(e))
            del messages[rollback_len:]  # discard the user message + any partial mutation from this turn
            continue
        last_stats = {"elapsed": round(time.time() - t0, 2), "reason": reason, **stats}
        if reason != "answered":
            # "answered" replies are streamed live inside run_turn; only
            # round_cap/needs_user text needs printing here.
            print(text)
        if reason == "needs_user":
            print(f"{DIM}(model is waiting on you -- see request above){RESET}")


if __name__ == "__main__":
    main()
