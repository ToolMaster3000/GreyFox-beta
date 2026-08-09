# GreyFox-CLI

A local, on-device chat CLI for Android/Termux that talks to a `llama.cpp`
server and gives the model a tool-calling loop: web search, arbitrary HTTP
requests, sandboxed file access, Python execution, keyword notes search, and
Termux:API integration.

**Model is up to you.** This script doesn't download or assume any specific
model — point `MODEL_URL` at whatever tool-calling-capable chat GGUF you
want to run (sized to your device's RAM).

## Requirements

| | |
|---|---|
| Device | Android phone, 4GB+ RAM recommended |
| Runtime | [Termux](https://termux.dev) — the F-Droid build |
| Storage | Enough for llama.cpp build artifacts + your chosen model |
| Optional | [Termux:API](https://wiki.termux.com/wiki/Termux:API) app, only for the clipboard/notification tool |

**Compatibility note:** `greyfox_cli.py` streams every reply and uses tool
calling on every turn, together, on the same request. Some older llama.cpp
builds reject that combination outright (`500: "Cannot use tools with
stream"`). `setup.sh`'s pinned build (`LLAMACPP_PIN`) is recent enough to
support it, and if you point GreyFox at an older/different `llama-server`
that doesn't, the CLI detects the specific error, warns once, and falls back
to non-streaming for the rest of the session rather than breaking. Run
`./setup.sh --selftest` after setup to check this (and a couple of other
things) explicitly.

## Quickstart

```bash
pkg install git
pkg install libandroid-spawn
git clone https://github.com/ToolMaster3000/GreyFox-beta ~/greyfox-src
cd ~/greyfox-src
MODEL_URL="https://huggingface.co/<repo>/resolve/main/<file>.gguf" bash setup.sh
```

If you got `setup.sh` and `greyfox_cli.py` as separate downloads rather than
via `git clone` (e.g. from a chat/forum), make sure both files -- plus
`README.md` if you have it -- are sitting in the **same folder** before
running `bash setup.sh`. `setup.sh` copies `greyfox_cli.py` from its own
directory; if that file's missing, it now fails with a clear message telling
you where it looked, instead of a bare `cp: cannot stat` error.

First run will: install packages, build `llama.cpp` (pinned commit), download
your chosen model, set up a Python venv, print device-specific battery
whitelist instructions, install a Termux autostart hook, and start the
server + CLI. Re-running `bash setup.sh` is idempotent.

## Setup script flags

| Flag | What it does |
|---|---|
| `bash setup.sh` | Full install + launch |
| `--setup-only` | Install/build only, don't launch |
| `--run-only` | Skip checks, just start server + CLI (used by autostart) |
| `--force-rebuild` | Rebuild llama.cpp even if already built |
| `--selftest` | Start the server and run one health-check completion |
| `--stop` | Stop the background `llama-server` |
| `--no-autostart` | Install without the Termux autostart hook |
| `--disable-autostart` | Remove a previously installed autostart hook |
| `--version` | Print version and exit |

## In-CLI commands

| Command | What it does |
|---|---|
| `/help` | List commands |
| `/reset` | Clear conversation, keep system prompt |
| `/system <prompt>` | Replace the system prompt |
| `/save [name]` / `/load <name>` | Persist/restore a conversation |
| `/sessions` | List saved conversations |
| `/regenerate` | Re-run the last prompt |
| `/auto <goal>` | Run turns autonomously (up to 8, `MAX_AUTO_TURNS`) until the model says `TASK_COMPLETE`, calls `request_user_input`, or the cap is hit |
| `/plan` | Show the current `/auto` plan + definition of done, if any |
| `/stats` | Show basic stats about the last response |
| `/quit` | Exit (server keeps running in the background — `bash setup.sh --stop` to free RAM) |

## Tools available to the model

| Tool | What it does |
|---|---|
| `web_search` | DuckDuckGo HTML scrape, no API key |
| `http_request` | Arbitrary GET/POST/PUT/PATCH/DELETE to a model-constructed URL |
| `read_file` / `write_file` / `list_directory` | Sandboxed to `~/greyfox-cli/workspace/` |
| `run_python` | Sandboxed subprocess, 15s timeout, output capped |
| `search_notes` | Keyword-ranked search over workspace text files |
| `record_fact` / `query_facts` | Structured fact ledger, separate from conversation history |
| `update_plan` | Write/update the ordered plan for the current `/auto` run (see below) |
| `termux_api` | Clipboard read/write and notifications via Termux:API |
| `request_user_input` | Explicit "I genuinely cannot proceed without you" signal |

Tool-call structure is constrained via llama.cpp server's native OpenAI-style
`tools` support (JSON-schema-constrained decoding), so malformed tool-call
arguments are rejected at the decoding layer rather than just discouraged by
the prompt — no hand-rolled grammar needed.

When more than one tool call comes back in the same round (e.g. two
independent `web_search` calls), they're dispatched concurrently on a small
thread pool instead of one at a time — `web_search`/`http_request` are
I/O-bound, so this cuts real wall-clock time on multi-call rounds. Each
call's progress (`… name` then `✓`/`✗ name`) prints as it starts/finishes
rather than as one blob at the end.

If the same tool fails twice in a row with a similar-looking error, GreyFox
stops just blindly retrying it: it drops a note into the conversation nudging
the model to try different arguments, a different tool, or
`request_user_input` instead of grinding on the same broken call.

## Context management

- **`invariants.md`** (in `~/greyfox-cli/`, created on first run): a small
  always-reinjected file for load-bearing facts/constraints — things the
  model must never lose or contradict. It's read fresh on every call and
  lives outside the compactable message history, so no amount of compaction
  can drop it. Edit it directly, or ask the model to.
- **Context compaction**: once the running transcript crosses roughly 70% of
  the detected context window, older tool-result-heavy turns are compressed
  into a short bullet-point summary via one cheap extra model call. The
  system prompt and the most recent turns are always kept verbatim.
  `invariants.md` is never part of what gets summarized.
- **Fact ledger** (`record_fact` / `query_facts`): a small structured
  key/value log the model can write to and search, independent of both the
  conversation history and `invariants.md` — useful once a task spans
  several subtasks or a long `/auto` run.

## How `/auto` works here

`/auto <goal>` first makes one cheap upfront call (no tools) to break the
goal into an ordered step plan and an explicit **definition of done** --
the specific, checkable conditions under which the task is actually
complete. Both are written to disk (`plan.md`, `auto_goal.md`) and
reinjected fresh on every turn, same as `invariants.md`, so they survive
context compaction and give the model (and you) a written contract to check
progress against instead of relying purely on its own in-context judgment
call many turns in. Check the captured plan any time with `/plan`.

The model can revise the plan itself mid-run via the `update_plan` tool --
call it with the full step list (not a diff) whenever a step finishes or
the plan needs to change.

`/auto` then repeats the normal single-turn tool-calling loop across up to
`MAX_AUTO_TURNS` (default 8) turns, nudging the model to continue each time,
until:

- the model's reply contains the literal line `TASK_COMPLETE`, or
- the model calls `request_user_input` (genuinely stuck / needs you), or
- the turn cap is hit.

In all three cases the final plan status prints so you can see what
actually got done vs. what didn't.

If the upfront planning call fails to parse (small models sometimes ignore
"respond with only JSON"), `/auto` just continues without a captured plan --
it's scaffolding, not a hard requirement.


## Directory layout

```
~/greyfox-cli/
├── setup.sh              # copied here on install; used by --run-only/autostart
├── greyfox_cli.py
├── llama.cpp/          # cloned + built, pinned to LLAMACPP_PIN
├── models/              # your downloaded GGUF
├── venv/
├── logs/
│   └── llama-server.log
├── sessions/             # /save'd conversations (JSON)
├── invariants.md         # always-reinjected facts/constraints (see below)
├── facts.json            # record_fact/query_facts ledger
├── plan.md               # current /auto run's step plan (see below)
├── auto_goal.md           # current /auto run's definition of done
├── server.pid
└── workspace/            # sandboxed - all file tools + run_python operate here
```

## Terminal UI

- **Streaming output**: replies print token-by-token as they're generated
  instead of blocking until the full reply is done — the difference between
  a long silent wait and visible progress on a slow phone CPU.
- **Per-phase spinner**: while a round is waiting on the model or running a
  single tool, a colored spinner shows what's actually happening
  (`searching`, `reading file`, `running python`, ...) instead of a flat
  `...thinking` line.
- **Live tok/s + context readout**: after every answered turn, a line shows
  roughly how many tokens were generated, tokens/sec, and how full the
  context window is — so you can judge whether a long `/auto` run is worth
  letting continue. Also available via `/stats`.
- **Startup status panel**: a compact block on launch showing detected RAM
  tier, context size, and whether Termux:API is available.
- **Batched tool-call progress**: when a round fires off several tool calls
  at once, each shows a live `…` while in flight and a `✓`/`✗` as it
  finishes, instead of one combined result at the end.

## Security & privacy

- `read_file`, `write_file`, `list_directory`, and `run_python` are confined
  to `~/greyfox-cli/workspace/` — path traversal is checked and rejected.
  `run_python` runs as a fresh subprocess with a timeout, not in-process.
- **Network egress is not restricted**: `web_search` and `http_request` can
  reach any public URL the model constructs. That's the point of the tools,
  but it means open internet access, not an allowlisted one — a real
  constraint to weigh before pointing this at anything sensitive.
- Whatever model you choose determines its own safety behavior; this script
  doesn't modify or bypass a model's built-in judgment. Choose your model
  deliberately, especially given open tool access.
- `run_python` and `http_request` output should be treated as untrusted
  before you act on it.
- Termux:API access only works if you've explicitly installed that app.

## Known limitations

- `search_notes` is keyword-ranked, not semantic.
- Context-window estimation is a rough chars/4 heuristic (no local
  tokenizer), so compaction can fire a bit early or late relative to the
  server's actual token count.
- `/auto` has no checkpointing — if Termux is killed mid-task, that `/auto`
  run is gone; start over.
- Small/quantized models will sometimes skip a tool they should have used,
  or misjudge when one's needed. This is a much smaller system than a
  full-scale agent harness; it won't out-plan a large cloud model.

## Uninstalling

```bash
bash setup.sh --stop
bash setup.sh --disable-autostart
rm -rf ~/greyfox-cli
```
