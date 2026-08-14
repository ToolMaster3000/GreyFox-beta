# GreyFox-CLI

A full agent harness for on-device LLMs, not just a chat wrapper around
`llama.cpp` — built for the current generation of small, reasoning-tuned
models that have made real agentic work viable on a phone. Models like
Nanbeige's 3B reasoning/agentic line are trained specifically for sustained
multi-step reasoning and tool use rather than raw parameter count, and their
authors report them matching or beating much larger general-purpose models
on exactly the things an agent needs: planning, tool selection, and staying
on-task across long tool-calling chains. Point GreyFox at one of those (or
any tool-calling GGUF sized to your device), and it gives the model the
scaffolding to actually finish complex, autonomous, multi-step tasks instead
of losing the thread a few tool calls in — an explicit plan and
definition-of-done for every `/auto` run, context compaction that never
drops load-bearing facts or splits a tool call from its result, failure
escalation instead of blind retries, concurrent tool dispatch, and a
structured fact ledger that survives compaction. All of it running locally
in Termux, on hardware you already own.

The full tool surface: web search, arbitrary HTTP requests, sandboxed file
access, Python execution, keyword notes search, and Termux:API integration
(clipboard, notifications) — enough for most real multi-step tasks without
leaving the device.

**Model is up to you.** This script doesn't download or assume any specific
model — point `MODEL_URL` at whatever tool-calling-capable chat GGUF you
want to run. A 3B-class reasoning model in Q4 quantization is a comfortable
fit for the 4GB+ tier below; scale up with more RAM.

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
`bash setup.sh --selftest` after setup to check this (and a couple of other
things) explicitly.

## Quickstart

```bash
pkg install -y git libandroid-spawn
git clone https://github.com/ToolMaster3000/GreyFox-beta ~/greyfox-src
cd ~/greyfox-src
MODEL_URL="https://huggingface.co/<repo>/resolve/main/<file>.gguf" bash setup.sh
```

`libandroid-spawn` needs to be installed up front — `run_python` and parts
of the `llama.cpp` build spawn subprocesses in a way that needs it, and it
isn't reliably pulled in as a side effect of anything else. `setup.sh`
handles the rest of the packages itself (cmake, clang, Python, build tools).

If you got `setup.sh` and `greyfox_cli.py` as separate downloads rather than
via `git clone` (e.g. from a chat or forum), make sure both files are
sitting in the **same folder** before running `bash setup.sh` — it looks for
`greyfox_cli.py` right next to itself and fails with a clear message
(showing exactly what it found instead) if it can't.

First run will: install packages, build `llama.cpp` (pinned commit),
download your chosen model, save your config, set up a Python venv, install
the CLI, print device-specific battery-whitelist instructions, install a
Termux autostart hook, and start the server + CLI. Re-running `bash setup.sh`
is idempotent — it skips whatever's already done (use `--force-rebuild` to
redo the llama.cpp build specifically).

**Always run `bash setup.sh` from this original folder when you want to
update** (a new `greyfox_cli.py`, a new `LLAMACPP_PIN`, etc.) — not from
`~/greyfox-cli`. Setup also copies itself into `~/greyfox-cli` so the
autostart hook has something to invoke, but that copy is a fixed snapshot;
it won't pick up newer files placed elsewhere. If you do run it from there
by mistake, it detects that and tells you where to go instead of silently
using stale files.

## Setup script flags

| Flag | What it does |
|---|---|
| `bash setup.sh` | Full install + launch |
| `--setup-only` | Install/build only, don't launch |
| `--run-only` | Skip checks, just start server + CLI (used by autostart) |
| `--force-rebuild` | Rebuild llama.cpp even if already built |
| `--selftest` | Start the server and check completions, streaming, tool calling, and context detection |
| `--stop` | Stop the background `llama-server` |
| `--no-autostart` | Install without the Termux autostart hook |
| `--disable-autostart` | Remove a previously installed autostart hook |
| `--version` | Print version and exit |
| `--minimal` | Skip the full banner; launch with a one-line startup message instead (passed through to `greyfox_cli.py`) |

`MODEL_URL`, `MODEL_FILE`, and `SERVER_PORT` are read from the environment
the first time you set them and then remembered automatically (saved to
`~/greyfox-cli/config.env`) — you only need to pass them again if you want
to change one.

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
| `/copy` | Copy the last reply to the clipboard (needs Termux:API) |
| `/theme fancy\|plain` | Toggle color/banner on or off for this session |
| `/stats` | Show basic stats about the last response |
| `/quit` | Exit (server keeps running in the background — `bash setup.sh --stop` to free RAM) |

Arrow-key history and line editing work in the prompt, persisted across
sessions. An unrecognized command gets a typo suggestion (`/atuo` → "did you
mean /auto?") instead of just failing.

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
  system prompt and the most recent turns are always kept verbatim, and a
  summary boundary is never allowed to split a tool call from its result.
  `invariants.md` is never part of what gets summarized.
- **Fact ledger** (`record_fact` / `query_facts`): a small structured
  key/value log the model can write to and search, independent of both the
  conversation history and `invariants.md` — useful once a task spans
  several subtasks or a long `/auto` run.

## How `/auto` works here

`/auto <goal>` first makes one cheap upfront call (no tools) to break the
goal into an ordered step plan and an explicit **definition of done** —
the specific, checkable conditions under which the task is actually
complete. Both are written to disk (`plan.md`, `auto_goal.md`) and
reinjected fresh on every turn, same as `invariants.md`, so they survive
context compaction and give the model (and you) a written contract to check
progress against instead of relying purely on its own in-context judgment
call many turns in. Check the captured plan any time with `/plan`.

The model can revise the plan itself mid-run via the `update_plan` tool —
call it with the full step list (not a diff) whenever a step finishes or
the plan needs to change.

`/auto` then repeats the normal single-turn tool-calling loop across up to
`MAX_AUTO_TURNS` (default 8) turns, nudging the model to continue each time,
until:

- the model's reply contains the literal line `TASK_COMPLETE`, or
- the model calls `request_user_input` (genuinely stuck / needs you), or
- the turn cap is hit.

Every exit path ends with a summary (turns used, wall time, tokens
generated, tools called) and, if Termux:API is installed, a notification —
useful since a run can take minutes and you'll likely have tabbed away.

If the upfront planning call fails to parse (small models sometimes ignore
"respond with only JSON"), `/auto` just continues without a captured plan —
it's scaffolding, not a hard requirement.

## Directory layout

```
~/greyfox-cli/
├── setup.sh              # copied here on install; used by --run-only/autostart
├── greyfox_cli.py
├── config.env             # remembered MODEL_URL/MODEL_FILE/SERVER_PORT
├── llama.cpp/            # cloned + built, pinned to LLAMACPP_PIN
├── models/                # your downloaded GGUF
├── venv/
├── logs/
│   └── llama-server.log
├── sessions/              # /save'd conversations (JSON)
├── history                # readline input history
├── invariants.md          # always-reinjected facts/constraints (see below)
├── facts.json             # record_fact/query_facts ledger
├── plan.md                # current /auto run's step plan (see below)
├── auto_goal.md           # current /auto run's definition of done
├── server.pid
└── workspace/             # sandboxed - all file tools + run_python operate here
```

## Terminal UI

- **Streaming output**: replies print token-by-token as they're generated
  instead of blocking until the full reply is done — the difference between
  a long silent wait and visible progress on a slow phone CPU.
- **Lightweight markdown rendering, live**: `**bold**`, `` `inline code` ``,
  `# headers`, and ```` ```fenced code blocks``` ```` get styled in color as
  they stream, instead of printing as literal asterisks/backticks. Handles
  chunk boundaries correctly (a token split can land mid-delimiter). Only
  active when color is on — with color off, the raw markdown source prints
  unchanged, so a piped/logged transcript keeps full fidelity.
- **Chat chrome**: a colored marker appears once, right before the model's
  actual reply text (never before an internal tool-calling round, which has
  no visible text of its own), so it's visually clear where the assistant's
  turn starts. The input prompt itself is colored too, and shows roughly how
  full the context window is (`[12%] ❯`) at a glance.
- **Per-phase spinner**: while a round is waiting on the model or running a
  single tool, a colored spinner shows what's actually happening
  (`searching`, `reading file`, `running python`, ...) instead of a flat
  `...thinking` line.
- **Live tok/s + context readout**: after every answered turn, a line shows
  roughly how many tokens were generated, tokens/sec, and how full the
  context window is — so you can judge whether a long `/auto` run is worth
  letting continue. Also available via `/stats`.
- **Startup status panel**: a compact box on launch showing detected RAM
  tier, context size, and whether Termux:API is available. Box width adapts
  to the terminal, wrapping instead of overflowing on a narrow phone screen.
- **Batched tool-call progress**: when a round fires off several tool calls
  at once, each shows a live `…` while in flight and a `✓`/`✗` as it
  finishes, instead of one combined result at the end.
- **Colored, styled errors**: failures (a bad `/save`, a corrupted session,
  the streaming-fallback warning) print as a color-coded callout instead of
  a bare `[error] ...` line.
- **`/theme fancy|plain`**: toggle the banner/colors off for a plain,
  low-noise REPL, or back on. `NO_COLOR` (see [no-color.org](https://no-color.org))
  and non-interactive output (piping to a file) are also detected
  automatically. `--minimal` at launch skips the full banner in favor of a
  one-line startup message.

## Uninstalling

Run this from either copy of `setup.sh` (your source folder or
`~/greyfox-cli`) — both work identically for stopping and removing the
autostart hook, since they both act on the same `~/greyfox-cli`:

```bash
bash setup.sh --stop
bash setup.sh --disable-autostart
rm -rf ~/greyfox-cli
```

`--disable-autostart` removes `~/.shortcuts/greyfox-autostart.sh`, which
lives outside `~/greyfox-cli` — run it *before* deleting the directory, or
the boot hook is left pointing at a `setup.sh` that no longer exists. Your
original source folder (e.g. `~/greyfox-src`) isn't touched by any of this;
remove it separately if you want it gone too.
