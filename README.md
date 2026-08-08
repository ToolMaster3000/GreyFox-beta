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

## Quickstart

```bash
pkg install git
pkg install libandroid-spawn
git clone https://github.com/ToolMaster3000/GreyFox-beta ~/greyfox-src
cd ~/greyfox-src
MODEL_URL="https://huggingface.co/<repo>/resolve/main/<file>.gguf" bash setup.sh
```

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
| `termux_api` | Clipboard read/write and notifications via Termux:API |
| `request_user_input` | Explicit "I genuinely cannot proceed without you" signal |

Tool-call structure is constrained via llama.cpp server's native OpenAI-style
`tools` support (JSON-schema-constrained decoding), so malformed tool-call
arguments are rejected at the decoding layer rather than just discouraged by
the prompt — no hand-rolled grammar needed.

## How `/auto` works here

`/auto <goal>` posts the goal, then repeats the normal single-turn
tool-calling loop across up to `MAX_AUTO_TURNS` (default 8) turns, nudging
the model to continue each time, until:

- the model's reply contains the literal line `TASK_COMPLETE`, or
- the model calls `request_user_input` (genuinely stuck / needs you), or
- the turn cap is hit.


## Directory layout

```
~/greyfox-cli/
├── setup.sh
├── greyfox_cli.py
├── llama.cpp/          # cloned + built, pinned to LLAMACPP_PIN
├── models/              # your downloaded GGUF
├── venv/
├── logs/
│   └── llama-server.log
├── sessions/             # /save'd conversations (JSON)
├── server.pid
└── workspace/            # sandboxed - all file tools + run_python operate here
```

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
- No context compaction: once you approach the server's context window,
  older turns will start getting dropped/rejected by the server rather than
  intelligently summarized. `/reset` or trim with `/regenerate` if you hit
  this.
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
