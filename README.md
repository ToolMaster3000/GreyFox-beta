# GreyFox-CLI

A local, on-device chat CLI for Android/Termux that talks to a `llama.cpp` server and gives the model a tool-calling loop: web search, arbitrary HTTP
requests, sandboxed file access, Python execution, keyword notes search, and
Termux:API integration.

**Default model: Nanbeige4.2-3B-heretic (Q5_K_M).** `setup.sh` downloads and
runs this by default if you don't override `MODEL_URL`/`MODEL_FILE`. See
[Model](#model) below before you run it -- this is a deliberately
uncensored variant, and it's now what drives the tool loop unless you change it.

## Requirements

|          |                                                                                                     |
| -------- | --------------------------------------------------------------------------------------------------- |
| Device   | Android phone, 4GB+ RAM recommended                                                                 |
| Runtime  | [Termux](https://termux.dev) — the F-Droid build                                                    |
| Storage  | Enough for llama.cpp build artifacts + your chosen model (default model is ~3GB)                    |
| Optional | [Termux:API](https://wiki.termux.com/wiki/Termux:API) app, only for the clipboard/notification tool |

## Quickstart

```
pkg install git
pkg install libandroid-spawn
git clone https://github.com/ToolMaster3000/GreyFox-beta ~/greyfox-src
cd ~/greyfox-src
bash setup.sh
```

First run will: install packages, build `llama.cpp` from Nanbeige's fork
(pinned commit), download the default model (Nanbeige4.2-3B-heretic,
Q5_K_M, ~3GB), set up a Python venv, print device-specific battery
whitelist instructions, install a Termux autostart hook, and start the
server + CLI. Re-running `bash setup.sh` is idempotent.

To run a different model instead:

```
MODEL_URL="https://huggingface.co/<repo>/resolve/main/<file>.gguf" \
MODEL_FILE="<file>.gguf" \
bash setup.sh
```

Note: a non-Nanbeige model will still build against Nanbeige's llama.cpp
fork (see [Model](#model)) unless you also edit `LLAMACPP_REPO`/`LLAMACPP_PIN`
in `setup.sh` back to mainline `ggml-org/llama.cpp` -- the fork is a superset
branch, so ordinary GGUFs should still load fine on it, but it's not the
default upstream build.

## Model

[#model](#model)

Default: [`WaveCut/Nanbeige4.2-3B-heretic-GGUF`](https://huggingface.co/WaveCut/Nanbeige4.2-3B-heretic-GGUF),
`Nanbeige4.2-3B-heretic-Q5_K_M.gguf` quant (~3GB, 4B params).

Two things worth knowing before relying on the default:

- **Build requirement:** Nanbeige4.2 is a looped Transformer (22 physical
  layers executed twice). That architecture isn't in mainline
  `ggml-org/llama.cpp` yet -- support is a still-open, unmerged PR
  ([#25994](https://github.com/ggml-org/llama.cpp/pull/25994)). Until it
  merges, `setup.sh` builds `llama.cpp` from Nanbeige's own fork/branch
  instead of mainline. This is handled automatically; you don't need to do
  anything, but `--force-rebuild` and troubleshooting steps that reference
  "llama.cpp" mean this fork, not upstream.
- **This is the "heretic" (abliterated) variant.** Its
  [model card](https://huggingface.co/WaveCut/Nanbeige4.2-3B-heretic) reports
  refusals dropping from 17/100 to 1/100 on held-out harmful-behavior
  prompts, and explicitly recommends access controls, monitoring, and
  downstream safety measures for production use. Combined with this
  project's unrestricted `http_request`/`run_python` tools and unattended
  `/auto` mode (see [Security & privacy](#security--privacy)), that's a
  meaningfully more permissive default than picking a model per run. Swap
  `MODEL_URL`/`MODEL_FILE` for something else if that's not what you want.

## Setup script flags

| Flag                  | What it does                                             |
| --------------------- | ---------------------------------------------------------- |
| `bash setup.sh`       | Full install + launch                                    |
| `--setup-only`        | Install/build only, don't launch                         |
| `--run-only`          | Skip checks, just start server + CLI (used by autostart) |
| `--force-rebuild`     | Rebuild llama.cpp even if already built                  |
| `--selftest`          | Start the server and run one health-check completion     |
| `--stop`              | Stop the background `llama-server`                       |
| `--no-autostart`      | Install without the Termux autostart hook                |
| `--disable-autostart` | Remove a previously installed autostart hook              |
| `--version`           | Print version and exit                                   |

## In-CLI commands

| Command                         | What it does                                                                                                                           |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `/help`                         | List commands                                                                                                                          |
| `/reset`                        | Clear conversation, keep system prompt                                                                                                 |
| `/system <prompt>`              | Replace the system prompt                                                                                                              |
| `/save [name]` / `/load <name>` | Persist/restore a conversation                                                                                                         |
| `/sessions`                     | List saved conversations                                                                                                               |
| `/regenerate`                   | Re-run the last prompt                                                                                                                 |
| `/auto <goal>`                  | Run turns autonomously (up to 8, `MAX_AUTO_TURNS`) until the model says `TASK_COMPLETE`, calls `request_user_input`, or the cap is hit |
| `/stats`                        | Show basic stats about the last response                                                                                               |
| `/quit`                         | Exit (server keeps running in the background — `bash setup.sh --stop` to free RAM)                                                    |

## Tools available to the model

| Tool                                          | What it does                                                   |
| --------------------------------------------- | -------------------------------------------------------------- |
| `web_search`                                  | DuckDuckGo HTML scrape, no API key                             |
| `http_request`                                | Arbitrary GET/POST/PUT/PATCH/DELETE to a model-constructed URL |
| `read_file` / `write_file` / `list_directory` | Sandboxed to `~/greyfox-cli/workspace/`                        |
| `run_python`                                  | Sandboxed subprocess, 15s timeout, output capped               |
| `search_notes`                                | Keyword-ranked search over workspace text files                |
| `termux_api`                                  | Clipboard read/write and notifications via Termux:API           |
| `request_user_input`                          | Explicit "I genuinely cannot proceed without you" signal        |

Tool-call structure is constrained via llama.cpp server's native OpenAI-style `tools` support (JSON-schema-constrained decoding), so malformed tool-call
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
├── llama.cpp/            # cloned from Nanbeige's fork, built, pinned to LLAMACPP_PIN
├── models/                # your downloaded GGUF
├── venv/
├── logs/
│   └── llama-server.log
├── sessions/              # /save'd conversations (JSON)
├── server.pid
└── workspace/              # sandboxed - all file tools + run_python operate here
```

## Security & privacy

[#security--privacy](#security--privacy)

- `read_file`, `write_file`, `list_directory`, and `run_python` are confined
  to `~/greyfox-cli/workspace/` — path traversal is checked and rejected.
  `run_python` runs as a fresh subprocess with a timeout, not in-process.
- **Network egress is not restricted**: `web_search` and `http_request` can
  reach any public URL the model constructs. That's the point of the tools,
  but it means open internet access, not an allowlisted one — a real
  constraint to weigh before pointing this at anything sensitive.
- Whatever model you choose determines its own safety behavior; this script
  doesn't modify or bypass a model's built-in judgment. **The default model
  (Nanbeige4.2-3B-heretic) has intentionally reduced refusal behavior** --
  see [Model](#model). Choose your model deliberately, especially given open
  tool access.
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

```
bash setup.sh --stop
bash setup.sh --disable-autostart
rm -rf ~/greyfox-cli
```
