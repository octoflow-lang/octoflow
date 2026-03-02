# `octoflow chat` — AI Code Generation

Generate OctoFlow code from natural language. Describe what you want,
and OctoFlow writes, validates, and runs the code for you.

## Quick Start

```bash
octoflow chat
```

```
OctoFlow v1.4.0 — Chat Mode (type :help for commands)
Model: Qwen3-1.7B (Q5_K_M)

> Write a fibonacci function that prints the first 20 numbers

fn fibonacci(n)
  let mut a = 0
  let mut b = 1
  for i in range(0, n)
    print("{a}")
    let tmp = b
    b = a + b
    a = tmp
  end
end

fibonacci(20)

[Running...]
0
1
1
2
3
5
8
13
21
34
...
```

## How It Works

1. You describe what you want in plain English
2. The context engine loads relevant syntax and module signatures (L0-L3 knowledge tree)
3. The LLM generates OctoFlow code (optionally grammar-constrained via GBNF)
4. `octoflow check` validates the code for syntax errors
5. `octoflow run` executes with sandbox protection
6. If errors occur, the structured error JSON is fed back to the LLM (up to 3 repair attempts)

## Modes

### Local Model (Default)

Uses a local GGUF model. No network required. GPU-accelerated via Vulkan.

```bash
octoflow chat                          # uses default model
octoflow chat --model ./my-model.gguf  # specific model
```

OctoFlow looks for GGUF models in:
1. `./models/` (current directory)
2. `~/.octoflow/models/`

The default model is Qwen3-1.7B Q5_K_M (1.26 GB).

### API Mode

Connect to any OpenAI-compatible API endpoint.

```bash
octoflow chat --api http://localhost:8080           # local server
octoflow chat --api https://api.openai.com/v1       # OpenAI
```

API key precedence: `--api-key` flag > `OCTOFLOW_API_KEY` env > `OPENAI_API_KEY` env.

```bash
export OCTOFLOW_API_KEY=sk-...
octoflow chat --provider api
```

### Thinking Mode

For complex problems, enable extended reasoning with `--think`. The model
produces internal reasoning tokens before generating code.

```bash
octoflow chat --think
```

Best for multi-step problems (e.g., "build a CSV pipeline that filters,
groups, and exports to JSON"). Slower but more accurate.

### Web Tools

Enable web search and page reading during code generation. The LLM can
issue `SEARCH:` and `READ:` commands to gather context before writing code.

```bash
octoflow chat --allow-net --web-tools
```

```
> Build a program that fetches the current Bitcoin price

[SEARCH: bitcoin price API endpoint]
[READ: https://api.coindesk.com/v1/bpi/currentprice.json]
[Generating code with enriched context...]

let page = web_read("https://api.coindesk.com/v1/bpi/currentprice.json")
let data = json_decode(page.text)
let price = map_get(data, "bpi.USD.rate")
print("Bitcoin: {price}")
```

Up to 3 tool calls per turn via a ReAct loop.

### Grammar-Constrained Decoding

Local models can use GBNF grammar constraints to guarantee syntactically
valid OctoFlow output. Enabled by default when a grammar file is available.

```bash
octoflow chat                   # grammar auto-loaded
octoflow chat --no-grammar      # disable constraints
octoflow chat --grammar my.gbnf # custom grammar
```

## Context Engine

Chat mode automatically loads relevant knowledge based on your request:

- **L0 (always loaded):** Core syntax and rules (~440 tokens)
- **L1 (per domain):** Domain overview — loaded when keywords match (e.g., "GPU", "CSV", "image")
- **L2 (per module):** Function signatures — loaded when a domain has 2+ keyword hits
- **L3 (on demand):** Full working examples

The system stays within an 800-token skill budget. If your request touches
many domains, it loads L1 summaries only. For focused requests, it loads
deeper (L2 signatures + L3 examples).

## Conversation Memory

### Session Memory

Chat mode maintains an 8-message conversation window. You can refer
to previous code and ask for modifications.

```
> Write a function that calculates the mean of an array

fn mean(arr)
  let mut total = 0.0
  for x in arr
    total = total + x
  end
  return total / len(arr)
end

> Now add standard deviation to that

fn std_dev(arr)
  let avg = mean(arr)
  ...
end
```

### Persistent Memory

OctoFlow remembers across sessions:
- Which stdlib modules you use frequently
- Corrections the LLM had to make
- Your coding patterns

Stored in `~/.octoflow/memory.json` (32 KB cap, auto-eviction).

Disable with `--no-memory` for privacy.

## Project Configuration

Create an `OCTOFLOW.md` file in your project root to give the LLM
project-specific instructions (like `.eslintrc` for code style):

```markdown
# OCTOFLOW.md
Always use `let mut` for loop counters.
Prefer gpu_fill + gpu_add over manual loops.
Output format: JSON with keys "status" and "data".
```

OctoFlow also reads `~/.octoflow/preferences.md` for global preferences.

## Chat Commands

| Command | Alias | Action |
|---------|-------|--------|
| `:help` | `:h` | Show available commands |
| `:run` | `:r` | Re-run the generated file |
| `:show` | `:s` | Print the generated file |
| `:undo` | `:u` | Revert last generated code |
| `:diff` | `:d` | Show diff from last change |
| `:edit` | `:e` | Open last code in `$EDITOR` |
| `:history` | | Show conversation log |
| `:clear` | | Clear conversation history |
| `:clear file` | | Clear the generated .flow file |
| `:quit` | `:q` | Exit chat mode |

Multiline input: end a line with `\` to continue on the next line.

## Sandbox

Chat-generated code runs in a sandbox by default:
- File I/O scoped to current working directory
- Network access denied
- Maximum 1,000,000 loop iterations
- No process execution

Grant specific permissions with flags:

```bash
octoflow chat --allow-read=./data --allow-write=./output --allow-net
```

## Auto-Repair

When generated code has errors, the structured error (code, message, suggestion)
is fed back to the LLM automatically. The LLM sees exactly what went wrong
and generates a fix. This repeats up to 3 times before giving up.

```
> Plot a sine wave using the GUI

[Generating...]
[Error E003: undefined function 'gui_plot' — did you mean 'plot_create'?]
[Repairing (attempt 1/3)...]
[Fixed. Running...]
[Window opened with sine wave plot]
```

## All Flags

| Flag | Description |
|------|-------------|
| `--model`, `-m` | Path to GGUF model file |
| `--output`, `-o` | Output .flow file (default: `main.flow`) |
| `--max-tokens` | Generation limit (default: 512) |
| `--provider` | `local` or `api` |
| `--api-key` | API key (or use `OCTOFLOW_API_KEY` env) |
| `--api` | API endpoint URL |
| `--api-model` | Model name for API (default: `gpt-4o-mini`) |
| `--web-tools` | Enable ReAct web search + page reading |
| `--allow-net` | Allow network access (also enables web tools) |
| `--allow-read=PATH` | Allow file reads from PATH |
| `--allow-write=PATH` | Allow file writes to PATH |
| `--grammar PATH` | Custom GBNF grammar file |
| `--no-grammar` | Disable grammar-constrained decoding |
| `--think` | Enable thinking mode (deeper reasoning) |
| `--no-memory` | Disable persistent memory for privacy |
