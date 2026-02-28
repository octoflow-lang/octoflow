# `octoflow chat` — AI Code Generation

Generate OctoFlow code from natural language. Describe what you want,
and OctoFlow writes, validates, and runs the code for you.

## Quick Start

```bash
octoflow chat
```

```
OctoFlow v1.2 — Chat Mode (type :help for commands)

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
2. The LLM generates OctoFlow code (guided by a system prompt with full syntax reference)
3. `octoflow check` validates the code for syntax errors
4. `octoflow run` executes with sandbox protection
5. If errors occur, the structured error JSON is fed back to the LLM (up to 3 repair attempts)

## Modes

### Local Model (Default)

Uses a local GGUF model for generation. No network required.

```bash
octoflow chat
```

Requires a GGUF model file. OctoFlow looks for models in `~/.octoflow/models/`.

### API Mode

Connect to any OpenAI-compatible API endpoint.

```bash
octoflow chat --api http://localhost:8080
octoflow chat --api https://api.openai.com/v1
```

Set `OPENAI_API_KEY` environment variable for authenticated endpoints.

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
print("Bitcoin: ${price}")
```

Up to 3 tool calls per turn via a ReAct loop.

## Conversation Memory

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
  let mut sum_sq = 0.0
  for x in arr
    sum_sq = sum_sq + (x - avg) * (x - avg)
  end
  return sqrt(sum_sq / len(arr))
end
```

## Chat Commands

| Command | Action |
|---------|--------|
| `:help` | Show available commands |
| `:undo` | Revert last generated code |
| `:diff` | Show diff from last change |
| `:edit` | Open last code in editor |
| `:clear` | Clear conversation history |
| `:quit` | Exit chat mode |

Multiline input: end a line with `\` to continue on the next line.

## Sandbox

Chat-generated code runs in a sandbox:
- File I/O scoped to current working directory
- Network access denied by default
- Maximum 1,000,000 loop iterations
- No process execution

Use `--allow-read`, `--allow-write`, `--allow-net` to grant permissions:

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
