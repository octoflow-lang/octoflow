# MCP Server

OctoFlow includes a built-in [Model Context Protocol](https://modelcontextprotocol.io/)
server. Connect it to Claude Desktop, Cursor, VS Code, or any MCP client to
give your AI assistant GPU compute, data analysis, and code generation tools.

## Quick Start

```bash
octoflow mcp-serve
```

This starts a JSON-RPC 2.0 server over stdio. It responds to MCP `tools/list`
and `tools/call` requests.

## Available Tools

| Tool | Description |
|------|-------------|
| `octoflow_run` | Execute OctoFlow code and return output |
| `octoflow_chat` | Generate code from natural language, then execute it |
| `octoflow_check` | Validate code without executing (syntax + semantic errors) |
| `octoflow_analyze_csv` | Load CSV, compute stats, correlations, trends (GPU-accelerated) |
| `octoflow_gpu_sort` | Sort a numeric array using GPU acceleration |
| `octoflow_gpu_stats` | Compute mean, median, stddev, min, max on numeric data |
| `octoflow_time_series` | Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands |
| `octoflow_hypothesis_test` | t-test, chi-squared — returns statistic, p-value, significance |
| `octoflow_image` | Image operations: info, grayscale, flip, brightness (BMP) |
| `octoflow_llm` | Local LLM inference using GGUF model |

## Claude Desktop

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "octoflow": {
      "command": "octoflow",
      "args": ["mcp-serve"]
    }
  }
}
```

Restart Claude Desktop. OctoFlow tools appear in the tool picker.

## VS Code / Cursor

For Cursor, add to your MCP settings:

```json
{
  "mcpServers": {
    "octoflow": {
      "command": "octoflow",
      "args": ["mcp-serve"]
    }
  }
}
```

## Agent Mode

For programmatic use, OctoFlow supports structured JSON output:

```bash
octoflow run script.flow --output json
```

Returns:
```json
{"status": "ok", "output": "Hello!", "error": null, "exit_code": 0, "time_ms": 12}
```

Pipe data into scripts:

```bash
cat data.csv | octoflow run analyze.flow --stdin-as data --output json
```

## Sandbox

The MCP server inherits CLI permission flags. By default, all I/O is denied.
See [Permissions](permissions.md) for details.
