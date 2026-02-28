# Installation

OctoFlow is a single binary with zero dependencies. Download, unzip, run.

## Download

### Windows (PowerShell)

```powershell
# Download and extract
Invoke-WebRequest -Uri https://github.com/octoflow-lang/octoflow/releases/latest/download/octoflow-windows.zip -OutFile octoflow.zip
Expand-Archive octoflow.zip -DestinationPath $env:LOCALAPPDATA\octoflow

# Add to PATH (current session)
$env:PATH += ";$env:LOCALAPPDATA\octoflow"

# Add to PATH (permanent)
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";$env:LOCALAPPDATA\octoflow", "User")

# Verify
octoflow --version
```

### Linux

```bash
# Download and extract
curl -L https://github.com/octoflow-lang/octoflow/releases/latest/download/octoflow-linux.tar.gz | tar xz

# Install
sudo mv octoflow /usr/local/bin/

# Verify
octoflow --version
```

## GPU Requirements

OctoFlow uses Vulkan for GPU compute. Any GPU with Vulkan 1.0+ support works:

- **NVIDIA**: GTX 900 series or newer (driver 470+)
- **AMD**: RX 400 series or newer (Adrenalin 21.1+)
- **Intel**: Gen 9 (Skylake) or newer

### No GPU? No Problem.

OctoFlow v1.2 includes CPU fallback for all GPU operations. Programs that use
`gpu_matmul`, `gpu_add`, `gpu_sort`, etc. run on CPU-only machines. You'll see:

```
[note] No GPU detected — using CPU fallback
```

Performance will be lower for GPU-heavy workloads, but everything works.

## VS Code Extension

Syntax highlighting for `.flow` files:

```bash
code --install-extension octoflow-0.1.0.vsix
```

Download `octoflow-0.1.0.vsix` from the
[releases page](https://github.com/octoflow-lang/octoflow/releases).

Features:
- 90+ builtin keywords highlighted
- String interpolation (`{var}` inside strings)
- Code folding for all block types
- Comment toggling
- All operators and control flow keywords

## Verify Installation

```bash
# Version check
octoflow --version

# GPU detection
octoflow repl
# Shows: GPU: NVIDIA GeForce GTX 1660 SUPER (or CPU fallback message)

# Run a test program
echo 'print("Hello, OctoFlow!")' > hello.flow
octoflow run hello.flow

# Try chat mode
octoflow chat
```

## File Associations

OctoFlow uses the `.flow` file extension. If you're using VS Code with the
extension installed, `.flow` files will be automatically recognized.

For other editors, OctoFlow syntax is similar to Ruby/Python:
- `//` line comments
- `end` block terminator
- No semicolons, no braces
- String interpolation with `{variable}`

## Updating

Download the latest release and replace the binary. OctoFlow is a single file
with no configuration to migrate. Your `.flow` programs are fully backward
compatible between versions.
