# Installation

OctoFlow is a single binary. No installer, no dependencies, no SDK.

## Download

Get the latest release for your platform:

- **Windows x64:** [octoflow-v1.5.9-windows-x64.zip](https://github.com/octoflow-lang/octoflow/releases/latest)
- **Linux x64:** [octoflow-v1.5.9-linux-x64.tar.gz](https://github.com/octoflow-lang/octoflow/releases/latest)
- **macOS aarch64:** [octoflow-v1.5.9-aarch64-macos.tar.gz](https://github.com/octoflow-lang/octoflow/releases/latest) (Apple Silicon)

## Windows

1. Download and unzip
2. Move `octoflow.exe` to a directory on your PATH (e.g., `C:\Tools\`)
3. Or add the unzip directory to PATH:
   ```
   setx PATH "%PATH%;C:\path\to\octoflow"
   ```
4. Verify: `octoflow --version`

**Windows Defender:** You may see a SmartScreen warning on first run.
Click "More info" then "Run anyway". The binary is unsigned.

## Linux

```bash
tar xzf octoflow-v1.5.9-linux-x64.tar.gz
chmod +x octoflow
sudo mv octoflow /usr/local/bin/
octoflow --version
```

Or use the Linux installer script:

```bash
curl -fsSL https://octoflow-lang.github.io/octoflow/install.sh | sh
```

## macOS (Apple Silicon)

```bash
tar xzf octoflow-v1.5.9-aarch64-macos.tar.gz
chmod +x octoflow
mv octoflow /usr/local/bin/
octoflow --version
```

Use direct release download on macOS. The `install.sh` script is Linux-only.

GPU compute uses MoltenVK (Vulkan on Metal). No additional setup needed
on M1/M2/M3/M4 Macs.

## GPU Detection

OctoFlow auto-detects Vulkan-capable GPUs. Verify in the REPL:

```
$ octoflow repl
OctoFlow v1.5.9 — GPU-native language
GPU: NVIDIA GeForce GTX 1660 SUPER     <-- your GPU here
766 stdlib modules | :help | :time

>>> :gpu
Device: NVIDIA GeForce GTX 1660 SUPER
API: Vulkan 1.1
Memory: 6144 MB
f16: supported
```

If no GPU line appears, check:
- **Vulkan drivers** are installed (NVIDIA: included with driver, AMD: included with Adrenalin, Intel: included with driver)
- **Integrated GPUs** work if they support Vulkan 1.0+
- OctoFlow falls back to CPU for all GPU operations — everything still runs

### macOS troubleshooting (MoltenVK portability)

As of v1.5.9, OctoFlow enables `VK_KHR_portability_enumeration` and
`VK_KHR_portability_subset` automatically when available. This should
resolve `No GPU detected` on Apple Silicon with MoltenVK.

If you still see GPU detection failures, please open an issue with:

```bash
VK_LOADER_DEBUG=all octoflow run examples/hello_gpu.flow
```

Include the output so maintainers can diagnose the issue.

## Updating

```bash
octoflow update          # download and install latest
octoflow update --check  # check for updates without installing
```

## Requirements

- **GPU:** Any Vulkan 1.0+ GPU (optional — CPU fallback for all operations)
- **OS:** Windows 10+, Linux (glibc 2.17+), macOS 12+ (Apple Silicon)
- **Disk:** ~3.7 MB for the binary
- **RAM:** 64 MB minimum, more for large GPU arrays
