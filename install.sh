#!/bin/sh
# OctoFlow installer for Linux
# Usage: curl -fsSL https://octoflow-lang.github.io/octoflow/install.sh | sh
set -e

REPO="octoflow-lang/octoflow"
INSTALL_DIR="$HOME/.local/bin"

echo ""
echo "  OctoFlow Installer"
echo "  =================="
echo ""

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" != "Linux" ]; then
    echo "  Error: This installer is for Linux. On Windows, use:"
    echo "  irm https://octoflow-lang.github.io/octoflow/install.ps1 | iex"
    exit 1
fi

if [ "$ARCH" != "x86_64" ]; then
    echo "  Error: Only x86_64 is supported (detected: $ARCH)"
    exit 1
fi

ASSET="octoflow-linux-x64.tar.gz"

# Get latest release URL
echo "  Fetching latest release..."
TAG=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | grep '"tag_name"' | head -1 | cut -d'"' -f4)

if [ -z "$TAG" ]; then
    echo "  Error: Could not fetch latest release"
    exit 1
fi

URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"
echo "  Downloading OctoFlow $TAG..."

# Download and extract
TMP=$(mktemp -d)
curl -fsSL "$URL" -o "$TMP/$ASSET"

mkdir -p "$INSTALL_DIR"
tar xzf "$TMP/$ASSET" -C "$INSTALL_DIR"
chmod +x "$INSTALL_DIR/octoflow"
rm -rf "$TMP"

# Check PATH
case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *)
        echo ""
        echo "  Add to your PATH by adding this to ~/.bashrc or ~/.zshrc:"
        echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
        ;;
esac

# Verify
if [ -x "$INSTALL_DIR/octoflow" ]; then
    echo ""
    VERSION=$("$INSTALL_DIR/octoflow" --version 2>&1 || true)
    echo "  Installed: $VERSION"
    echo "  Location:  $INSTALL_DIR/octoflow"
    echo ""
    echo "  Run: octoflow run hello.flow"
    echo ""
else
    echo "  Error: Installation failed"
    exit 1
fi
