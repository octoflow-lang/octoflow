#!/usr/bin/env pwsh
# OctoFlow installer for Windows
# Usage: irm https://octoflow-lang.github.io/octoflow/install.ps1 | iex

$ErrorActionPreference = "Stop"

$repo = "octoflow-lang/octoflow"
$installDir = "$env:LOCALAPPDATA\octoflow"

Write-Host ""
Write-Host "  OctoFlow Installer" -ForegroundColor Cyan
Write-Host "  ==================" -ForegroundColor DarkGray
Write-Host ""

# Get latest release URL
Write-Host "  Fetching latest release..." -ForegroundColor Gray
$release = Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest"
$tag = $release.tag_name
$asset = $release.assets | Where-Object { $_.name -like "*windows*" -and $_.name -like "*.zip" }

if (-not $asset) {
    Write-Host "  Error: No Windows binary found in release $tag" -ForegroundColor Red
    exit 1
}

$url = $asset.browser_download_url
Write-Host "  Downloading OctoFlow $tag..." -ForegroundColor Gray

# Download to temp
$tmp = Join-Path $env:TEMP "octoflow-download.zip"
Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing

# Create install directory
if (-not (Test-Path $installDir)) {
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
}

# Extract
Write-Host "  Installing to $installDir..." -ForegroundColor Gray
$extractDir = Join-Path $env:TEMP "octoflow-extract"
if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force }
Expand-Archive -Path $tmp -DestinationPath $extractDir -Force
Remove-Item $tmp -Force

# Copy binary (handles both flat and nested layouts)
$nested = Join-Path $extractDir "octoflow\octoflow.exe"
$flat = Join-Path $extractDir "octoflow.exe"
if (Test-Path $nested) { Copy-Item $nested $installDir -Force }
elseif (Test-Path $flat) { Copy-Item $flat $installDir -Force }
else { Get-ChildItem $extractDir -Recurse -Filter "octoflow.exe" | Select-Object -First 1 | Copy-Item -Destination $installDir -Force }
Remove-Item $extractDir -Recurse -Force

# Unblock the exe (removes Mark of the Web if any)
$exe = Join-Path $installDir "octoflow.exe"
if (Test-Path $exe) {
    Unblock-File -Path $exe
}

# Add to user PATH if not already there
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$installDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$installDir", "User")
    Write-Host "  Added to PATH (restart your terminal to use)" -ForegroundColor Yellow
}

# Verify
if (Test-Path $exe) {
    Write-Host ""
    $version = & $exe --version 2>&1
    Write-Host "  Installed: $version" -ForegroundColor Green
    Write-Host "  Location:  $exe" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Run: octoflow run hello.flow" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "  Error: Installation failed" -ForegroundColor Red
    exit 1
}
