# OctoFlow Demo Recording Script
# ================================
# Simulates human typing for screen recording with ScreenToGif or OBS.
#
# Usage:
#   1. Open terminal, resize to 100x35 (wider = better)
#   2. Start ScreenToGif or OBS recording
#   3. Run: powershell -File scripts\demo-recording.ps1
#   4. Stop recording when script completes
#
# Prerequisites:
#   - OctoFlow release binary on PATH or in target\release\
#   - examples\hello.flow and examples\hello_gpu.flow must exist

$ErrorActionPreference = "Stop"

# --- Configuration ---
$CharDelay  = 35    # ms between characters (human speed)
$LineDelay  = 800   # ms pause after pressing Enter
$SectionDelay = 2000 # ms pause between demo sections
$OutputDelay = 1500  # ms to read output before continuing

# --- Helpers ---
function Type-Slow {
    param([string]$Text)
    foreach ($char in $Text.ToCharArray()) {
        Write-Host -NoNewline $char
        Start-Sleep -Milliseconds $CharDelay
    }
}

function Type-Line {
    param([string]$Text)
    Type-Slow $Text
    Start-Sleep -Milliseconds 200
    Write-Host ""  # newline
    Start-Sleep -Milliseconds $LineDelay
}

function Run-Command {
    param([string]$Command)
    Type-Slow $Command
    Start-Sleep -Milliseconds 300
    Write-Host ""
    # Actually execute the command
    Invoke-Expression $Command
    Start-Sleep -Milliseconds $OutputDelay
}

function Show-Comment {
    param([string]$Text)
    Write-Host ""
    Write-Host -ForegroundColor DarkCyan "# $Text"
    Start-Sleep -Milliseconds $SectionDelay
}

function Show-File {
    param([string]$Path, [string]$Label)
    Write-Host ""
    Write-Host -ForegroundColor Yellow "--- $Label ---"
    Get-Content $Path | ForEach-Object {
        Write-Host -ForegroundColor White $_
        Start-Sleep -Milliseconds 60
    }
    Write-Host -ForegroundColor Yellow "---"
    Start-Sleep -Milliseconds $SectionDelay
}

# --- Find OctoFlow binary ---
$octo = $null
if (Get-Command octoflow -ErrorAction SilentlyContinue) {
    $octo = "octoflow"
} elseif (Test-Path "target\release\octoflow.exe") {
    $octo = ".\target\release\octoflow.exe"
} else {
    Write-Host -ForegroundColor Red "Error: octoflow not found. Build with: cargo build --release --bin octoflow"
    exit 1
}

# === DEMO START ===
Clear-Host
Write-Host ""
Write-Host -ForegroundColor Cyan "  OctoFlow — GPU-native programming language"
Write-Host -ForegroundColor DarkGray "  Zero dependencies. Vulkan compute. One binary."
Write-Host ""
Start-Sleep -Milliseconds $SectionDelay

# --- Part 1: Hello World ---
Show-Comment "Part 1: Hello World"

Show-File "examples\hello.flow" "hello.flow"

Run-Command "$octo run examples\hello.flow"

# --- Part 2: GPU Hello World ---
Show-Comment "Part 2: GPU arrays — 10 million elements"

Show-File "examples\hello_gpu.flow" "hello_gpu.flow"

Run-Command "$octo run examples\hello_gpu.flow"

# --- Part 3: GPU Fractal ---
Show-Comment "Part 3: GPU fractal — 57,600 pixels parallel on GPU"

Run-Command "$octo run examples\fractal.flow --allow-write"

Write-Host -ForegroundColor Green "  → fractal.ppm saved (open with any image viewer)"
Start-Sleep -Milliseconds $SectionDelay

# --- Part 4: Benchmark ---
Show-Comment "Part 4: 10K-row CSV benchmark"

Run-Command "$octo run examples\bench_csv.flow --allow-write"

Start-Sleep -Milliseconds $SectionDelay

# === DEMO END ===
Write-Host ""
Write-Host -ForegroundColor Cyan "  github.com/anthropics/octoflow"
Write-Host -ForegroundColor DarkGray "  Zero deps • GPU compute • Single binary • MIT license"
Write-Host ""
Start-Sleep -Milliseconds 3000
