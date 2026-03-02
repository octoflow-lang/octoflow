$ErrorActionPreference = "Continue"

# Discover MSVC — try vswhere, then well-known paths
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($vsPath) {
        $vsToolsDir = Get-ChildItem "$vsPath\VC\Tools\MSVC" -Directory | Sort-Object Name -Descending | Select-Object -First 1
        $vsBase = $vsToolsDir.FullName
    }
}
if (-not $vsBase) {
    # Fallback: scan known install locations
    foreach ($ed in @("Community", "Professional", "Enterprise", "BuildTools")) {
        $candidate = "C:\Program Files\Microsoft Visual Studio\2022\$ed\VC\Tools\MSVC"
        if (Test-Path $candidate) {
            $vsToolsDir = Get-ChildItem $candidate -Directory | Sort-Object Name -Descending | Select-Object -First 1
            $vsBase = $vsToolsDir.FullName
            break
        }
    }
}
if (-not $vsBase) {
    Write-Error "MSVC not found. Install Visual Studio with C++ workload."
    exit 1
}

# Windows SDK
$sdkBase = "C:\Program Files (x86)\Windows Kits\10"
$sdkVer = (Get-ChildItem "$sdkBase\Lib" -Directory | Sort-Object Name -Descending | Select-Object -First 1).Name

# Vulkan SDK — use env var, then scan common paths
$vulkanBin = ""
if ($env:VULKAN_SDK) {
    $vulkanBin = "$env:VULKAN_SDK\Bin"
} else {
    foreach ($dir in (Get-ChildItem "C:\VulkanSDK" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending)) {
        if (Test-Path "$($dir.FullName)\Bin") {
            $vulkanBin = "$($dir.FullName)\Bin"
            break
        }
    }
}

$env:PATH = "$vsBase\bin\Hostx64\x64;$env:USERPROFILE\.cargo\bin;$vulkanBin;" + $env:PATH
$env:LIB = "$vsBase\lib\x64;$sdkBase\Lib\$sdkVer\um\x64;$sdkBase\Lib\$sdkVer\ucrt\x64"
$env:INCLUDE = "$vsBase\include;$sdkBase\Include\$sdkVer\ucrt;$sdkBase\Include\$sdkVer\um;$sdkBase\Include\$sdkVer\shared"

Set-Location $PSScriptRoot
& cargo @args 2>&1 | Out-String -Stream
exit $LASTEXITCODE
