param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExe
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Desktop = Join-Path $Root "desktop"
$Tauri = Join-Path $Desktop "src-tauri"
$Binaries = Join-Path $Tauri "binaries"
$TargetTriple = "x86_64-pc-windows-msvc"
$WorkerSource = Join-Path $Root "dist\ibl2svs-worker.exe"
$WorkerTarget = Join-Path $Binaries "ibl2svs-worker-$TargetTriple.exe"

Push-Location $Root
try {
    & $PythonExe -m pip install -r requirements.txt
    & $PythonExe -m PyInstaller IBL2SVSWorker.spec --clean --noconfirm

    if (!(Test-Path $Binaries)) {
        New-Item -ItemType Directory -Path $Binaries | Out-Null
    }
    Copy-Item -Force $WorkerSource $WorkerTarget

    Push-Location $Desktop
    try {
        npm ci
        npm run tauri build
    }
    finally {
        Pop-Location
    }
}
finally {
    Pop-Location
}

Write-Host "Build finished. Check desktop\src-tauri\target\release\bundle\nsis"
