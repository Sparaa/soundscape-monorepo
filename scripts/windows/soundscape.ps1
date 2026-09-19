<#
Soundscape helper for Windows PowerShell (5.1 or 7). Run from anywhere inside the repo:

    .\scripts\windows\soundscape.ps1 <command> [arg]

  check      Docker, Compose v2, WSL 2 engine, GPU visible from a container, .env, line endings
  up         docker compose --profile gpu --profile llm up -d --build   (creates .env from .env.example if missing)
  down       stop everything (keeps images, volumes and the library)
  status     compose ps + api health + Ollama models
  logs [svc] follow a service's log (default api): api web yue2 sheetsage clipgrab ollama
  pull-llm   download the writer model named by LLM_MODEL in .env into the bundled Ollama
  weights    pre-download the ~12 GB of music weights (YuE2-3B, YuE2-Vae, SheetSage2, MERT)
  update     git pull + rebuild
  test       api + web test suites in throwaway containers (needs nothing on the host)

Everything here is a thin wrapper over plain docker compose commands; docs/WINDOWS.md lists the equivalents.
#>
param(
  [Parameter(Position = 0)][string]$Command = "help",
  [Parameter(Position = 1)][string]$Arg = ""
)

# "Continue", not "Stop": Windows PowerShell 5.1 turns redirected stderr of native commands (docker prints progress
# there) into terminating errors under "Stop". Failures are detected through $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$Profiles = @("--profile", "gpu", "--profile", "llm")

function Compose {
  param([string[]]$A)
  & docker compose @Profiles @A
  if ($LASTEXITCODE -ne 0) { throw "docker compose $($A -join ' ') failed (exit $LASTEXITCODE)" }
}

function Read-DotEnv {
  $vars = @{}
  if (Test-Path ".env") {
    foreach ($line in Get-Content ".env") {
      $t = $line.Trim()
      if ($t -eq "" -or $t.StartsWith("#")) { continue }
      $i = $t.IndexOf("=")
      if ($i -lt 1) { continue }
      $vars[$t.Substring(0, $i).Trim()] = $t.Substring($i + 1).Trim()
    }
  }
  return $vars
}

function Ensure-DotEnv {
  if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "created .env from .env.example (edit MUSIC_BUDGET_GIB=12 if your GPU has 16 GB)" -ForegroundColor Yellow
  }
}

function Step {
  param([string]$Name, [scriptblock]$Test, [string]$Hint)
  $ok = $false
  try { $ok = & $Test } catch { $ok = $false }
  if ($ok) { Write-Host ("  [ OK ] " + $Name) -ForegroundColor Green }
  else { Write-Host ("  [FAIL] " + $Name + "  ->  " + $Hint) -ForegroundColor Red }
  return [bool]$ok
}

function Do-Check {
  Write-Host "Soundscape preflight ($Root)"
  $all = $true
  $all = (Step "docker CLI" { (& docker version --format '{{.Client.Version}}' 2>$null) -ne $null } "install Docker Desktop and start it") -and $all
  $all = (Step "docker engine is Linux (WSL 2 backend)" { (& docker info --format '{{.OSType}}' 2>$null) -eq "linux" } "Docker Desktop -> Settings -> General -> 'Use the WSL 2 based engine'") -and $all
  $all = (Step "docker compose v2" { ((& docker compose version --short 2>$null) -split '\.')[0] -ge 2 } "update Docker Desktop") -and $all
  $all = (Step "GPU visible from a container (nvidia-smi)" {
      $out = & docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi 2>&1
      ($LASTEXITCODE -eq 0) -and (($out | Out-String) -match "NVIDIA-SMI")
    } "update the Windows NVIDIA driver; WSL 2 engine must be on; never install drivers inside WSL") -and $all
  $all = (Step "VM memory >= 12 GB" {
      $gb = [int](((& docker run --rm alpine free -g 2>$null) | Select-String '^Mem:') -split '\s+')[1]
      $gb -ge 12
    } "create C:\Users\<you>\.wslconfig with [wsl2] memory=24GB, then wsl --shutdown and restart Docker Desktop") -and $all
  $all = (Step "sidecar entrypoint has LF line endings" {
      -not ((Get-Content "sidecars\clipgrab-sidecar\entrypoint.sh" -Raw) -match "`r")
    } "git config core.autocrlf false; git rm -r --cached .; git reset --hard") -and $all
  $all = (Step ".env exists" { Test-Path ".env" } "copy .env.example .env") -and $all
  if (Test-Path ".env") {
    $e = Read-DotEnv
    Write-Host ("  LLM: " + $e["LLM_BASE_URL"] + "  model " + $e["LLM_MODEL"] + "   VRAM budget: " + $e["MUSIC_BUDGET_GIB"] + " GiB")
  }
  if ($all) { Write-Host "all checks passed - run: .\scripts\windows\soundscape.ps1 up" -ForegroundColor Green }
  else { Write-Host "fix the FAIL lines first (docs/WINDOWS.md explains each)" -ForegroundColor Yellow }
}

function Do-Status {
  Compose @("ps")
  Write-Host "`napi health (http://localhost:3021/healthz):"
  try {
    $h = Invoke-RestMethod -Uri "http://localhost:3021/healthz" -TimeoutSec 10
    $h | ConvertTo-Json -Depth 6
  } catch { Write-Host "  api not reachable: $($_.Exception.Message)" -ForegroundColor Red }
  Write-Host "`nOllama models:"
  & docker compose @Profiles exec ollama ollama list 2>&1
  Write-Host "`nGPU:"
  & docker compose @Profiles exec yue2 nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>&1
}

switch ($Command.ToLower()) {
  "check"    { Do-Check }
  "up"       { Ensure-DotEnv; Compose @("up", "-d", "--build")
               Write-Host "`nup. Next: pull-llm (once), then status, then open http://localhost:3020" -ForegroundColor Green }
  "down"     { Compose @("down") }
  "status"   { Do-Status }
  "logs"     { $svc = if ($Arg) { $Arg } else { "api" }; Compose @("logs", "-f", "--tail", "100", $svc) }
  "pull-llm" { $e = Read-DotEnv; $m = if ($e["LLM_MODEL"]) { $e["LLM_MODEL"] } else { "qwen2.5:7b-instruct" }
               Write-Host "pulling $m into the soundscape-ollama volume (about 4.7 GB for the default)"
               Compose @("exec", "ollama", "ollama", "pull", $m) }
  "weights"  { Write-Host "downloading YuE2-3B + YuE2-Vae (~7.8 GB) ..."
               Compose @("run", "--rm", "--no-deps", "yue2", "python", "-c", "from huggingface_hub import snapshot_download as d; d('m-a-p/YuE2-3B'); d('m-a-p/YuE2-Vae')")
               Write-Host "downloading SheetSage2 + MERT-v2-FullSong (~2.7 GB) ..."
               Compose @("run", "--rm", "--no-deps", "sheetsage", "python", "-c", "from huggingface_hub import snapshot_download as d; d('m-a-p/SheetSage2'); d('m-a-p/MERT-v2-FullSong')") }
  "update"   { & git pull; if ($LASTEXITCODE -ne 0) { throw "git pull failed" }; Compose @("up", "-d", "--build") }
  "test"     { $pwdLinux = $Root
               & docker run --rm -v "${pwdLinux}\api:/app" -w /app python:3.12-slim sh -c "apt-get install -y -qq ffmpeg >/dev/null 2>&1; pip install -q -r requirements.txt >/dev/null && python -m pytest -q tests -p no:cacheprovider"
               & docker run --rm -v "${pwdLinux}\web:/app" -w /app node:22-alpine sh -c "[ -d node_modules ] || npm install --silent --no-audit --no-fund; node node_modules/vitest/vitest.mjs run && node node_modules/typescript/bin/tsc --noEmit -p . && echo 'tsc: clean'" }
  default    { $src = Get-Content $PSCommandPath -Raw
               $m = [regex]::Match($src, '(?s)<#(.*?)#>')
               Write-Host $m.Groups[1].Value.Trim() }
}
