# Windows twin of gliner.sh: install, run and inspect the GLiNER2.5 classifier daemon.
#   scripts\gliner.ps1 setup|start|stop|status|check|logs
param([string]$Cmd = '')
$ErrorActionPreference = 'Stop'

$Server = Join-Path $PSScriptRoot '..\server\classifier.py'
$Url = 'http://127.0.0.1:8765' # ponytail: fixed, matches DAEMON in hooks/register.tsx
$Log = Join-Path $HOME '.cache\model-router\classifier.log'
$Wait = if ($env:ROUTER_WAIT) { [int]$env:ROUTER_WAIT } else { 900 }
$env:PATH = "$HOME\.local\bin;$env:PATH"

function Health { try { (Invoke-WebRequest "$Url/health" -TimeoutSec 2 -UseBasicParsing).Content } catch { $null } }
function NeedUv {
  if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv is not installed: run 'scripts\gliner.ps1 setup' or see https://docs.astral.sh/uv/" }
}
function Start-Daemon {
  NeedUv
  if (-not (Health)) {
    New-Item -ItemType Directory -Force (Split-Path $Log) | Out-Null
    # Same command the mod runs at session start, so both share uv's cached environment.
    Start-Process cmd -WindowStyle Hidden -ArgumentList '/c', "uv run --script `"$Server`" >> `"$Log`" 2>&1"
    "starting daemon (log: $Log)"
  }
  for ($i = 0; -not ((Health) -match '"ready": ?true'); $i += 2) {
    if ($i -gt $Wait) { throw "not ready after ${Wait}s; see $Log" }
    Start-Sleep 2
  }
  "ready: $(Health)"
}

switch ($Cmd) {
  'setup' {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
      'installing uv (https://astral.sh/uv)'
      $env:UV_NO_MODIFY_PATH = '1'
      Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    }
    NeedUv
    'installing torch + gliner2 (first run only)'
    uv run --script $Server --check
    Start-Daemon
  }
  'start' { Start-Daemon }
  'stop' {
    $p = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*server*classifier.py*' -and $_.ProcessId -ne $PID }
    if ($p) { $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; 'stopped' } else { 'not running' }
  }
  'status' { if ($h = Health) { $h } else { 'not running'; exit 1 } }
  'check' { NeedUv; uv run --script $Server --check }
  'logs' { New-Item -ItemType File -Force $Log | Out-Null; Get-Content $Log -Wait -Tail 20 }
  default { Get-Content $PSCommandPath -TotalCount 3 | Select-Object -Skip 1; exit 2 }
}
