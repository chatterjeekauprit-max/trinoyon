$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
  Write-Error 'Project environment missing. Follow INSTALL.md first.'
}
& $python (Join-Path $PSScriptRoot 'app.py') @args
