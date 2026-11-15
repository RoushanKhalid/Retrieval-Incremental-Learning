# Root launcher forwarding to scripts\run_demo.ps1
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $scriptDir "scripts\run_demo.ps1") @args
