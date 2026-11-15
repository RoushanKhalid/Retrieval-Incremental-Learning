# Root launcher forwarding to scripts\launch_app.ps1
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $scriptDir "scripts\launch_app.ps1") @args
