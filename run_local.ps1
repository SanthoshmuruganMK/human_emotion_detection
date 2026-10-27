$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$app = Join-Path $PSScriptRoot "app.py"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment was not found: $python"
}
if (-not (Test-Path -LiteralPath $app)) {
    throw "App entry point was not found: $app"
}

$listeners = @(
    Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -in @(4000, 5000) }
)
if ($listeners.Count -gt 0) {
    $occupiedPorts = ($listeners | ForEach-Object {
        "port $($_.LocalPort) (process $($_.OwningProcess))"
    }) -join ", "
    throw "Emotion Studio needs ports 4000 and 5000, but $occupiedPorts already in use. Stop the existing app, then run this script again."
}

$ownerCodePath = Join-Path $env:LOCALAPPDATA "EmotionStudio\owner_access_code.dpapi"
$ownerCodeSecure = $null
if (Test-Path -LiteralPath $ownerCodePath) {
    $encryptedOwnerCode = Get-Content -LiteralPath $ownerCodePath -Raw
    try {
        $ownerCodeSecure = ConvertTo-SecureString $encryptedOwnerCode
    }
    catch {
        throw "The saved owner passphrase could not be decrypted for this Windows account. Remove '$ownerCodePath' and run this script to set it again."
    }
    Write-Host "Using the owner passphrase saved for this Windows account."
}
else {
    $ownerCodeSecure = Read-Host "Set a private owner passphrase (at least 16 characters; saved securely for this Windows account)" -AsSecureString
    $ownerCode = [System.Net.NetworkCredential]::new("", $ownerCodeSecure).Password
    if ($ownerCode.Length -lt 16) {
        throw "The owner passphrase must contain at least 16 characters."
    }

    $ownerCodeDirectory = Split-Path -Parent $ownerCodePath
    New-Item -ItemType Directory -Path $ownerCodeDirectory -Force | Out-Null
    ConvertFrom-SecureString $ownerCodeSecure |
        Set-Content -LiteralPath $ownerCodePath -NoNewline
}

$ownerCode = [System.Net.NetworkCredential]::new("", $ownerCodeSecure).Password
$env:OWNER_ACCESS_CODE = $ownerCode
$ownerCode = $null
$ownerCodeSecure = $null
$env:APP_HOST = "127.0.0.1"
$env:LOCAL_DUAL_PORTS = "1"
try {
    Write-Host "Starting Emotion Studio. Keep this window open."
    Write-Host "Sign in: http://127.0.0.1:5000/login"
    Write-Host "Private owner data: http://127.0.0.1:4000/owner"
    & $python $app
    if ($LASTEXITCODE -ne 0) {
        throw "Emotion Studio exited with code $LASTEXITCODE."
    }
}
finally {
    Remove-Item Env:OWNER_ACCESS_CODE -ErrorAction SilentlyContinue
    Remove-Item Env:LOCAL_DUAL_PORTS -ErrorAction SilentlyContinue
}
