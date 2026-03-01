#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Sync ML model files to EC2 server.
    Run this after retraining models locally.

.DESCRIPTION
    Uploads only the required "latest" + "down_risk" model files
    from ml_models/saved_models/ to the EC2 server.
    These files are git-ignored, so CI/CD won't deploy them.

.EXAMPLE
    .\deploy\sync-models.ps1
    .\deploy\sync-models.ps1 -DryRun
    .\deploy\sync-models.ps1 -All   # upload ALL files in saved_models/
#>

param(
    [switch]$DryRun,
    [switch]$All,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\titan-bot-key.pem",
    [string]$ServerIP = "13.232.89.92",
    [string]$ServerUser = "ubuntu",
    [string]$DeployDir = "/home/ubuntu/titan"
)

$ErrorActionPreference = "Stop"

$LocalModelsDir = Join-Path $PSScriptRoot "..\agentic_trader\ml_models\saved_models"
$RemoteModelsDir = "$DeployDir/agentic_trader/ml_models/saved_models"

if (-not (Test-Path $LocalModelsDir)) {
    Write-Error "Local models directory not found: $LocalModelsDir"
    exit 1
}

if (-not (Test-Path $KeyPath)) {
    Write-Error "SSH key not found: $KeyPath"
    exit 1
}

# Determine which files to upload
if ($All) {
    $files = Get-ChildItem -Path $LocalModelsDir -File
} else {
    # Only upload files needed for live scoring:
    #   *_latest*  (move_predictor, meta_gate, meta_direction, meta_labeling)
    #   down_risk_*
    $files = Get-ChildItem -Path $LocalModelsDir -File | Where-Object {
        $_.Name -match "^(move_predictor_latest|meta_gate_latest|meta_direction_latest|meta_labeling_latest|down_risk_)" -or
        $_.Name -match "_latest_"
    }
}

if ($files.Count -eq 0) {
    Write-Host "No matching model files found." -ForegroundColor Yellow
    exit 0
}

$totalSize = ($files | Measure-Object -Property Length -Sum).Sum
$sizeMB = [math]::Round($totalSize / 1MB, 1)

Write-Host ""
Write-Host "=== Titan Model Sync ===" -ForegroundColor Cyan
Write-Host "Files to upload: $($files.Count)"
Write-Host "Total size:      $sizeMB MB"
Write-Host "Destination:     ${ServerUser}@${ServerIP}:${RemoteModelsDir}"
Write-Host ""

if ($DryRun) {
    Write-Host "[DRY RUN] Would upload:" -ForegroundColor Yellow
    $files | ForEach-Object { Write-Host "  $($_.Name) ($([math]::Round($_.Length / 1KB, 1)) KB)" }
    exit 0
}

# Ensure remote directory exists
Write-Host "Ensuring remote directory exists..." -ForegroundColor Gray
ssh -i $KeyPath -o StrictHostKeyChecking=no "${ServerUser}@${ServerIP}" "mkdir -p $RemoteModelsDir"

# Upload files
$uploaded = 0
$failed = 0

foreach ($f in $files) {
    Write-Host "  Uploading $($f.Name)..." -NoNewline
    try {
        scp -i $KeyPath -o StrictHostKeyChecking=no $f.FullName "${ServerUser}@${ServerIP}:${RemoteModelsDir}/"
        Write-Host " OK" -ForegroundColor Green
        $uploaded++
    } catch {
        Write-Host " FAILED" -ForegroundColor Red
        $failed++
    }
}

Write-Host ""
Write-Host "=== Sync Complete ===" -ForegroundColor Cyan
Write-Host "Uploaded: $uploaded  Failed: $failed" -ForegroundColor $(if ($failed -gt 0) { "Yellow" } else { "Green" })

# Verify remote file count
$remoteCount = ssh -i $KeyPath -o StrictHostKeyChecking=no "${ServerUser}@${ServerIP}" "ls $RemoteModelsDir | wc -l"
Write-Host "Remote model files: $remoteCount"

# Optionally restart the bot to pick up new models
$restart = Read-Host "Restart titan-bot to load new models? (y/N)"
if ($restart -eq 'y') {
    Write-Host "Restarting titan-bot..." -ForegroundColor Yellow
    ssh -i $KeyPath -o StrictHostKeyChecking=no "${ServerUser}@${ServerIP}" "sudo systemctl restart titan-bot"
    Start-Sleep -Seconds 3
    $status = ssh -i $KeyPath -o StrictHostKeyChecking=no "${ServerUser}@${ServerIP}" "sudo systemctl is-active titan-bot"
    Write-Host "Service status: $status" -ForegroundColor $(if ($status -eq 'active') { "Green" } else { "Red" })
}
