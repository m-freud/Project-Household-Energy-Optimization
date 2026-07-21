$ErrorActionPreference = 'Stop'

$fullReset = $false

foreach ($arg in $args) {
    switch ($arg) {
        '--full' {
            $fullReset = $true
        }
        '-h' {
            Write-Output 'Usage: ./reset_sqlite_results.ps1 [--full]'
            Write-Output ''
            Write-Output 'Clears simulation output tables.'
            Write-Output '--full: also run VACUUM to reclaim DB file space.'
            exit 0
        }
        '--help' {
            Write-Output 'Usage: ./reset_sqlite_results.ps1 [--full]'
            Write-Output ''
            Write-Output 'Clears simulation output tables.'
            Write-Output '--full: also run VACUUM to reclaim DB file space.'
            exit 0
        }
        default {
            [Console]::Error.WriteLine("Unknown argument: $arg")
            exit 1
        }
    }
}

$scriptDir = $PSScriptRoot
Set-Location $scriptDir

$dbPath = Join-Path $scriptDir 'sqlite\energy.db'

if (-not (Test-Path -LiteralPath $dbPath)) {
    [Console]::Error.WriteLine("Database not found at: $dbPath")
    exit 1
}

$tables = @(
    'results',
    'net_load',
    'net_cost',
    'total_consumption',
    'total_cost',
    'bess_soc',
    'bess_power',
    'ev1_soc',
    'ev1_power',
    'ev2_soc',
    'ev2_power'
)

Write-Output "Resetting simulation output tables in $dbPath"

$sqliteCmd = Get-Command sqlite3 -ErrorAction SilentlyContinue

if ($null -ne $sqliteCmd) {
    foreach ($table in $tables) {
        try {
            & $sqliteCmd.Source $dbPath "DELETE FROM $table WHERE 1;" 2>$null | Out-Null
        } catch {
            # Ignore missing-table and other per-table errors to mirror sh behavior.
        }
    }

    if ($fullReset) {
        try {
            & $sqliteCmd.Source $dbPath 'VACUUM;' 2>$null | Out-Null
        } catch {
            # Keep behavior tolerant if VACUUM fails.
        }
    }
} else {
    $pythonBin = Join-Path $scriptDir '.venv\Scripts\python.exe'

    if (-not (Test-Path -LiteralPath $pythonBin)) {
        [Console]::Error.WriteLine("sqlite3 CLI not found and Python venv missing at $pythonBin")
        [Console]::Error.WriteLine('Install sqlite3 or create the venv, then retry.')
        exit 1
    }

    $env:FULL_RESET = if ($fullReset) { 'true' } else { 'false' }

    $pythonSnippet = @'
import os
import sqlite3
from pathlib import Path

db_path = Path("sqlite/energy.db")
full_reset = os.environ.get("FULL_RESET", "false") == "true"

tables = [
    "results",
    "net_load",
    "net_cost",
    "total_consumption",
    "total_cost",
    "bess_soc",
    "bess_power",
    "ev1_soc",
    "ev1_power",
    "ev2_soc",
    "ev2_power",
]

with sqlite3.connect(db_path) as conn:
    for table in tables:
        try:
            conn.execute(f"DELETE FROM {table} WHERE 1")
        except sqlite3.OperationalError:
            pass
    conn.commit()

if full_reset:
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.execute("VACUUM")

print(f"Reset done: {db_path}")
'@

    $tempPy = Join-Path $env:TEMP ("sqlite_reset_" + [guid]::NewGuid().ToString() + ".py")
    Set-Content -LiteralPath $tempPy -Value $pythonSnippet -Encoding UTF8

    try {
        & $pythonBin $tempPy
    } finally {
        if (Test-Path -LiteralPath $tempPy) {
            Remove-Item -LiteralPath $tempPy -Force
        }
    }
}

if ($fullReset) {
    Write-Output 'Done. Simulation outputs cleared and DB vacuumed.'
} else {
    Write-Output 'Done. Simulation outputs cleared.'
}
