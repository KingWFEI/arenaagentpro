param(
    [switch]$ListModels
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$envFile = Join-Path $projectRoot ".env"

if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Missing environment file: $envFile"
}

foreach ($rawLine in Get-Content -LiteralPath $envFile -Encoding UTF8) {
    $line = $rawLine.Trim()
    if (-not $line -or $line.StartsWith("#")) {
        continue
    }
    if ($line.StartsWith("export ")) {
        $line = $line.Substring(7).TrimStart()
    }
    $separator = $line.IndexOf("=")
    if ($separator -le 0) {
        continue
    }
    $key = $line.Substring(0, $separator).Trim()
    $value = $line.Substring($separator + 1).Trim()
    if ($value.Length -ge 2 -and (($value[0] -eq '"' -and $value[-1] -eq '"') -or ($value[0] -eq "'" -and $value[-1] -eq "'"))) {
        $value = $value.Substring(1, $value.Length - 2)
    }
    if (-not [Environment]::GetEnvironmentVariable($key, "Process")) {
        [Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

$uvCommand = Get-Command "uv.exe" -ErrorAction SilentlyContinue
$uv = if ($uvCommand) { $uvCommand.Source } else { $null }

if (-not $uv) {
    $searchRoot = $projectRoot
    while ($searchRoot) {
        $candidate = Join-Path $searchRoot ".tools\uv\bin\uv.exe"
        if (Test-Path -LiteralPath $candidate) {
            $uv = $candidate
            break
        }
        $parent = Split-Path -Parent $searchRoot
        if (-not $parent -or $parent -eq $searchRoot) {
            break
        }
        $searchRoot = $parent
    }
}

if (-not $uv) {
    throw "uv executable not found. Install uv or add uv.exe to PATH."
}

Push-Location $projectRoot
try {
    if ($ListModels) {
        & $uv run --no-sync arenaagent --agent_name preliminary_baseline_agent --config config.toml --get_vlm_model
    }
    else {
        & $uv run --no-sync arenaagent --agent_name preliminary_baseline_agent --config config.toml --vlm_model VLMGPT5Config --run_times 1
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Jigsaw agent exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
