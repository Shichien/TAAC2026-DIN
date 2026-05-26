param(
    [string]$JobInternalId = "",
    [string]$CkptUrl = "",
    [string]$EvalTaskId = "",
    [string]$CookieFile = "taiji-output/secrets/taiji-cookie.txt",
    [string]$Out = "",
    [switch]$All,
    [switch]$NoIncremental
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

function Resolve-RepoPath([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
    if ([IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

function Resolve-TaacCommand {
    $global = Get-Command "taac2026" -ErrorAction SilentlyContinue
    if ($null -ne $global) {
        return @{ Command = $global.Source; Prefix = @() }
    }
    $fallback = "C:\Users\26421\.codex\skills\taac2026-cli\bin\taac2026.mjs"
    if (-not (Test-Path -LiteralPath $fallback)) {
        throw "taac2026 command not found and fallback path is missing: $fallback"
    }
    return @{ Command = "node"; Prefix = @($fallback) }
}

function Invoke-Taac([object]$Taac, [string[]]$ArgsList) {
    & $Taac.Command @($Taac.Prefix + $ArgsList)
    if ($LASTEXITCODE -ne 0) {
        throw "taac2026 failed with exit code $LASTEXITCODE"
    }
}

function Write-JsonFile($Object, [string]$Path) {
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent | Out-Null
    }
    $Object | ConvertTo-Json -Depth 32 | Set-Content -Path $Path -Encoding UTF8
}

$cookiePath = Resolve-RepoPath $CookieFile
if (-not (Test-Path -LiteralPath $cookiePath)) {
    throw "Cookie file not found: $cookiePath"
}

if (-not [string]::IsNullOrWhiteSpace($EvalTaskId)) {
    $nodeScript = Join-Path $PSScriptRoot "..\mjs\taiji_eval_logs.mjs"
    if (-not (Test-Path -LiteralPath $nodeScript)) {
        throw "Missing helper script: $nodeScript"
    }
    $argsList = @($nodeScript, "--task-id", $EvalTaskId, "--cookie-file", $cookiePath)
    if (-not [string]::IsNullOrWhiteSpace($Out)) { $argsList += @("--out", $Out) }
    Push-Location $RepoRoot
    try {
        & node @argsList
        if ($LASTEXITCODE -ne 0) {
            throw "Eval log fetch failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
    return
}

if (-not $All -and [string]::IsNullOrWhiteSpace($JobInternalId) -and [string]::IsNullOrWhiteSpace($CkptUrl)) {
    throw "Pass -JobInternalId, -CkptUrl, -EvalTaskId, or -All."
}

$taac = Resolve-TaacCommand
$argsList = @("scrape", "--cookie-file", $cookiePath, "--direct")
if (-not [string]::IsNullOrWhiteSpace($CkptUrl)) {
    $argsList += @("--url", $CkptUrl)
} else {
    $argsList += "--all"
    if (-not $NoIncremental) { $argsList += "--incremental" }
    if (-not [string]::IsNullOrWhiteSpace($JobInternalId)) {
        $argsList += @("--job-internal-id", $JobInternalId)
    }
}

Push-Location $RepoRoot
try {
    Invoke-Taac $taac $argsList
}
finally {
    Pop-Location
}
