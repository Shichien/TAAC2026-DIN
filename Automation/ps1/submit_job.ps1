param(
    [string]$TemplateJobUrl = "",
    [string]$TemplateJobInternalId = "",
    [string]$FileDir = "",
    [string]$Zip = "",
    [string]$Config = "",
    [string]$RunSh = "",
    [string[]]$File = @(),
    [string]$Name = "",
    [string]$Description = "",
    [string]$CookieFile = "taiji-output/secrets/taiji-cookie.txt",
    [string]$Bundle = "taiji-output/submit-bundle",
    [string]$Out = "",
    [switch]$Execute,
    [switch]$RunAfterSubmit,
    [switch]$AllowAddFile,
    [switch]$SkipDoctor
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

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

function Resolve-RepoPath([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
    if ([IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

function Invoke-Taac([object]$Taac, [string[]]$ArgsList) {
    & $Taac.Command @($Taac.Prefix + $ArgsList)
    if ($LASTEXITCODE -ne 0) {
        throw "taac2026 failed with exit code $LASTEXITCODE"
    }
}

if ($RunAfterSubmit -and -not $Execute) {
    throw "-RunAfterSubmit requires -Execute."
}

$hasSource = -not [string]::IsNullOrWhiteSpace($FileDir) `
    -or -not [string]::IsNullOrWhiteSpace($Zip) `
    -or -not [string]::IsNullOrWhiteSpace($Config) `
    -or -not [string]::IsNullOrWhiteSpace($RunSh) `
    -or $File.Count -gt 0

if ($hasSource) {
    if ([string]::IsNullOrWhiteSpace($TemplateJobUrl)) {
        throw "-TemplateJobUrl is required when preparing a new submit bundle."
    }
    if ([string]::IsNullOrWhiteSpace($Name)) {
        throw "-Name is required when preparing a new submit bundle."
    }
}

if ([string]::IsNullOrWhiteSpace($TemplateJobInternalId)) {
    $match = [regex]::Match($TemplateJobUrl, "/(\d{4,})(?:\D*)$")
    if ($match.Success) {
        $TemplateJobInternalId = $match.Groups[1].Value
    }
}
if ([string]::IsNullOrWhiteSpace($TemplateJobInternalId)) {
    throw "-TemplateJobInternalId is required for live submit."
}

$cookiePath = Resolve-RepoPath $CookieFile
if (-not (Test-Path -LiteralPath $cookiePath)) {
    throw "Cookie file not found: $cookiePath"
}

$taac = Resolve-TaacCommand
$bundlePath = Resolve-RepoPath $Bundle

Push-Location $RepoRoot
try {
    if ($hasSource) {
        $prepareArgs = @("prepare-submit", "--template-job-url", $TemplateJobUrl, "--name", $Name, "--out", $Bundle)
        if (-not [string]::IsNullOrWhiteSpace($Description)) { $prepareArgs += @("--description", $Description) }
        if ($RunAfterSubmit) { $prepareArgs += "--run" }
        if (-not [string]::IsNullOrWhiteSpace($FileDir)) { $prepareArgs += @("--file-dir", (Resolve-RepoPath $FileDir)) }
        if (-not [string]::IsNullOrWhiteSpace($Zip)) { $prepareArgs += @("--zip", (Resolve-RepoPath $Zip)) }
        if (-not [string]::IsNullOrWhiteSpace($Config)) { $prepareArgs += @("--config", (Resolve-RepoPath $Config)) }
        if (-not [string]::IsNullOrWhiteSpace($RunSh)) { $prepareArgs += @("--run-sh", (Resolve-RepoPath $RunSh)) }
        foreach ($item in $File) { $prepareArgs += @("--file", $item) }
        Invoke-Taac $taac $prepareArgs
    }

    if (-not $SkipDoctor) {
        Invoke-Taac $taac @("submit", "doctor", "--bundle", $bundlePath)
    }

    $submitArgs = @(
        "submit",
        "--bundle", $bundlePath,
        "--cookie-file", $cookiePath,
        "--template-job-internal-id", $TemplateJobInternalId
    )
    if (-not [string]::IsNullOrWhiteSpace($Out)) { $submitArgs += @("--out", $Out) }
    if ($AllowAddFile) { $submitArgs += "--allow-add-file" }
    if ($Execute) { $submitArgs += @("--execute", "--yes") }
    if ($RunAfterSubmit) { $submitArgs += "--run" }

    Invoke-Taac $taac $submitArgs
}
finally {
    Pop-Location
}
