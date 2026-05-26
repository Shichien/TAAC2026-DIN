param(
    [Parameter(Mandatory = $true)][int]$MouldId,
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$InferDir = "",
    [string[]]$File = @(),
    [string]$CookieFile = "taiji-output/secrets/taiji-cookie.txt",
    [string]$Creator = "ams_2026_1029731869646210001",
    [string]$ImageName = "",
    [string]$Out = "",
    [switch]$Execute
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$nodeScript = Join-Path $PSScriptRoot "..\mjs\taiji_eval_submit.mjs"
if (-not (Test-Path -LiteralPath $nodeScript)) {
    throw "Missing helper script: $nodeScript"
}

$argsList = @(
    $nodeScript,
    "--mould-id", [string]$MouldId,
    "--name", $Name,
    "--cookie-file", $CookieFile,
    "--creator", $Creator
)
if (-not [string]::IsNullOrWhiteSpace($InferDir)) { $argsList += @("--infer-dir", $InferDir) }
foreach ($item in $File) { $argsList += @("--file", $item) }
if (-not [string]::IsNullOrWhiteSpace($ImageName)) { $argsList += @("--image-name", $ImageName) }
if (-not [string]::IsNullOrWhiteSpace($Out)) { $argsList += @("--out", $Out) }
if ($Execute) { $argsList += @("--execute", "--yes") }

Push-Location $RepoRoot
try {
    & node @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "Eval submit failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
