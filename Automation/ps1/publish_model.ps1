param(
    [Parameter(Mandatory = $true)][string]$InstanceId,
    [Parameter(Mandatory = $true)][string]$Ckpt,
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Description,
    [string]$CookieFile = "taiji-output/secrets/taiji-cookie.txt",
    [string]$Out = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$nodeScript = Join-Path $PSScriptRoot "..\mjs\publish_model.mjs"
if (-not (Test-Path -LiteralPath $nodeScript)) {
    throw "Missing helper script: $nodeScript"
}

$argsList = @(
    $nodeScript,
    "--instance-id", $InstanceId,
    "--ckpt", $Ckpt,
    "--name", $Name,
    "--description", $Description,
    "--cookie-file", $CookieFile
)
if (-not [string]::IsNullOrWhiteSpace($Out)) { $argsList += @("--out", $Out) }

Push-Location $RepoRoot
try {
    & node @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "Publish model failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
