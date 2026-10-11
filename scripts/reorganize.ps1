# One-time, idempotent migration of the previously assembled Portable layout.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$portableRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot 'dist\Portable'))
$legacyRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot 'build\legacy-dotnet'))
if (-not $legacyRoot.StartsWith([IO.Path]::GetFullPath($projectRoot) + '\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe migration path' }
New-Item -ItemType Directory -Path (Join-Path $portableRoot 'app'),(Join-Path $portableRoot 'runtime\python'),(Join-Path $portableRoot 'docs\distribution'),$legacyRoot -Force | Out-Null
function Move-WithinTask([string]$source, [string]$destination) {
    $source = [IO.Path]::GetFullPath($source)
    $destination = [IO.Path]::GetFullPath($destination)
    $taskAbsolute = [IO.Path]::GetFullPath($projectRoot) + '\'
    if (-not $source.StartsWith($taskAbsolute,[StringComparison]::OrdinalIgnoreCase) -or -not $destination.StartsWith($taskAbsolute,[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe migration target' }
    if (Test-Path -LiteralPath $source) { Move-Item -LiteralPath $source -Destination $destination }
}
Move-WithinTask (Join-Path $portableRoot 'src\backend') (Join-Path $portableRoot 'app\backend')
$oldSrc = Join-Path $portableRoot 'src'
if ((Test-Path -LiteralPath $oldSrc) -and @(Get-ChildItem -LiteralPath $oldSrc -Force).Count -eq 0) { Remove-Item -LiteralPath $oldSrc }
Move-WithinTask (Join-Path $portableRoot 'vendor') (Join-Path $portableRoot 'app\vendor')
Move-WithinTask (Join-Path $portableRoot 'inspect') (Join-Path $portableRoot 'app\inspector')
Move-WithinTask (Join-Path $portableRoot 'legal') (Join-Path $portableRoot 'licenses')
foreach ($item in Get-ChildItem -LiteralPath (Join-Path $portableRoot 'runtime') -Force | Where-Object Name -ne 'python') {
    Move-WithinTask $item.FullName (Join-Path $portableRoot "runtime\python\$($item.Name)")
}
foreach ($name in @('package-manifest.json','SHA256SUMS.txt')) {
    Move-WithinTask (Join-Path $portableRoot $name) (Join-Path $portableRoot "docs\distribution\$name")
}
Move-WithinTask (Join-Path $portableRoot 'createdump.exe') (Join-Path $legacyRoot 'createdump.exe')
foreach ($file in Get-ChildItem -LiteralPath $portableRoot -File | Where-Object { $_.Extension -eq '.dll' -or $_.Name -like '*.deps.json' -or $_.Name -like '*.runtimeconfig.json' }) {
    Move-WithinTask $file.FullName (Join-Path $legacyRoot $file.Name)
}
foreach ($name in @('cs','de','es','fr','it','ja','ko','pl','pt-BR','ru','tr','zh-Hans','zh-Hant')) {
    Move-WithinTask (Join-Path $portableRoot $name) (Join-Path $legacyRoot $name)
}
Write-Output 'Portable dependencies organized under app, runtime, docs and licenses.'
