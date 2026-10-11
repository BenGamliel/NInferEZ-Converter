param(
    [string]$PythonSource = '',
    [string]$InspectorSource = '',
    [switch]$PrepareRuntime
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$portableRoot = Join-Path $projectRoot 'dist\Portable'
$runtimeRoot = Join-Path $portableRoot 'runtime\python'
$backendRoot = Join-Path $portableRoot 'app\backend'
$inspectorRoot = Join-Path $portableRoot 'app\inspector'
$licenseRoot = Join-Path $portableRoot 'licenses'
$distributionRoot = Join-Path $portableRoot 'docs\distribution'
New-Item -ItemType Directory -Path $portableRoot,$runtimeRoot,$backendRoot,$inspectorRoot,$licenseRoot,$distributionRoot -Force | Out-Null
& (Join-Path $PSScriptRoot 'build-icon.ps1')
if ($PrepareRuntime) {
    if (-not $PythonSource -or -not (Test-Path -LiteralPath (Join-Path $PythonSource 'python.exe'))) {
        throw 'Pass -PythonSource with a Python 3.12 installation when preparing the runtime.'
    }
    New-Item -ItemType Directory -Path (Join-Path $runtimeRoot 'Lib\site-packages') -Force | Out-Null
    foreach ($name in @('python.exe','python3.dll','python312.dll','vcruntime140.dll','vcruntime140_1.dll','LICENSE.txt')) {
        Copy-Item -LiteralPath (Join-Path $PythonSource $name) -Destination $runtimeRoot -Force
    }
    Copy-Item -LiteralPath (Join-Path $PythonSource 'DLLs') -Destination $runtimeRoot -Recurse -Force
    Get-ChildItem -LiteralPath (Join-Path $PythonSource 'Lib') | Where-Object Name -notin @('site-packages','test','idlelib','tkinter','__pycache__','ensurepip') | Copy-Item -Destination (Join-Path $runtimeRoot 'Lib') -Recurse -Force
    & uv --cache-dir (Join-Path $projectRoot 'build\uv-cache') pip install --python (Join-Path $PythonSource 'python.exe') --target (Join-Path $runtimeRoot 'Lib\site-packages') --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0+cpu' 'numpy==2.3.3'
    if ($LASTEXITCODE) { throw 'CPU runtime installation failed' }
    & uv --cache-dir (Join-Path $projectRoot 'build\uv-cache') pip install --python (Join-Path $PythonSource 'python.exe') --target (Join-Path $runtimeRoot 'Lib\site-packages') --index-url https://pypi.org/simple 'gguf==0.17.1'
    if ($LASTEXITCODE) { throw 'GGUF runtime installation failed' }
}
foreach ($project in @('BackendHost','Frontend')) {
    $publishRoot = Join-Path $projectRoot "build\publish\$project"
    & dotnet publish (Join-Path $projectRoot "src\$project\$project.csproj") -c Release -o $publishRoot --self-contained true -r win-x64 -p:NuGetAudit=false -p:RestoreIgnoreFailedSources=true
    if ($LASTEXITCODE) { throw "Build failed: $project" }
    $exeName = if ($project -eq 'Frontend') { 'NInferEZ.Converter.exe' } else { 'NInferEZ.Converter.Backend.exe' }
    Copy-Item -LiteralPath (Join-Path $publishRoot $exeName) -Destination $portableRoot -Force
}
Copy-Item -LiteralPath (Join-Path $projectRoot 'vendor') -Destination (Join-Path $portableRoot 'app') -Recurse -Force
Copy-Item -Path (Join-Path $projectRoot 'src\backend\*.py') -Destination $backendRoot -Force
Copy-Item -LiteralPath (Join-Path $projectRoot 'LICENSE') -Destination $portableRoot -Force
Copy-Item -LiteralPath (Join-Path $projectRoot 'docs') -Destination $portableRoot -Recurse -Force
$fullReadme = Get-Content -LiteralPath (Join-Path $projectRoot 'README.md') -Raw
$fullReadme.Replace('](docs/','](') | Set-Content -LiteralPath (Join-Path $portableRoot 'docs\README.md') -Encoding utf8
Copy-Item -LiteralPath (Join-Path $projectRoot 'docs\PORTABLE-README.md') -Destination (Join-Path $portableRoot 'README.md') -Force
Remove-Item -LiteralPath (Join-Path $portableRoot 'docs\PORTABLE-README.md')
foreach ($name in @('msvcp140.dll','msvcp140_1.dll','msvcp140_2.dll','concrt140.dll','vcomp140.dll')) {
    $redist = Join-Path $env:SystemRoot "System32\$name"
    if (Test-Path -LiteralPath $redist) { Copy-Item -LiteralPath $redist -Destination $runtimeRoot -Force }
}
if (-not (Test-Path -LiteralPath (Join-Path $runtimeRoot 'msvcp140.dll'))) { throw 'Missing app-local MSVC runtime for PyTorch' }
& (Join-Path $runtimeRoot 'python.exe') -I -B (Join-Path $PSScriptRoot 'compact_runtime.py')
if ($LASTEXITCODE) { throw 'Python runtime organization failed' }
if ($InspectorSource) {
    foreach ($name in @('ninfer-inspect.exe','msvcp140.dll','vcruntime140.dll','vcruntime140_1.dll')) {
        Copy-Item -LiteralPath (Join-Path $InspectorSource $name) -Destination $inspectorRoot -Force
    }
    Copy-Item -LiteralPath (Join-Path $InspectorSource 'licenses'),(Join-Path $InspectorSource 'LICENSING.md'),(Join-Path $InspectorSource 'UPSTREAM.md'),(Join-Path $InspectorSource 'THIRD-PARTY-NOTICES.txt') -Destination $licenseRoot -Recurse -Force
}
$portableAbsolute = [IO.Path]::GetFullPath($portableRoot)
$torchLibrary = Join-Path $runtimeRoot 'Lib\site-packages\torch\lib'
if (Test-Path -LiteralPath $torchLibrary) {
    Get-ChildItem -LiteralPath $torchLibrary -File -Filter '*.lib' | ForEach-Object { Remove-Item -LiteralPath $_.FullName }
}
$torchHeaders = [IO.Path]::GetFullPath((Join-Path $runtimeRoot 'Lib\site-packages\torch\include'))
if (-not $torchHeaders.StartsWith($portableAbsolute + '\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe header cleanup path' }
if (Test-Path -LiteralPath $torchHeaders) { Remove-Item -LiteralPath $torchHeaders -Recurse -Force }
# Metadata belongs with distribution records; the runtime itself is bundled in each EXE.
$runtimeConfigPath = Join-Path $projectRoot 'src\Frontend\bin\Release\net10.0-windows\win-x64\NInferEZ.Converter.runtimeconfig.json'
Copy-Item -LiteralPath $runtimeConfigPath -Destination (Join-Path $distributionRoot 'dotnet-runtime.json') -Force
$runtimeConfig = Get-Content -LiteralPath $runtimeConfigPath -Raw | ConvertFrom-Json
foreach ($framework in $runtimeConfig.runtimeOptions.includedFrameworks) {
    $packageName = if ($framework.name -eq 'Microsoft.NETCore.App') { 'microsoft.netcore.app.runtime.win-x64' } else { 'microsoft.windowsdesktop.app.runtime.win-x64' }
    $packageRoot = Join-Path $env:USERPROFILE ".nuget\packages\$packageName\$($framework.version)"
    foreach ($name in @('LICENSE','LICENSE.TXT','THIRD-PARTY-NOTICES.TXT')) {
        $notice = Join-Path $packageRoot $name
        if (Test-Path -LiteralPath $notice) { Copy-Item -LiteralPath $notice -Destination (Join-Path $licenseRoot "$packageName-$name") -Force }
    }
}
$owner = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$acl = Get-Acl -LiteralPath $projectRoot
$acl.SetAccessRule([System.Security.AccessControl.FileSystemAccessRule]::new($owner,'FullControl','ContainerInherit,ObjectInherit','None','Allow'))
Set-Acl -LiteralPath $projectRoot -AclObject $acl
Write-Output "Built organized Portable at $portableRoot"
