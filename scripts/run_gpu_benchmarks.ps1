param(
    [string]$OutputRoot = '.benchmarks/gpu-runner/manual',
    [string[]]$BenchGroups = @('all')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $repoRoot

$benchMatrix = @(
    [pscustomobject]@{ Name = 'fft'; Package = 'apollo-fft-wgpu'; Bench = 'buffer_reuse' },
    [pscustomobject]@{ Name = 'nufft'; Package = 'apollo-nufft-wgpu'; Bench = 'buffer_reuse' },
    [pscustomobject]@{ Name = 'stft'; Package = 'apollo-stft-wgpu'; Bench = 'stft_bench' },
    [pscustomobject]@{ Name = 'radon'; Package = 'apollo-radon-wgpu'; Bench = 'radon_wgpu_bench' }
)

$validGroups = @('all') + ($benchMatrix | ForEach-Object { $_.Name })
$requestedGroups = @($BenchGroups | Where-Object { $_ -and $_.Trim().Length -gt 0 } | ForEach-Object { $_.Trim().ToLowerInvariant() })
if ($requestedGroups.Count -eq 0) {
    $requestedGroups = @('all')
}

$invalidGroups = @($requestedGroups | Where-Object { $_ -notin $validGroups })
if ($invalidGroups.Count -gt 0) {
    throw "Unsupported benchmark group(s): $($invalidGroups -join ', '). Valid values: $($validGroups -join ', ')"
}

if ('all' -in $requestedGroups) {
    $selectedBenches = $benchMatrix
} else {
    $selectedBenches = @($benchMatrix | Where-Object { $_.Name -in $requestedGroups })
}

if ($selectedBenches.Count -eq 0) {
    throw 'No benchmarks selected.'
}

$outputRootPath = Join-Path $repoRoot $OutputRoot
$logsDir = Join-Path $outputRootPath 'logs'
$criterionDst = Join-Path $outputRootPath 'criterion'

if (Test-Path $outputRootPath) {
    Remove-Item $outputRootPath -Recurse -Force
}
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null

if (Test-Path 'target/criterion') {
    Remove-Item 'target/criterion' -Recurse -Force
}

$gitSha = (& git rev-parse HEAD).Trim()
$gitBranch = (& git rev-parse --abbrev-ref HEAD).Trim()
$cargoVersion = (& cargo --version).Trim()
$rustcVersion = ((& rustc -Vv) | Out-String).Trim()

$summaryLines = [System.Collections.Generic.List[string]]::new()
$summaryLines.Add('# GPU Benchmark Runner Summary')
$summaryLines.Add("")
$summaryLines.Add("- Generated at: $((Get-Date).ToUniversalTime().ToString('o'))")
$summaryLines.Add("- Git SHA: $gitSha")
$summaryLines.Add("- Git branch: $gitBranch")
$summaryLines.Add("- Runner: $($env:RUNNER_NAME) / $($env:RUNNER_OS) / $($env:RUNNER_ARCH)")
$summaryLines.Add("- Output root: $OutputRoot")
$summaryLines.Add("")
$summaryLines.Add('## Executed benches')
$summaryLines.Add("")

$benchManifest = [System.Collections.Generic.List[object]]::new()

foreach ($bench in $selectedBenches) {
    $logFileName = "$($bench.Package)--$($bench.Bench).log"
    $logPath = Join-Path $logsDir $logFileName
    $commandText = "cargo bench -p $($bench.Package) --bench $($bench.Bench) -- --noplot"

    $summaryLines.Add("- $commandText")

    & cargo bench -p $bench.Package --bench $bench.Bench -- --noplot 2>&1 |
        Tee-Object -FilePath $logPath

    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark failed: $commandText"
    }

    $benchManifest.Add([ordered]@{
        name = $bench.Name
        package = $bench.Package
        bench = $bench.Bench
        command = $commandText
        log = "logs/$logFileName"
    })
}

if (Test-Path 'target/criterion') {
    Copy-Item 'target/criterion' $criterionDst -Recurse -Force
}

$manifest = [ordered]@{
    generated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    repo_root = $repoRoot.Path
    output_root = $OutputRoot
    runner = [ordered]@{
        name = $env:RUNNER_NAME
        os = $env:RUNNER_OS
        arch = $env:RUNNER_ARCH
    }
    git = [ordered]@{
        sha = $gitSha
        branch = $gitBranch
    }
    rust = [ordered]@{
        cargo = $cargoVersion
        rustc_vv = $rustcVersion
    }
    benches = @($benchManifest)
    criterion_dir = $(if (Test-Path $criterionDst) { 'criterion' } else { $null })
}

$manifestPath = Join-Path $outputRootPath 'manifest.json'
$summaryPath = Join-Path $outputRootPath 'summary.md'
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $manifestPath -Encoding utf8
$summaryLines | Set-Content -Path $summaryPath -Encoding utf8

Write-Host "GPU benchmark bundle written to: $outputRootPath"