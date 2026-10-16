# AURA 基线一键运行脚本（PowerShell）
# 用法（在项目根目录 d:\py\AURA 下运行）：
#   powershell -ExecutionPolicy Bypass -File baseline/scripts/run_baseline.ps1 -Stage all
# 参数 Stage 取值：train / test / diagnose / visualize / all
#
# 说明：baseline 现在是一个独立包，统一用 `python -m baseline.xxx` 运行，
#       配置路径由 baseline/utils.py 相对“包目录”解析，因此这里传 configs/baseline.yaml。

param(
    [string]$Stage = "all",
    [string]$Python = "E:\conda\envs\pytorch\python.exe",
    [string]$Config = "configs/baseline.yaml"
)

$ErrorActionPreference = "Stop"
# 脚本位于 baseline/scripts/ 下：先退到 scripts 的父目录(baseline)，再退一级才是项目根
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$env:PYTHONPATH = $Root     # 让 `python -m baseline.xxx` 能找到 baseline 包

function Run-Train      { & $Python -m baseline.train_baseline --config $Config }
function Run-Test       { & $Python -m baseline.test_baseline --config $Config }
function Run-Diagnose   { & $Python -m baseline.analysis.diagnose --config $Config }
function Run-Visualize  { & $Python -m baseline.analysis.visualize --config $Config }

switch ($Stage) {
    "train"     { Run-Train }
    "test"      { Run-Test }
    "diagnose"  { Run-Diagnose }
    "visualize" { Run-Visualize }
    "all"       { Run-Train; Run-Test; Run-Diagnose; Run-Visualize }
    default     { Write-Host "未知 Stage: $Stage" }
}
