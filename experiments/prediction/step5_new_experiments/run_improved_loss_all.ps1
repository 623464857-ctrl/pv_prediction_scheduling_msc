# ============================================================================
# 光伏功率预测 - 改进损失函数实验一键运行脚本
# 
# 损失函数配置:
#   - Huber损失 (δ=0.1): 所有 horizon 启用物理约束
#   - CombinedV2: H1 启用物理约束，H4/H16 禁用物理约束后处理
#   - Quantile Weighted: 所有 horizon 启用物理约束
#
# 使用方法:
#   .\run_improved_loss_all.ps1 -Horizon 1    (运行 H1 实验)
#   .\run_improved_loss_all.ps1 -Horizon 4    (运行 H4 实验)
#   .\run_improved_loss_all.ps1 -Horizon 16   (运行 H16 实验)
#   .\run_improved_loss_all.ps1 -Horizon all  (运行所有 horizon 实验)
#   .\run_improved_loss_all.ps1               (交互模式)
# ============================================================================

param(
    [Parameter(Position=0)]
    [ValidateSet(1, 4, 16, "all")]
    [string]$Horizon = $null,
    
    [switch]$SkipSamplePrep,
    [switch]$SkipPhysicsExp,
    [switch]$SkipReport,
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"

# 获取脚本所在目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item "$ScriptDir\..\..\..").FullName

# ============================================================================
# 实验配置
# ============================================================================
# 每个实验配置: @{loss, alpha, delta, peak_weight, night_weight, smoothness, sunset, physics}
# physics: $true = 启用, $false = 禁用 (H4/H16 禁用物理约束后处理)

$Experiments = @(
    # 2.7 Huber损失 (delta=0.1, 对异常值鲁棒, 所有 horizon 启用物理约束)
    @{
        Name = "Huber损失 (δ=0.1)"
        Loss = "huber"
        Alpha = 0.5
        Delta = 0.1
        PeakWeight = 2.0
        NightWeight = 3.0
        Smoothness = 0.0
        Sunset = 0.0
        Physics = $true
        Horizons = @(1, 4, 16)
    },
    # 2.5 CombinedV2损失 (无peak_weight, 含日落约束, 仅 H1 启用物理约束)
    @{
        Name = "CombinedV2损失 (H1)"
        Loss = "combined_v2"
        Alpha = 0.5
        Delta = 0.1
        PeakWeight = 0.0
        NightWeight = 0.0
        Smoothness = 0.05
        Sunset = 0.1
        Physics = $true
        Horizons = @(1)  # 仅 H1
    },
    # CombinedV2 + 禁用物理约束后处理 (H4/H16 专用)
    @{
        Name = "CombinedV2损失 (H4/H16 无物理约束)"
        Loss = "combined_v2"
        Alpha = 0.5
        Delta = 0.1
        PeakWeight = 0.0
        NightWeight = 0.0
        Smoothness = 0.05
        Sunset = 0.1
        Physics = $false  # 禁用物理约束后处理
        Horizons = @(4, 16)  # 仅 H4/H16
    },
    # 2.8 Quantile Weighted损失 (所有 horizon 启用物理约束)
    @{
        Name = "分位数加权损失"
        Loss = "quantile_weighted"
        Alpha = 0.5
        Delta = 1.0
        PeakWeight = 2.0
        NightWeight = 3.0
        Smoothness = 0.0
        Sunset = 0.0
        Physics = $true
        Horizons = @(1, 4, 16)
    }
)

# ============================================================================
# 函数定义
# ============================================================================

function Write-Banner {
    param([string]$Text)
    $line = "=" * 70
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host " $Text" -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
}

function Write-Step {
    param([string]$Text)
    Write-Host "[Step] $Text" -ForegroundColor Green
}

function Write-Info {
    param([string]$Text)
    if ($Verbose) {
        Write-Host "[INFO] $Text" -ForegroundColor Gray
    }
}

function Write-Warn {
    param([string]$Text)
    Write-Host "[WARN] $Text" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Text)
    Write-Host "[ERROR] $Text" -ForegroundColor Red
}

function Test-Python {
    try {
        $result = & python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Info "Python: $result"
            return "python"
        }
    } catch {}
    
    try {
        $result = & python3 --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Info "Python: $result"
            return "python3"
        }
    } catch {}
    
    Write-Err "未找到 Python，请确保 Python 已安装并添加到 PATH"
    exit 1
}

function Get-SampleDir {
    param([int]$H)
    return Join-Path $ProjectRoot "data\prediction\step5_new_experiments\samples\h$H"
}

function Test-SamplesExist {
    param([int]$H)
    $sampleDir = Get-SampleDir -H $H
    $trainFile = Join-Path $sampleDir "X_train_seq.npy"
    return (Test-Path $trainFile)
}

function Run-SamplePreparation {
    param([int]$H, [string]$Python)
    
    Write-Step "准备 Horizon $H 样本数据..."
    $script = Join-Path $ScriptDir "run_exp_p05_prepare_samples.py"
    
    if (-not (Test-Path $script)) {
        Write-Err "样本准备脚本不存在: $script"
        return $false
    }
    
    & $Python $script --horizon $H
    if ($LASTEXITCODE -ne 0) {
        Write-Err "样本准备失败 (H$H)"
        return $false
    }
    
    Write-Info "样本准备完成 (H$H)"
    return $true
}

function Run-ImprovedLossExperiment {
    param(
        [int]$H,
        [hashtable]$Exp,
        [string]$Python
    )
    
    $script = Join-Path $ScriptDir "run_exp_p07_improved_loss.py"
    
    if (-not (Test-Path $script)) {
        Write-Err "实验脚本不存在: $script"
        return $false
    }
    
    # 构建参数
    $args = @(
        $script,
        "--horizon", $H,
        "--loss", $Exp.Loss,
        "--alpha", $Exp.Alpha,
        "--delta", $Exp.Delta,
        "--peak-weight", $Exp.PeakWeight,
        "--night-weight", $Exp.NightWeight,
        "--huber-delta", $Exp.Delta,
        "--smoothness-weight", $Exp.Smoothness,
        "--sunset-weight", $Exp.Sunset
    )
    
    if (-not $Exp.Physics) {
        $args += "--no-physics"
    }
    
    Write-Host "  损失函数: $($Exp.Loss)" -ForegroundColor Gray
    Write-Host "  Alpha: $($Exp.Alpha), Delta: $($Exp.Delta)" -ForegroundColor Gray
    Write-Host "  峰值权重: $($Exp.PeakWeight), 夜间权重: $($Exp.NightWeight)" -ForegroundColor Gray
    Write-Host "  平滑权重: $($Exp.Smoothness), 日落权重: $($Exp.Sunset)" -ForegroundColor Gray
    Write-Host "  物理约束: $($Exp.Physics)" -ForegroundColor Gray
    Write-Host ""
    
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "实验失败: $($Exp.Name) (H$H)"
        return $false
    }
    
    return $true
}

function Run-PhysicsFeatureExperiment {
    param([int]$H, [string]$Python)
    
    if ($SkipPhysicsExp) {
        Write-Info "跳过物理约束特征实验"
        return $true
    }
    
    if ($H -ne 1) {
        Write-Info "物理约束特征实验仅在 H1 上运行"
        return $true
    }
    
    Write-Step "运行物理约束特征实验 (H1)..."
    $script = Join-Path $ScriptDir "run_exp_p08_physics.py"
    
    if (-not (Test-Path $script)) {
        Write-Warn "物理约束特征脚本不存在，跳过"
        return $true
    }
    
    & $Python $script --horizon $H
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "物理约束特征实验失败 (H$H)"
        return $false
    }
    
    Write-Info "物理约束特征实验完成"
    return $true
}

function Run-ReportGeneration {
    param([int]$H, [string]$Python)
    
    if ($SkipReport) {
        Write-Info "跳过报告生成"
        return $true
    }
    
    Write-Step "生成 Horizon $H 实验报告..."
    $script = Join-Path $ScriptDir "run_exp_p05_report.py"
    
    if (-not (Test-Path $script)) {
        Write-Warn "报告脚本不存在，跳过"
        return $true
    }
    
    & $Python $script --horizon $H
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "报告生成失败 (H$H)"
        return $false
    }
    
    Write-Info "报告生成完成 (H$H)"
    return $true
}

# ============================================================================
# 主程序
# ============================================================================

Write-Banner "光伏功率预测 - 改进损失函数实验"

# 交互模式选择
if ([string]::IsNullOrEmpty($Horizon)) {
    Write-Host "请选择要运行的 Horizon:" -ForegroundColor Yellow
    Write-Host "  1 - Horizon 1 (15min 预测)" -ForegroundColor White
    Write-Host "  4 - Horizon 4 (1h 预测)" -ForegroundColor White
    Write-Host " 16 - Horizon 16 (4h 预测)" -ForegroundColor White
    Write-Host " all - 运行所有 Horizon" -ForegroundColor White
    Write-Host ""
    
    $choice = Read-Host "请输入选择 (1/4/16/all)"
    $Horizon = $choice
}

# 检查 Python
$Python = Test-Python

# 确定要运行的 horizons
$Horizons = @()
if ($Horizon -eq "all") {
    $Horizons = @(1, 4, 16)
} else {
    $Horizons = @([int]$Horizon)
}

Write-Host ""
Write-Host "将运行 Horizon: $($Horizons -join ', ')" -ForegroundColor Green
Write-Host ""

# 显示实验列表
Write-Host "实验配置列表:" -ForegroundColor Yellow
for ($i = 0; $i -lt $Experiments.Count; $i++) {
    $exp = $Experiments[$i]
    Write-Host "  [$($i+1)] $($exp.Name) - $($exp.Loss)" -ForegroundColor White
}
Write-Host ""

$confirm = Read-Host "是否开始运行? (Y/n)"
if ($confirm -eq "n" -or $confirm -eq "N") {
    Write-Host "已取消"
    exit 0
}

# ============================================================================
# 执行实验
# ============================================================================

$TotalStart = Get-Date

foreach ($H in $Horizons) {
    $HStart = Get-Date
    
    Write-Banner "Horizon $H 实验"
    
    # 检查样本数据
    if (-not $SkipSamplePrep -and -not (Test-SamplesExist -H $H)) {
        Write-Step "样本数据不存在，开始准备..."
        if (-not (Run-SamplePreparation -H $H -Python $Python)) {
            Write-Err "样本准备失败，退出"
            exit 1
        }
    } else {
        Write-Info "样本数据已存在，跳过准备"
    }
    
    Write-Host ""
    
    # 运行改进损失函数实验
    Write-Banner "改进损失函数实验 (H$H)"
    
    $expNum = 1
    foreach ($exp in $Experiments) {
        # 检查该实验是否适用于当前 horizon
        if ($exp.Horizons -and $H -notin $exp.Horizons) {
            Write-Info "跳过 $($exp.Name)，不适用于 H$H"
            continue
        }
        
        Write-Host ""
        Write-Host "[$expNum / $($Experiments.Count)] $($exp.Name)" -ForegroundColor Magenta
        Write-Host "---------------------------------------------------------------"
        
        $success = Run-ImprovedLossExperiment -H $H -Exp $exp -Python $Python
        $expNum++
        
        if ($success) {
            Write-Info "实验完成: $($exp.Name)"
        }
        
        Start-Sleep -Milliseconds 500  # 避免太快
    }
    
    Write-Host ""
    
    # 运行物理约束特征实验 (仅 H1)
    if (-not $SkipPhysicsExp -and $H -eq 1) {
        Write-Host ""
        Run-PhysicsFeatureExperiment -H $H -Python $Python
    }
    
    # 生成报告
    if (-not $SkipReport) {
        Write-Host ""
        Run-ReportGeneration -H $H -Python $Python
    }
    
    $HEnd = Get-Date
    $HDuration = $HEnd - $HStart
    
    Write-Banner "Horizon $H 完成"
    Write-Host "  耗时: $([math]::Round($HDuration.TotalMinutes, 1)) 分钟" -ForegroundColor Cyan
    Write-Host ""
}

# ============================================================================
# 完成
# ============================================================================

$TotalEnd = Get-Date
$TotalDuration = $TotalEnd - $TotalStart

Write-Banner "所有实验完成!"
Write-Host ""
Write-Host "总耗时: $([math]::Round($TotalDuration.TotalMinutes, 1)) 分钟" -ForegroundColor Green
Write-Host ""
Write-Host "结果文件位置:" -ForegroundColor Yellow
Write-Host "  - 指标: data\prediction\step5_new_experiments\metrics\h[1,4,16]\improved_loss_*.json" -ForegroundColor White
Write-Host "  - 预测: data\prediction\step5_new_experiments\predictions\h[1,4,16]\*_improved_test.csv" -ForegroundColor White
Write-Host "  - 报告: data\prediction\step5_new_experiments\reports\EXP-P05_h[1,4,16]_*.md" -ForegroundColor White
Write-Host ""
Write-Host "日志文件位置:" -ForegroundColor Yellow
Write-Host "  - logs\prediction\step5_new_experiments\EXP-P07_h[1,4,16]_*.log" -ForegroundColor White
Write-Host ""
Write-Host "提示: 如需查看特定损失函数结果，请检查上述目录中的 JSON 文件" -ForegroundColor Gray
Write-Host ""
