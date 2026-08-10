# ============================================================================
# 光伏功率预测 - 完整实验流程一键运行脚本
# 
# 流程: 样本准备 → 基线模型 → 残差预测 → 参数搜索 → 评价 → 损失函数实验 → 报告
#
# 使用方法:
#   .\run_full_pipeline.ps1 -Horizon all           (运行所有 horizon 完整流程)
#   .\run_full_pipeline.ps1 -Horizon 1             (仅 H1)
#   .\run_full_pipeline.ps1 -Horizon 4 -Step residual  (仅 H4 残差训练)
#   .\run_full_pipeline.ps1 -Horizon all -SkipBaselines  (跳过基线)
# ============================================================================

param(
    [Parameter(Position=0)]
    [ValidateSet(1, 4, 16, "all")]
    [string]$Horizon = $null,
    
    [ValidateSet("all", "prepare", "baselines", "residual", "evaluation", "hybrid", "improved_loss", "physics", "report")]
    [string[]]$Step = @("all"),
    
    [string]$HybridStrategy = "S6",
    
    [switch]$SkipBaselines,
    [switch]$SkipPhysics,
    [switch]$SkipReport,
    [switch]$Verbose,
    [switch]$ForceRerun,
    [switch]$NoHybridRetrain
)

$ErrorActionPreference = "Continue"

# 获取脚本所在目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item "$ScriptDir\..\..\..").FullName

# ============================================================================
# 改进损失函数实验配置
# ============================================================================
$LossExperiments = @(
    # Huber损失 (所有 horizon 启用物理约束)
    @{
        Name = "Huber损失"
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
    # CombinedV2 (H1 启用物理约束)
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
        Horizons = @(1)
    },
    # CombinedV2 (H4/H16 禁用物理约束)
    @{
        Name = "CombinedV2损失 (H4/H16)"
        Loss = "combined_v2"
        Alpha = 0.5
        Delta = 0.1
        PeakWeight = 0.0
        NightWeight = 0.0
        Smoothness = 0.05
        Sunset = 0.1
        Physics = $false
        Horizons = @(4, 16)
    },
    # Quantile Weighted (所有 horizon)
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
    Write-Host ""
    Write-Host "[STEP] $Text" -ForegroundColor Green -BackgroundColor Black
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
    Write-Err "未找到 Python"
    exit 1
}

function Get-SampleDir {
    param([int]$H)
    return Join-Path $ProjectRoot "data\prediction\step5_new_experiments\samples\h$H"
}

function Test-SamplesExist {
    param([int]$H)
    $trainFile = Join-Path (Get-SampleDir -H $H) "X_train_seq.npy"
    return (Test-Path $trainFile)
}

function Should-Run-Step {
    param([string]$StepName, [string[]]$RequestedSteps)
    if ("all" -in $RequestedSteps) { return $true }
    return $StepName -in $RequestedSteps
}

# ============================================================================
# 步骤执行函数
# ============================================================================

function Step-Prepare-Samples {
    param([int]$H, [string]$Python)
    
    if (-not $ForceRerun -and (Test-SamplesExist -H $H)) {
        Write-Info "样本已存在，跳过: H$H"
        return $true
    }
    
    Write-Step "Step 1: 特征工程 + 样本构造 (H$H)"
    $script = Join-Path $ScriptDir "run_exp_p05_prepare_samples.py"
    
    & $Python $script --horizon $H
    if ($LASTEXITCODE -ne 0) {
        Write-Err "样本准备失败 (H$H)"
        return $false
    }
    Write-Info "样本准备完成"
    return $true
}

function Step-Baselines {
    param([int]$H, [string]$Python)
    
    Write-Step "Step 2: 强基线模型评估 (H$H)"
    $script = Join-Path $ScriptDir "run_exp_p05_baselines.py"
    
    & $Python $script --horizon $H
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "基线评估失败 (H$H)，继续..."
    } else {
        Write-Info "基线评估完成"
    }
    return $true
}

function Step-Residual-Training {
    param([int]$H, [string]$Python)
    
    Write-Step "Step 3: 残差预测模型训练 (H$H)"
    $script = Join-Path $ScriptDir "run_exp_p05_residual_train.py"
    
    & $Python $script --horizon $H
    if ($LASTEXITCODE -ne 0) {
        Write-Err "残差训练失败 (H$H)"
        return $false
    }
    Write-Info "残差训练完成"
    return $true
}

function Step-Evaluation {
    param([int]$H, [string]$Python)
    
    Write-Step "Step 4+5: 分段评价 + 推理计时 (H$H)"
    $script = Join-Path $ScriptDir "run_exp_p05_evaluation.py"
    
    & $Python $script --horizon $H
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "评价失败 (H$H)，继续..."
    } else {
        Write-Info "分段评价完成"
    }
    return $true
}

function Step-Hybrid-Search {
    param([int]$H, [string]$Python, [string]$Strategy)
    
    Write-Step "Step 6: Optuna-AFSA 混合搜索 + 最优参数训练 (H$H, Strategy=$Strategy)"
    $script = Join-Path $ScriptDir "run_exp_p05_hybrid_search.py"
    
    $args = @(
        $script,
        "--horizon", $H,
        "--strategy", $Strategy
    )
    
    if ($NoHybridRetrain) {
        $args += "--no-retrain"
        Write-Info "将跳过重新训练步骤（仅搜索参数）"
    }
    
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "混合搜索失败 (H$H)，继续..."
    } else {
        Write-Info "混合搜索完成，最优模型已保存"
    }
    return $true
}

function Step-Improved-Loss {
    param([int]$H, [string]$Python, [hashtable]$Exp)
    
    $script = Join-Path $ScriptDir "run_exp_p07_improved_loss.py"
    
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
    
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "损失函数实验失败: $($Exp.Name) (H$H)"
        return $false
    }
    Write-Info "损失函数实验完成: $($Exp.Name)"
    return $true
}

function Step-Physics-Features {
    param([int]$H, [string]$Python)
    
    if ($SkipPhysics) {
        Write-Info "跳过物理约束特征实验"
        return $true
    }
    
    Write-Step "Step 7: 物理约束特征实验 (H$H)"
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

function Step-Report {
    param([int]$H, [string]$Python)
    
    if ($SkipReport) {
        Write-Info "跳过报告生成"
        return $true
    }
    
    Write-Step "Step 8: 生成实验报告 (H$H)"
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
    Write-Info "报告生成完成"
    return $true
}

# ============================================================================
# 主程序
# ============================================================================

Write-Banner "光伏功率预测 - 完整实验流程"

# 交互模式
if ([string]::IsNullOrEmpty($Horizon)) {
    Write-Host "请选择要运行的 Horizon:" -ForegroundColor Yellow
    Write-Host "  1   - Horizon 1 (15min 预测)" -ForegroundColor White
    Write-Host "  4   - Horizon 4 (1h 预测)" -ForegroundColor White
    Write-Host "  16  - Horizon 16 (4h 预测)" -ForegroundColor White
    Write-Host "  all - 运行所有 Horizon" -ForegroundColor White
    Write-Host ""
    
    $choice = Read-Host "请输入选择 (1/4/16/all)"
    $Horizon = $choice
}

$Python = Test-Python

# 确定 horizons
$Horizons = @()
if ($Horizon -eq "all") {
    $Horizons = @(1, 4, 16)
} else {
    $Horizons = @([int]$Horizon)
}

Write-Host ""
Write-Host "将运行 Horizon: $($Horizons -join ', ')" -ForegroundColor Green
Write-Host "执行步骤: $($Step -join ' -> ')" -ForegroundColor Green
if ($HybridStrategy -ne "S6") {
    Write-Host "混合搜索策略: $HybridStrategy" -ForegroundColor Yellow
}
Write-Host ""

# 显示完整流程
Write-Host "完整实验流程:" -ForegroundColor Yellow
Write-Host "  1. prepare      - 特征工程 + 样本构造" -ForegroundColor White
Write-Host "  2. baselines    - 强基线模型评估" -ForegroundColor White
Write-Host "  3. residual     - 残差预测模型训练" -ForegroundColor White
Write-Host "  4. evaluation   - 分段评价 + 推理计时" -ForegroundColor White
Write-Host "  5. hybrid       - Optuna-AFSA 混合搜索" -ForegroundColor White
Write-Host "  6. improved_loss - 改进损失函数实验" -ForegroundColor White
Write-Host "  7. physics      - 物理约束特征实验" -ForegroundColor White
Write-Host "  8. report       - 生成实验报告" -ForegroundColor White
Write-Host ""

$confirm = Read-Host "是否开始运行? (Y/n)"
if ($confirm -eq "n" -or $confirm -eq "N") {
    Write-Host "已取消"
    exit 0
}

# ============================================================================
# 执行流程
# ============================================================================

$TotalStart = Get-Date

foreach ($H in $Horizons) {
    $HStart = Get-Date
    
    Write-Banner "========== Horizon $H 实验 =========="
    
    # ----- Step 1: 样本准备 -----
    if (Should-Run-Step "prepare" $Step) {
        if (-not (Step-Prepare-Samples -H $H -Python $Python)) {
            Write-Err "样本准备失败，退出"
            exit 1
        }
    }
    
    # ----- Step 2: 基线模型 -----
    if ((Should-Run-Step "baselines" $Step) -and -not $SkipBaselines) {
        Step-Baselines -H $H -Python $Python
    }
    
    # ----- Step 3: 残差预测 -----
    if (Should-Run-Step "residual" $Step) {
        if (-not (Step-Residual-Training -H $H -Python $Python)) {
            Write-Err "残差训练失败，退出"
            exit 1
        }
    }
    
    # ----- Step 4+5: 分段评价 + 推理计时 -----
    if (Should-Run-Step "evaluation" $Step) {
        Step-Evaluation -H $H -Python $Python
    }
    
    # ----- Step 6: 混合搜索 -----
    if (Should-Run-Step "hybrid" $Step) {
        Step-Hybrid-Search -H $H -Python $Python -Strategy $HybridStrategy
    }
    
    # ----- Step 7: 改进损失函数 -----
    if (Should-Run-Step "improved_loss" $Step) {
        Write-Banner "改进损失函数实验 (H$H)"
        $lossNum = 1
        foreach ($exp in $LossExperiments) {
            if ($exp.Horizons -and $H -notin $exp.Horizons) {
                Write-Info "跳过 $($exp.Name)，不适用于 H$H"
                continue
            }
            Write-Host ""
            Write-Host "[$lossNum / $($LossExperiments.Count)] $($exp.Name)" -ForegroundColor Magenta
            Write-Host "---------------------------------------------------------------"
            Step-Improved-Loss -H $H -Python $Python -Exp $exp
            $lossNum++
            Start-Sleep -Milliseconds 500
        }
    }
    
    # ----- Step 8: 物理约束特征 (仅 H1) -----
    if ((Should-Run-Step "physics" $Step) -and $H -eq 1) {
        Step-Physics-Features -H $H -Python $Python
    }
    
    # ----- Step 9: 生成报告 -----
    if (Should-Run-Step "report" $Step) {
        Step-Report -H $H -Python $Python
    }
    
    $HEnd = Get-Date
    $HDuration = $HEnd - $HStart
    
    Write-Banner "Horizon $H 完成"
    Write-Host "  耗时: $([math]::Round($HDuration.TotalMinutes, 1)) 分钟" -ForegroundColor Cyan
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
Write-Host "输出目录:" -ForegroundColor Yellow
Write-Host "  data\prediction\step5_new_experiments\" -ForegroundColor White
Write-Host "  ├── samples\h[H]\         - 样本数据" -ForegroundColor Gray
Write-Host "  ├── models\h[H]\          - 模型权重" -ForegroundColor Gray
Write-Host "  ├── metrics\h[H]\        - 指标结果" -ForegroundColor Gray
Write-Host "  ├── predictions\h[H]\    - 预测结果" -ForegroundColor Gray
Write-Host "  └── reports\              - 实验报告" -ForegroundColor Gray
Write-Host ""
Write-Host "日志目录:" -ForegroundColor Yellow
Write-Host "  logs\prediction\step5_new_experiments\" -ForegroundColor White
Write-Host ""
