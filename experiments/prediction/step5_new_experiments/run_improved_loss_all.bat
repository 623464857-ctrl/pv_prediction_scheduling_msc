@echo off
chcp 65001 > nul
REM ============================================================================
REM 光伏功率预测 - 改进损失函数实验一键运行脚本
REM 
REM 损失函数配置:
REM   - Huber损失 (δ=0.1): 所有 horizon 启用物理约束
REM   - CombinedV2: H1 启用物理约束，H4/H16 禁用物理约束后处理
REM   - Quantile Weighted: 所有 horizon 启用物理约束
REM
REM 使用方法:
REM   run_improved_loss_all.bat 1    (运行 H1 实验)
REM   run_improved_loss_all.bat 4    (运行 H4 实验)
REM   run_improved_loss_all.bat 16   (运行 H16 实验)
REM   run_improved_loss_all.bat all  (运行所有 horizon 实验)
REM ============================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"

REM 解析参数
set "HORIZON=%~1"
if "%HORIZON%"=="" (
    echo [ERROR] 请指定 horizon 参数: 1, 4, 16 或 all
    echo 使用方法: run_improved_loss_all.bat [horizon^|all]
    exit /b 1
)

set "PYTHON=python"
where python >nul 2>&1
if errorlevel 1 (
    set "PYTHON=python3"
    where python3 >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] 未找到 Python，请确保 Python 已安装并添加到 PATH
        exit /b 1
    )
)

echo ============================================================================
echo 光伏功率预测 - 改进损失函数实验
echo ============================================================================
echo.

REM 设置需要运行的 horizon 列表
set "HORIZONS="
if /i "%HORIZON%"=="all" (
    set "HORIZONS=1 4 16"
) else (
    set "HORIZONS=%HORIZON%"
)

REM ============================================================================
REM 主循环 - 遍历每个 horizon
REM ============================================================================

for %%H in (%HORIZONS%) do (
    echo.
    echo ============================================================================
    echo  Horizon %%H (%%H = 15min/1h/4h)
    echo ============================================================================
    echo.

    set "H=%%H"
    
    REM ============================================================================
    REM 步骤1: 检查并准备样本数据
    REM ============================================================================
    set "SAMPLE_DIR=%SCRIPT_DIR%..\..\..\data\prediction\step5_new_experiments\samples\h%H%"
    if not exist "%SAMPLE_DIR%\X_train_seq.npy" (
        echo [Step 1] 准备 %%H 样本数据...
        %PYTHON% "%SCRIPT_DIR%run_exp_p05_prepare_samples.py" --horizon %H%
        if errorlevel 1 (
            echo [ERROR] 样本准备失败，退出
            exit /b 1
        )
        echo [OK] 样本准备完成
    ) else (
        echo [Skip] 样本数据已存在
    )
    echo.

    REM ============================================================================
    REM 步骤2: 运行改进损失函数实验
    REM ============================================================================
    echo [Step 2] 运行改进损失函数实验...
    echo.

    REM === Huber损失 (所有 horizon 启用物理约束) ===
    echo ---------------------------------------------------------------
    echo Huber损失 (δ=0.1) - 所有 horizon - 启用物理约束
    echo ---------------------------------------------------------------
    %PYTHON% "%SCRIPT_DIR%run_exp_p07_improved_loss.py" ^
        --horizon %H% ^
        --loss huber ^
        --alpha 0.5 ^
        --delta 0.1 ^
        --peak-weight 2.0 ^
        --night-weight 3.0 ^
        --huber-delta 0.1
    if errorlevel 1 (
        echo [WARNING] Huber损失实验失败 (H%H%)
    ) else (
        echo [OK] Huber损失完成 (H%H%)
    )
    echo.

    REM === CombinedV2 (H1 启用物理约束, H4/H16 禁用) ===
    if "%H%"=="1" (
        echo ---------------------------------------------------------------
        echo CombinedV2损失 - H1 - 启用物理约束
        echo ---------------------------------------------------------------
        %PYTHON% "%SCRIPT_DIR%run_exp_p07_improved_loss.py" ^
            --horizon %H% ^
            --loss combined_v2 ^
            --alpha 0.5 ^
            --delta 0.1 ^
            --peak-weight 0.0 ^
            --night-weight 0.0 ^
            --huber-delta 0.1 ^
            --smoothness-weight 0.05 ^
            --sunset-weight 0.1
        if errorlevel 1 (
            echo [WARNING] CombinedV2实验失败 (H%H%)
        ) else (
            echo [OK] CombinedV2完成 (H%H%)
        )
        echo.
    ) else (
        echo ---------------------------------------------------------------
        echo CombinedV2损失 - H%H% - 禁用物理约束后处理
        echo ---------------------------------------------------------------
        %PYTHON% "%SCRIPT_DIR%run_exp_p07_improved_loss.py" ^
            --horizon %H% ^
            --loss combined_v2 ^
            --alpha 0.5 ^
            --delta 0.1 ^
            --peak-weight 0.0 ^
            --night-weight 0.0 ^
            --huber-delta 0.1 ^
            --smoothness-weight 0.05 ^
            --sunset-weight 0.1 ^
            --no-physics
        if errorlevel 1 (
            echo [WARNING] CombinedV2实验失败 (H%H%)
        ) else (
            echo [OK] CombinedV2完成 (H%H%)
        )
        echo.
    )

    REM === Quantile Weighted损失 (所有 horizon 启用物理约束) ===
    echo ---------------------------------------------------------------
    echo 分位数加权损失 - 所有 horizon - 启用物理约束
    echo ---------------------------------------------------------------
    %PYTHON% "%SCRIPT_DIR%run_exp_p07_improved_loss.py" ^
        --horizon %H% ^
        --loss quantile_weighted ^
        --alpha 0.5 ^
        --delta 1.0 ^
        --peak-weight 2.0 ^
        --night-weight 3.0
    if errorlevel 1 (
        echo [WARNING] Quantile Weighted损失实验失败 (H%H%)
    ) else (
        echo [OK] Quantile Weighted损失完成 (H%H%)
    )
    echo.

    REM ============================================================================
    REM 步骤3: 运行物理约束特征实验 (H1)
    REM ============================================================================
    if "%H%"=="1" (
        echo ---------------------------------------------------------------
        echo 物理约束特征实验 - H1
        echo ---------------------------------------------------------------
        %PYTHON% "%SCRIPT_DIR%run_exp_p08_physics.py" --horizon %H%
        if errorlevel 1 (
            echo [WARNING] 物理约束特征实验失败
        ) else (
            echo [OK] 物理约束特征实验完成
        )
        echo.
    )

    REM ============================================================================
    REM 步骤4: 生成报告
    REM ============================================================================
    echo ---------------------------------------------------------------
    echo 生成 Horizon %H% 报告
    echo ---------------------------------------------------------------
    %PYTHON% "%SCRIPT_DIR%run_exp_p05_report.py" --horizon %H%
    if errorlevel 1 (
        echo [WARNING] 报告生成失败 (H%H%)
    ) else (
        echo [OK] 报告生成完成 (H%H%)
    )
    echo.

    echo ============================================================================
    echo  Horizon %H% 实验完成
    echo ============================================================================
    echo.
)

REM ============================================================================
REM 完成
REM ============================================================================
echo ============================================================================
echo 所有实验完成!
echo ============================================================================
echo.
echo 结果文件位置:
echo   - 指标: data\prediction\step5_new_experiments\metrics\h[1,4,16]\improved_loss_*.json
echo   - 预测: data\prediction\step5_new_experiments\predictions\h[1,4,16]\*_improved_test.csv
echo   - 报告: data\prediction\step5_new_experiments\reports\EXP-P05_h[1,4,16]_*.md
echo.
echo 日志文件位置:
echo   - logs\prediction\step5_new_experiments\EXP-P07_h[1,4,16]_*.log
echo.

endlocal
