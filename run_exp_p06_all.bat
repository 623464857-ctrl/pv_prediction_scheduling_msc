@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo  EXP-P06 全流程一键运行脚本
echo  顺序: 混合搜索 (H1 -> H4 -> H16) -> 残差训练 (H1 -> H4 -> H16)
echo ============================================================
echo.

:: ---------------------------------------------------------------
:: 阶段 1：混合搜索
:: ---------------------------------------------------------------
echo [1/6] 开始 H1 混合搜索 (S1-S6 x 5模型) ...
python experiments/prediction/step5_new_experiments/run_exp_p06_hybrid_search.py --horizon 1 --all-strategies
if %ERRORLEVEL% neq 0 (
    echo [ERROR] H1 混合搜索失败，脚本终止。
    pause
    exit /b 1
)
echo [OK] H1 混合搜索完成
echo.

echo [2/6] 开始 H4 混合搜索 (S1-S6 x 5模型) ...
python experiments/prediction/step5_new_experiments/run_exp_p06_hybrid_search.py --horizon 4 --all-strategies
if %ERRORLEVEL% neq 0 (
    echo [ERROR] H4 混合搜索失败，脚本终止。
    pause
    exit /b 1
)
echo [OK] H4 混合搜索完成
echo.

echo [3/6] 开始 H16 混合搜索 (S1-S6 x 5模型) ...
python experiments/prediction/step5_new_experiments/run_exp_p06_hybrid_search.py --horizon 16 --all-strategies
if %ERRORLEVEL% neq 0 (
    echo [ERROR] H16 混合搜索失败，脚本终止。
    pause
    exit /b 1
)
echo [OK] H16 混合搜索完成
echo.

:: ---------------------------------------------------------------
:: 阶段 2：残差训练
:: ---------------------------------------------------------------
echo [4/6] 开始 H1 残差训练 (5模型) ...
python experiments/prediction/step5_new_experiments/run_exp_p06_residual_train.py --horizon 1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] H1 残差训练失败，脚本终止。
    pause
    exit /b 1
)
echo [OK] H1 残差训练完成
echo.

echo [5/6] 开始 H4 残差训练 (5模型) ...
python experiments/prediction/step5_new_experiments/run_exp_p06_residual_train.py --horizon 4
if %ERRORLEVEL% neq 0 (
    echo [ERROR] H4 残差训练失败，脚本终止。
    pause
    exit /b 1
)
echo [OK] H4 残差训练完成
echo.

echo [6/6] 开始 H16 残差训练 (5模型) ...
python experiments/prediction/step5_new_experiments/run_exp_p06_residual_train.py --horizon 16
if %ERRORLEVEL% neq 0 (
    echo [ERROR] H16 残差训练失败，脚本终止。
    pause
    exit /b 1
)
echo [OK] H16 残差训练完成
echo.

echo ============================================================
echo  EXP-P06 全流程完成！
echo ============================================================
pause
