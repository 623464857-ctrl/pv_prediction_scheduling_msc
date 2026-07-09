# EXP-P05 实验报告 (h1)

## 主实验表（Daytime RMSE 优先）

                Model Horizon RMSE ↓  MAE ↓  MAPE ↓    R² ↑ nRMSE ↓ Params Inference (ms/sample)
              XGBoost   15min 0.0300 0.0108  10.69%  0.9880  0.0300      -                     -
             LightGBM   15min 0.0300 0.0108  10.75%  0.9881  0.0300      -                     -
     Ridge Regression   15min 0.0336 0.0161  14.87%  0.9850  0.0336      -                     -
          Persistence   15min 0.0408 0.0195  24.35%  0.9779  0.0408      -                     -
      LSTM (Residual)   15min 0.0458 0.0241  10.78%  0.9706  0.0458  57153                 0.004
    BiLSTM (Residual)   15min 0.0460 0.0252  11.31%  0.9703  0.0460 147073                 0.011
CNN-BiLSTM (Residual)   15min 0.0465 0.0253  12.67%  0.9697  0.0465 152321                 0.011
  CNN-LSTM (Residual)   15min 0.0468 0.0259  11.86%  0.9693  0.0468  61121                 0.005
  PatchTST (Residual)   15min 0.0473 0.0263  13.40%  0.9687  0.0473 107713                 0.010
       Moving Average   15min 0.7773 0.7199 393.46% -7.0340  0.7773      -                     -

## 残差预测对比

     Model Horizon RMSE ↓ (residual) MAE ↓ (residual)
      LSTM   15min            0.0295           0.0103
    BiLSTM   15min            0.0297           0.0109
  CNN-LSTM   15min            0.0302           0.0117
CNN-BiLSTM   15min            0.0300           0.0118
  PatchTST   15min            0.0305           0.0120

> 排名标准：RMSE（主要）> MAE（次要）

## 可视化

![h1_metrics_comparison](data/prediction/step5_new_experiments/figures/h1/h1_metrics_comparison.png)

![h1_prediction_overlay](data/prediction/step5_new_experiments/figures/h1/h1_prediction_overlay.png)

![h1_residual_comparison](data/prediction/step5_new_experiments/figures/h1/h1_residual_comparison.png)

![h1_inference_benchmark](data/prediction/step5_new_experiments/figures/h1/h1_inference_benchmark.png)

![h1_loss_overview](data/prediction/step5_new_experiments/figures/h1/h1_loss_overview.png)

## 跨 Horizon 综合对比

![comparison_summary](data/prediction/step5_new_experiments/figures/comparison_summary.png)

![comparison_best_model](data/prediction/step5_new_experiments/figures/comparison_best_model.png)

![residual_comparison_all_horizons](data/prediction/step5_new_experiments/figures/residual_comparison_all_horizons.png)

![inference_benchmark_all_horizons](data/prediction/step5_new_experiments/figures/inference_benchmark_all_horizons.png)
