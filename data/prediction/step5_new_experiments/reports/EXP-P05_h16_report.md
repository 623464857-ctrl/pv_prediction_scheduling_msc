# EXP-P05 实验报告 (h16)

## 主实验表（Daytime RMSE 优先）

                Model Horizon RMSE ↓  MAE ↓  MAPE ↓    R² ↑ nRMSE ↓ Params Inference (ms/sample)
              XGBoost      4h 0.0843 0.0382  40.36%  0.9056  0.0843      -                     -
             LightGBM      4h 0.0856 0.0396  41.74%  0.9026  0.0856      -                     -
     Ridge Regression      4h 0.1041 0.0662  77.15%  0.8560  0.1041      -                     -
  PatchTST (Residual)      4h 0.1158 0.0710  42.82%  0.8421  0.1158 108688                 0.009
      LSTM (Residual)      4h 0.1159 0.0706  41.35%  0.8420  0.1159  58128                 0.003
  CNN-LSTM (Residual)      4h 0.1166 0.0744  45.48%  0.8401  0.1166  62096                 0.004
    BiLSTM (Residual)      4h 0.1183 0.0726  43.42%  0.8355  0.1183 149008                 0.009
CNN-BiLSTM (Residual)      4h 0.1215 0.0760  49.34%  0.8264  0.1215 154256                 0.010
          Persistence      4h 0.2266 0.1224 132.23%  0.3172  0.2266      -                     -
       Moving Average      4h 0.8941 0.8146 616.46% -9.6305  0.8941      -                     -

## 残差预测对比

     Model Horizon RMSE ↓ (residual) MAE ↓ (residual)
      LSTM      4h            0.0818           0.0361
    BiLSTM      4h            0.0834           0.0376
  CNN-LSTM      4h            0.0822           0.0380
CNN-BiLSTM      4h            0.0857           0.0395
  PatchTST      4h            0.0818           0.0379

> 排名标准：RMSE（主要）> MAE（次要）

## 可视化

![h16_metrics_comparison](data/prediction/step5_new_experiments/figures/h16/h16_metrics_comparison.png)

![h16_prediction_overlay](data/prediction/step5_new_experiments/figures/h16/h16_prediction_overlay.png)

![h16_residual_comparison](data/prediction/step5_new_experiments/figures/h16/h16_residual_comparison.png)

![h16_inference_benchmark](data/prediction/step5_new_experiments/figures/h16/h16_inference_benchmark.png)

![h16_loss_overview](data/prediction/step5_new_experiments/figures/h16/h16_loss_overview.png)

## 跨 Horizon 综合对比

![comparison_summary](data/prediction/step5_new_experiments/figures/comparison_summary.png)

![comparison_best_model](data/prediction/step5_new_experiments/figures/comparison_best_model.png)

![inference_benchmark_all_horizons](data/prediction/step5_new_experiments/figures/inference_benchmark_all_horizons.png)
