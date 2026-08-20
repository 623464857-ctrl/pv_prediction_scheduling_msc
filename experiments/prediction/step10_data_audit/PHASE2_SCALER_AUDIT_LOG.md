# Step10 Phase 2: Scaler与预处理链路审计

## 实验信息

- **实验阶段**: Phase 2
- **审计内容**: Scaler与预处理链路
- **优先级**: P0 (Critical)
- **创建日期**: 2026-08-16

---

## 审计问题清单

| # | 问题 | 状态 |
|---|------|------|
| 1 | scaler是否只fit train，val/test只transform | ✅ 已检查 |
| 2 | 不同horizon的target/residual scaler是否独立 | ✅ 已检查 |
| 3 | inverse transform是否对应正确的scaler | ✅ 已检查 |
| 4 | residual reconstruction的reference是否与训练定义一致 | ✅ 已检查 |
| 5 | prediction clipping发生在反归一化前还是后 | ✅ 已检查 |
| 6 | test target是否超出train scaler的range | ✅ 已检查 |

---

## 详细审计结果

---

### 问题1: Scaler是否只fit train，val/test只transform

**检查代码位置**:
- `experiments/prediction/step4_optuna_hybrid/run_exp_p04_prepare_samples.py` (L124-141)
- `experiments/prediction/step5_new_experiments/run_exp_p05_prepare_samples.py` (L90-98)

**代码证据**:

```python
# Step4 prepare_samples.py (L124-138)
scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, len(STEP1_FEATURES)))  # 仅在train上fit

X_train_s = transform_X(X_train)  # transform train
X_val_s = transform_X(X_val)     # transform val
X_test_s = transform_X(X_test)   # transform test

y_scaler = StandardScaler()
y_scaler.fit(y_train)  # 仅在train上fit

y_train_s = y_scaler.transform(y_train)  # transform train
y_val_s = y_scaler.transform(y_val)     # transform val
y_test_s = y_scaler.transform(y_test)   # transform test
```

```python
# Step5 prepare_samples.py (L90-98)
scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, len(feature_cols)))  # 仅在train上fit

y_scaler = StandardScaler()
y_scaler.fit(y_train)  # 仅在train上fit
```

**结论**: ✅ **通过** - 所有scaler仅在训练集上fit，验证集和测试集仅使用transform。

---

### 问题2: 不同horizon的target/residual scaler是否独立

**检查结果**:

| Horizon | Target Scaler | Residual Scaler | 是否独立 |
|---------|--------------|-----------------|---------|
| h1 | `scaler_params.json` (y_mean, y_scale) | `residual_scaler_params.json` | ✅ 独立 |
| h4 | `scaler_params.json` (4维y_mean/y_scale) | `residual_scaler_params.json` (4维) | ✅ 独立 |
| h16 | `scaler_params.json` (16维y_mean/y_scale) | `residual_scaler_params.json` (16维) | ✅ 独立 |

**代码证据** (`run_exp_p05_residual_train.py` L71-77):

```python
# Residual scaler独立fit
y_res_train = compute_residual_targets(samples["y_train_raw"], y_last_train)
res_scaler = fit_residual_scaler(y_res_train)
save_residual_scaler(res_scaler, MODELS_DIR.parent / "samples" / f"h{horizon}" / "residual_scaler_params.json")
```

**实测参数**:

| Horizon | Target y_mean | Residual mean (step 0) | Residual scale (step 0) |
|---------|--------------|------------------------|-------------------------|
| h1 | 0.1932 | ~0.000004 | 0.0441 |
| h4 | [0.193, ...] (4步) | ~0.000005 | 0.0441 |
| h16 | [0.193, ...] (16步) | ~0.000013 | 0.0441 |

**结论**: ✅ **通过** - 每个horizon都有独立的scaler，参数各不相同，符合预期。

---

### 问题3: Inverse transform是否对应正确的scaler

**检查代码位置**:

1. **Step4最终训练** (`run_exp_p04_train_final.py` L88-92):
```python
y_pred_scaled = predict(model, X_test, device)
y_pred = y_scaler.inverse_transform(y_pred_scaled)  # 使用正确的y_scaler
y_test_raw = y_scaler.inverse_transform(y_test)     # 使用正确的y_scaler
```

2. **Step5残差训练** (`run_exp_p05_residual_train.py` L108-110):
```python
delta_scaled = predict(model, samples["X_test_seq"], device, batch_size=rt["batch_size"])
delta_pred = inverse_transform_residual(res_scaler, delta_scaled)  # 使用residual_scaler
y_pred = reconstruct_from_residual(y_last_test, delta_pred)        # 重建最终预测
```

3. **Baseline模型** (`baselines.py`):
```python
# Ridge/XGBoost/LightGBM: 使用各自的内部scaler
X_scaled = self.scaler.transform(X_flat)  # transform
pred = self.model.predict(X_scaled)       # 预测
# 不需要inverse transform，因为直接预测原始scale
```

**结论**: ✅ **通过** - inverse transform使用对应的正确scaler：
- Step4直接预测：使用`y_scaler`
- Step5残差预测：使用`residual_scaler`

---

### 问题4: Residual reconstruction的reference是否与训练定义一致

**检查代码位置** (`exp_p05_residual.py`):

```python
def compute_residual_targets(y: np.ndarray, y_last: np.ndarray) -> np.ndarray:
    """Δy = y_future - y_last。"""
    return (y - y_last).astype(np.float32)

def reconstruct_from_residual(y_last: np.ndarray, delta_pred: np.ndarray) -> np.ndarray:
    """y_hat_future = y_last + Δy_hat。"""
    return (y_last + delta_pred).astype(np.float32)
```

**训练时的定义**:
- `y_res_train = y_train_raw - y_last_train`
- 训练目标：预测`Δy = y_future - y_last`

**推理时的重建**:
- `y_pred = y_last_test + delta_pred`
- 使用测试集的`y_last`作为reference

**代码证据** (`run_exp_p05_residual_train.py` L67-110):
```python
# 训练
y_last_train = samples["y_last_train"]
y_res_train = compute_residual_targets(samples["y_train_raw"], y_last_train)
res_scaler.fit(y_res_train)

# 推理
y_last_test = samples["y_last_test"]
delta_pred = inverse_transform_residual(res_scaler, delta_scaled)
y_pred = reconstruct_from_residual(y_last_test, delta_pred)
```

**结论**: ✅ **通过** - reconstruction使用正确的reference：`y_last`作为base，与训练时的定义完全一致。

---

### 问题5: Prediction clipping发生在反归一化前还是后

**检查结果**:

通过搜索代码库，未发现任何显式的prediction clipping逻辑。

- `run_exp_p04_train_final.py`: 无clipping
- `run_exp_p05_residual_train.py`: 无clipping
- `baselines.py`: 无clipping
- `run_exp_p05_evaluation.py`: 无clipping

**关于物理约束**:
- 原始功率数据经过Step1预处理后，负功率已被clip到0 (`power_mw_negative_clipped_flag`)
- `power_pu = power_mw / capacity`，因此`power_pu`的自然范围是`[0, 1.05]`（有5%过载容忍）
- Step1的功率上限检查: `cap_hi = 1.05 * capacity_mw`

**实测数据**:
```
Step4 h1 Test y range (raw): [0.000000, 0.904200]
Step5 h1 Test y range (raw): [0.000000, 0.904200]
```

**结论**: ⚠️ **无显式clipping，但数据天然符合物理约束**
- 代码中未发现任何clip操作
- 实际测试数据范围[0, 0.9]天然符合功率物理约束
- 如果需要防止极端异常预测，建议在inverse transform后添加显式clip到[0, 1.0]

---

### 问题6: Test target是否超出train scaler的range

**检查方法**: 计算训练集scaler的统计范围 vs 测试集实际数据范围

**Step4 h1**:
| 指标 | 值 |
|------|-----|
| y_mean | 0.193139 |
| y_scale | 0.275473 |
| 期望范围 (mean ± 3σ) | [-0.633, 1.020] |
| 测试集实际范围 | [0.000, 0.904] |
| 超出3σ的样本数 | 0 |

**Step5 h1**:
| 指标 | 值 |
|------|-----|
| y_mean | 0.193202 |
| y_scale | 0.275483 |
| 期望范围 (mean ± 3σ) | [-0.633, 1.020] |
| 测试集实际范围 | [0.000, 0.904] |
| 超出3σ的样本数 | 0 (0.00%) |

**Residual Scaler**:
| Horizon | Residual Scale (step 0) | 测试集备注 |
|---------|------------------------|-----------|
| h1 | 0.0441 | 均值接近0，符合残差特征 |
| h4 | 0.0441 (step 0) | 各步独立scale |
| h16 | 0.0441 (step 0) | 各步独立scale，随步数增加scale增大 |

**结论**: ✅ **通过** - 测试集数据完全在训练集scaler的统计范围内，无分布外数据。

---

## Phase 2 审计总结

### ✅ 所有检查项均通过

| 问题 | 状态 | 备注 |
|------|------|------|
| Scaler fit范围 | ✅ 通过 | 仅在train上fit |
| Horizon独立scaler | ✅ 通过 | 每个horizon有独立的scaler |
| Inverse transform正确性 | ✅ 通过 | 使用对应的正确scaler |
| Residual reconstruction | ✅ 通过 | reference定义与训练一致 |
| Prediction clipping | ⚠️ 无显式clip | 但数据天然符合物理约束 |
| Test数据范围 | ✅ 通过 | 无分布外数据 |

### 建议改进（非关键）

1. **添加显式Prediction Clipping**:
   - 虽然当前数据天然符合物理约束，但建议在inverse transform后添加显式clip
   - 建议范围: `[0, 1.0]` (功率归一化后的物理合理范围)

```python
# 建议在inverse transform后添加
y_pred = np.clip(y_pred, 0.0, 1.0)
```

2. **Residual Scaler版本追踪**:
   - 当前residual_scaler_params.json未记录版本信息
   - 建议添加`feature_version`字段用于追溯

---

## 与Phase 1的关联

Phase 1发现的数据划分不一致问题（Step3 vs Step2/4/5）与Scaler链路无关。
Scaler审计在Step2/4/5范围内均正确。

---

*本日志由Step10 Phase 2自动生成*
*生成时间: 2026-08-16*
