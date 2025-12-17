# Task 4.2 - Multi-Mode PV System Neural Network Model

## Task Overview

This task trains a neural network model to predict load voltage and power output for a multi-mode photovoltaic system.

**Dataset**: CodeP4.3F25.ipynb
- **Input parameters**: [M, T_air, I_D, R_L]
  - M: Mode number (0, 1, 2) - not normalized
  - T_air: Air temperature (°C) - normalized
  - I_D: Irradiance (W/m²) - normalized
  - R_L: Load resistance (Ω) - normalized

- **Output parameters**: [V_L, W_dot]
  - V_L: Load voltage (V) - normalized
  - W_dot: Power output (W) - normalized

**Data size**: Total 48 samples
- Training set: 32 samples (66.7%)
- Validation set: 16 samples (33.3%)

---

## (a) Data Normalization

### Normalization Strategy

**Median normalization** method (consistent with Task 4.1):
- Calculate median for each parameter (excluding Mode number M)
- Divide each parameter value by its corresponding median
- **Mode number M remains unchanged** (0, 1, 2)

### Normalization Parameters

| Parameter | Median | Description |
|------|--------|------|
| T_air | 10.00 °C | Air temperature |
| I_D | 600.00 W/m² | Irradiance |
| R_L | 27.45 Ω | Load resistance |
| V_L | 59.60 V | Load voltage |
| W_dot | 211.80 W | Power output |

### Normalization Formulas

**Input normalization** (M unchanged):
```
x_normalized = [M, T_air/T_air_med, I_D/I_D_med, R_L/R_L_med]
```

**Output normalization**:
```
y_normalized = [V_L/V_L_med, W_dot/W_dot_med]
```

---

## (b) Dataset Split

### Split Method

- **Random shuffle**: Use `numpy.random.shuffle()` to randomly shuffle data indices
- **Fixed seed**: `random_seed = 42` ensures reproducibility
- **Split ratio**: 2/3 training set, 1/3 validation set

### Split Results

| Dataset | Samples | Percentage |
|--------|--------|--------|
| Training | 32 | 66.7% |
| Validation | 16 | 33.3% |
| Total | 48 | 100% |

---

## (c) Neural Network Model Design

### Design Philosophy

Based on Task 4.1 successful experience, following design principles:

1. **Moderate complexity**: Use 3 hidden layers (best performer in 4.1.1)
2. **Medium scale**: Avoid too many parameters causing overfitting
3. **Validation monitoring**: Monitor validation loss to prevent blind fitting to training set
4. **Stable training**: Low learning rate ensures convergence stability

### Network Architecture

```
Sequential Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer (type)          Output Shape      Param #
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input Layer           (None, 4)         -
Dense (Hidden 1)      (None, 8)         40
Dense (Hidden 2)      (None, 14)        126
Dense (Hidden 3)      (None, 8)         120
Dense (Output)        (None, 2)         18
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total params: 304
Trainable params: 304
```

### Design Choices Explanation

| Component | Choice | Rationale |
|------|------|------|
| **Input Layer** | 4 neurons | [M, T_air_norm, I_D_norm, R_L_norm] |
| **Hidden Layer 1** | 8 neurons | Initial feature extraction |
| **Hidden Layer 2** | 14 neurons | Expand feature space (widest layer) |
| **Hidden Layer 3** | 8 neurons | Feature integration |
| **Output Layer** | 2 neurons | [V_L_norm, W_dot_norm] |
| **Activation** | ELU | Continuously differentiable, avoids dead neurons |
| **Output Activation** | Linear | Standard choice for regression tasks |
| **Total Parameters** | 304 | Moderate complexity (vs 4.1.1's 274) |

**Architecture Rationality Analysis**:
- **Moderate complexity**: 304 parameters for 48 samples (32 training + 16 validation) is reasonable
- **Sample-parameter ratio**: Approximately 0.16 (48/304), within acceptable range for deep learning
- **Layer design**: 8→14→8 "expansion-contraction" structure facilitates feature learning
- **Avoid overfitting**: More streamlined compared to skeleton code's 13→26→13 architecture

---

## Training Configuration

### Optimizer Parameters

| Parameter | Value | Description |
|------|------|------|
| Optimizer | RMSprop | Reliable adaptive learning rate optimizer |
| Learning Rate | 0.001 | Lower learning rate ensures stable convergence |
| Loss Function | Mean Absolute Error | Less sensitive to outliers |

### Training Strategy

**Early Stopping**:
- Monitoring metric: `val_loss` ⭐ (key improvement)
- patience: 100 epochs
- Mode: Minimize validation loss
- Weight restoration: Yes (rollback to best epoch)

**Model Checkpoint**:
- Monitoring metric: `val_loss`
- Save strategy: Only save model with minimum validation loss
- Filename: `best_model_4.2.keras`

**Training Parameters**:
- Maximum epochs: 3000 epochs
- Batch processing: Full batch (small dataset)
- Validation: Real-time validation monitoring

### Improvements vs Skeleton Code

| Item | Skeleton Code | Improved | Improvement Description |
|------|---------|--------|---------|
| Architecture | 13-26-13 | **8-14-8** | Reduce parameters, avoid overfitting |
| Learning Rate | 0.020 | **0.001** | Reduce by 20x, more stable |
| Monitoring Metric | loss | **val_loss** | Monitor validation not training ⭐ |
| Validation Data | None | **Yes** | Real-time validation monitoring ⭐ |
| patience | 80 | **100** | More patient waiting for convergence |
| epochs | 800 | **3000** | Allow sufficient training |

**Key Improvement**: Changed from monitoring training loss to **monitoring validation loss**, which is the core strategy to prevent overfitting!

---

## Training Results

### Training Process

- **Best epoch**: (to be filled after training)
- **Minimum training loss**: (to be filled)
- **Minimum validation loss**: (to be filled)
- **Training duration**: (to be filled)

### Model Performance

| Metric | Training Set | Validation Set | Target |
|------|--------|--------|------|
| MAE (normalized) | (to be filled) | (to be filled) | < 0.10 |
| Voltage MAE (V) | (to be filled) | (to be filled) | < 10 V |
| Power MAE (W) | (to be filled) | (to be filled) | < 50 W |
| Overfitting Ratio | - | (to be filled) | < 1.5 |

### Performance Evaluation Criteria

**Overfitting Ratio Judgment**:
- < 1.2: Good generalization ✓
- 1.2-1.5: Mild overfitting
- > 1.5: Significant overfitting ✗

---

## Design Decision Summary

### Lessons Learned from Task 4.1

1. **Architecture choice**: 3 hidden layers superior to 4 layers (on small datasets)
2. **Monitoring strategy**: Validation loss monitoring 50%+ better than training loss monitoring
3. **Learning rate**: ~0.001 low learning rate performs stably for this type of problem
4. **Early stopping patience**: patience=100 is reasonable for our data scale

### Special Considerations for This Task

1. **Mode number handling**: M not normalized, preserves discrete value (0,1,2) semantics
2. **Data scale**: 48 samples slightly less than Task 4.1's 54, need more caution to avoid overfitting
3. **Output dimension**: 2 outputs (vs 4.1's 2), comparable complexity
4. **Parameter scale**: 304 parameters (vs 4.1.1's 274), slightly increased but still reasonable

### Network Design Philosophy

**"Moderate complexity, robust training"**:
- ✓ Don't blindly pursue depth and width
- ✓ Validation monitoring prevents overfitting
- ✓ Low learning rate ensures stability
- ✓ Sufficient training time (automatically controlled by early stopping)

---

## Next Steps

1. Run training and record actual results
2. Analyze prediction accuracy for different modes (M=0,1,2)
3. If improvement needed, consider adjusting learning rate or network structure
4. Complete subsequent task predictions and visualizations

---

*Report generated: 2025-12-17*  
*Based on Task 4.1 optimization experience and CodeP4.2F25.ipynb skeleton code*

## 任务概述

本任务基于多模式光伏系统数据集，训练神经网络模型预测负载电压和功率输出。

**数据集**: CodeP4.3F25.ipynb
- **输入参数**: [M, T_air, I_D, R_L]
  - M: 模式编号 (0, 1, 2) - 不归一化
  - T_air: 空气温度 (°C) - 归一化
  - I_D: 辐照度 (W/m²) - 归一化
  - R_L: 负载电阻 (Ω) - 归一化

- **输出参数**: [V_L, W_dot]
  - V_L: 负载电压 (V) - 归一化
  - W_dot: 功率输出 (W) - 归一化

**数据规模**: 总样本 48 个
- 训练集: 32 个样本 (66.7%)
- 验证集: 16 个样本 (33.3%)

---

## (a) 数据归一化

### 归一化策略

采用**中位数归一化法**（与Task 4.1保持一致）：
- 对除模式编号M外的所有参数计算中位数
- 每个参数值除以其对应的中位数
- **模式编号M保持原值不变**（0, 1, 2）

### 归一化参数

| 参数 | 中位数 | 说明 |
|------|--------|------|
| T_air | 10.00 °C | 空气温度 |
| I_D | 600.00 W/m² | 辐照度 |
| R_L | 27.45 Ω | 负载电阻 |
| V_L | 59.60 V | 负载电压 |
| W_dot | 211.80 W | 功率输出 |

### 归一化公式

**输入归一化**（保持M不变）：
```
x_normalized = [M, T_air/T_air_med, I_D/I_D_med, R_L/R_L_med]
```

**输出归一化**：
```
y_normalized = [V_L/V_L_med, W_dot/W_dot_med]
```

---

## (b) 数据集分割

### 分割方法

- **随机打乱**: 使用 `numpy.random.shuffle()` 随机打乱数据索引
- **固定种子**: `random_seed = 42` 保证结果可重复
- **分割比例**: 2/3 训练集, 1/3 验证集

### 分割结果

| 数据集 | 样本数 | 百分比 |
|--------|--------|--------|
| 训练集 | 32 | 66.7% |
| 验证集 | 16 | 33.3% |
| 总计 | 48 | 100% |

---

## (c) 神经网络模型设计

### 架构设计理念

基于Task 4.1的成功经验，采用以下设计原则：

1. **适度复杂度**: 使用3个隐藏层（4.1.1中表现最佳）
2. **中等规模**: 避免参数过多导致过拟合
3. **验证监控**: 监控验证损失防止盲目拟合训练集
4. **稳定训练**: 低学习率确保收敛稳定性

### 网络架构

```
Sequential Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer (type)          Output Shape      Param #
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input Layer           (None, 4)         -
Dense (Hidden 1)      (None, 8)         40
Dense (Hidden 2)      (None, 14)        126
Dense (Hidden 3)      (None, 8)         120
Dense (Output)        (None, 2)         18
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total params: 304
Trainable params: 304
```

### 设计选择说明

| 组件 | 选择 | 理由 |
|------|------|------|
| **输入层** | 4个神经元 | [M, T_air_norm, I_D_norm, R_L_norm] |
| **隐藏层1** | 8个神经元 | 初始特征提取 |
| **隐藏层2** | 14个神经元 | 扩展特征空间（最宽层） |
| **隐藏层3** | 8个神经元 | 特征整合 |
| **输出层** | 2个神经元 | [V_L_norm, W_dot_norm] |
| **激活函数** | ELU | 连续可微，避免死神经元 |
| **输出激活** | 线性 | 回归任务标准选择 |
| **总参数** | 304个 | 适中复杂度（vs 4.1.1的274个） |

**架构合理性分析**：
- **复杂度适中**: 304个参数对于48个样本（32训练+16验证）是合理的
- **样本-参数比**: 约 0.16 (48/304)，在深度学习可接受范围
- **层次设计**: 8→14→8 的"扩张-收缩"结构有助于特征学习
- **避免过拟合**: 相比骨架代码的13→26→13架构更精简

---

## 训练配置

### 优化器参数

| 参数 | 值 | 说明 |
|------|------|------|
| 优化器 | RMSprop | 稳定可靠的自适应学习率优化器 |
| 学习率 | 0.001 | 较低学习率确保稳定收敛 |
| 损失函数 | Mean Absolute Error | 对异常值不敏感 |

### 训练策略

**早停机制 (Early Stopping)**:
- 监控指标: `val_loss` ⭐（关键改进）
- patience: 100 轮
- 模式: 最小化验证损失
- 权重恢复: 是（回滚到最佳轮次）

**模型检查点 (Model Checkpoint)**:
- 监控指标: `val_loss`
- 保存策略: 仅保存验证损失最小的模型
- 文件名: `best_model_4.2.keras`

**训练参数**:
- 最大轮次: 3000 epochs
- 批处理: 全批次（小数据集）
- 验证: 实时验证监控

### 与骨架代码的改进

| 项目 | 骨架代码 | 改进后 | 改进说明 |
|------|---------|--------|---------|
| 架构 | 13-26-13 | **8-14-8** | 减少参数，避免过拟合 |
| 学习率 | 0.020 | **0.001** | 降低20倍，更稳定 |
| 监控指标 | loss | **val_loss** | 监控验证而非训练 ⭐ |
| 验证数据 | 无 | **有** | 实时验证监控 ⭐ |
| patience | 80 | **100** | 更耐心等待收敛 |
| epochs | 800 | **3000** | 允许充分训练 |

**关键改进**：从监控训练损失改为**监控验证损失**，这是防止过拟合的核心策略！

---

## 训练结果

### 训练过程

- **最佳轮次**: （待训练后填写）
- **最小训练损失**: （待填写）
- **最小验证损失**: （待填写）
- **训练时长**: （待填写）

### 模型性能

| 指标 | 训练集 | 验证集 | 目标 |
|------|--------|--------|------|
| MAE（归一化）| （待填写） | （待填写） | < 0.10 |
| 电压 MAE (V) | （待填写） | （待填写） | < 10 V |
| 功率 MAE (W) | （待填写） | （待填写） | < 50 W |
| 过拟合比值 | - | （待填写） | < 1.5 |

### 性能评估标准

**过拟合比值判断**:
- < 1.2: 良好泛化 ✓
- 1.2-1.5: 轻度过拟合
- \> 1.5: 明显过拟合 ✗

---

## 设计决策总结

### 从Task 4.1学到的经验

1. **架构选择**: 3层隐藏层优于4层（在小数据集上）
2. **监控策略**: 验证损失监控比训练损失监控效果好50%+
3. **学习率**: 0.001左右的低学习率在这类问题上表现稳定
4. **早停耐心**: patience=100 在我们的数据规模下是合理的

### 本任务的特殊考虑

1. **模式编号处理**: M不归一化，保持离散值(0,1,2)的语义
2. **数据规模**: 48个样本比Task 4.1的54个略少，需要更谨慎避免过拟合
3. **输出维度**: 2个输出（vs 4.1的2个），复杂度相当
4. **参数规模**: 304个参数（vs 4.1.1的274个），略有增加但仍合理

### 网络设计理念

**"适度复杂，稳健训练"**：
- ✓ 不盲目追求深度和宽度
- ✓ 验证监控防止过拟合
- ✓ 低学习率确保稳定性
- ✓ 充分的训练时间（通过早停自动控制）

---

## 下一步工作

1. 运行训练并记录实际结果
2. 分析不同模式(M=0,1,2)的预测准确度
3. 如需改进，考虑调整学习率或网络结构
4. 完成后续任务的预测和可视化

---

*报告生成日期: 2025-12-17*  
*基于Task 4.1的优化经验和CodeP4.2F25.ipynb骨架代码*
