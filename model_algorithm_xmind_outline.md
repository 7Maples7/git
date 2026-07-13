# 回波三分类模型算法流程

## 1. 数据输入

### 支持格式

#### SQLite 数据库

##### 默认表

`mSignalProcEcho`

##### 默认数据列

`_data` 或 `data`

##### 默认标签列

`_type` / `type` / `label` / `target`

#### NPZ 文件

##### MTD 字段

`mtd_echo` / `doppler` / `doppler_pulse` / `pc_doppler` / `pulse_doppler`

##### PC 字段

`pc_echo` / `pc` / `pulse_compression` / `pulse_compressed`

##### 标签字段

`label` / `labels` / `y` / `target` / `type`

### 样本结构 RadarSample

#### 信号数据

`mtd_echo`

`pc_echo`

#### 波形参数

`range_m`

`pw_us`

`prt_us`

`prt_nbr`

`pluse_band`

`sample_freq`

#### 维度参数

`mtd_row`

`mtd_col`

`mtd_ch`

`pc_row`

`pc_col`

`pc_ch`

#### 标签

`label`

`sample_id`

## 2. 原始 payload reshape

### 输入

展平的一维 float payload

### 维度解释

`points = row * col`

`channels = max(ch, 1)`

### 输出

`[points, channels]`

### 典型 PC 维度

`pc_dim = [1, prt_nbr, 2]`

含义是长度为 `prt_nbr` 的慢时间 I/Q 双通道数据

## 3. MTD 时域输入

### 双通道

`mtd_echo = sqrt(I^2 + Q^2)`

### 单通道

`mtd_echo = abs(x)`

### 标准化

`log1p(abs(x))`

再做单样本 z-score

### Batch 变量

`mtd_echo`

`mtd_mask`

## 4. PC 时域输入

### 双通道

`pc_echo = sqrt(I^2 + Q^2)`

### 单通道

`pc_echo = abs(x)`

### 标准化

`log1p(abs(x))`

再做单样本 z-score

### Batch 变量

`pc_echo`

`pc_mask`

## 5. PC 频谱输入

### 复数序列

`pc_complex = I + jQ`

### 频谱变换

`pc_spectrum_raw = abs(fftshift(fft(pc_complex)))`

### 标准化

`log1p(abs(pc_spectrum_raw))`

再做单样本 z-score

### Batch 变量

`pc_spectrum`

`pc_spectrum_mask`

### 注意

模型输入的完整频谱序列是 abs 幅度谱，不是 dB 谱

## 6. Meta 特征

### 维度

9 维

### 特征列表

`log_range_m`

`log_pw_us`

`log_prt_us`

`log_prt_nbr`

`duty`

`log_time_bandwidth`

`log_mtd_energy`

`pc_spectrum_db_mean`

`log_pc_len`

### PC 频谱 dB 均值

`mean(20 * log10(max(abs(fftshift(fft(I+jQ))), 1e-12)))`

### Batch 变量

`meta`

## 7. Batch 拼接

### Padding

同一 batch 内按最长序列补零

### Mask

`True` 表示有效数据

`False` 表示 padding

### Batch 输出

`mtd_echo`

`mtd_mask`

`pc_echo`

`pc_mask`

`pc_spectrum`

`pc_spectrum_mask`

`meta`

`label`

`sample_id`

## 8. 模型结构

### MTD CNN 分支

输入 `mtd_echo`

输出 256 维

### PC 时域 CNN 分支

输入 `pc_echo`

输出 256 维

### PC 频谱 CNN 分支

输入 `pc_spectrum`

输出 256 维

### Meta MLP 分支

输入 9 维 meta

输出 64 维

## 9. CNN 分支结构

### 第一层

`Conv1d(1 -> 32, kernel=5, padding=2)`

`GroupNorm`

`SiLU`

### 第二层

`Conv1d(32 -> 64, kernel=5, stride=2, padding=2)`

`GroupNorm`

`SiLU`

### 第三层

`Conv1d(64 -> 128, kernel=3, stride=2, padding=1)`

`GroupNorm`

`SiLU`

### 池化

时间维 mean pooling

时间维 max pooling

### 输出

`128 + 128 = 256`

## 10. 融合与分类

### 融合维度

`256 + 256 + 256 + 64 = 832`

### 分类头

`Linear(832 -> 256)`

`LayerNorm`

`SiLU`

`Dropout`

`Linear(256 -> num_classes)`

### 输出

`logits`

默认三分类：无人机、鸟、杂波

## 11. 训练流程

### 损失函数

`CrossEntropyLoss`

### 类别权重

默认启用 class weight

### 优化器

`AdamW`

### 默认学习率

`1e-3`

### 默认 weight decay

`1e-4`

### 默认 dropout

`0.2`

### 默认梯度裁剪

`5.0`

## 12. 输出文件

### 最优模型

`best.pth`

### 最后一轮模型

`last.pth`

### 指标文件

`metrics.json`

### 训练配置

`train_config.json`
