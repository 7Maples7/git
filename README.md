# radar_three_cls 回波三分类模型

本目录用于训练基于 `DClsEcho` 输入协议的三分类 PyTorch 模型。当前输入已经按 `DClsEcho` 结构体对齐：

- `mtd_echo`：目标所在距离单元对应的多普勒维数据，长度按 `prt_nbr` 补零或截断。
- `pc_echo`：目标所在多普勒单元对应的脉压数据，固定为 129 点，代码会自动补零或截断。
- 标量参数：`range/pw_us/prt_us/prt_nbr/pluse_band/sample_freq`，用于构造 meta 特征。
- 标签：SQLite 数据库默认读取 `mSignalProcEcho._type`；`.npz` 默认读取 `label/labels/y/target/type`。

模型输出为三分类 logits，推理时输出：

- `pred_id`
- `pred_label`
- `score`
- `probabilities`

## 支持的数据来源

### 1. SQLite 实录数据

可以直接传入 `.db/.sqlite/.sqlite3` 文件，或包含这些文件的目录。程序会从 `mSignalProcEcho` 表中读取 `_data`，按 `DClsEcho::data()` 二进制协议解析。

默认识别的列：

- 数据列：`_data` 或 `data`
- 标签列：`_type`、`type`、`label` 或 `target`
- 样本编号列：`_id` 或 `id`

### 2. `.npz` 数据

推荐字段：

```python
import numpy as np

np.savez(
    "dataset.npz",
    mtd_echo=np.array([sample1_mtd_echo, sample2_mtd_echo], dtype=object),
    pc_echo=np.stack([sample1_pc_echo_129, sample2_pc_echo_129]),
    pw_us=np.array([...], dtype=np.float32),
    prt_us=np.array([...], dtype=np.float32),
    prt_nbr=np.array([...], dtype=np.float32),
    pluse_band=np.array([...], dtype=np.float32),
    sample_freq=np.array([...], dtype=np.float32),
    label=np.array([0, 1, 2], dtype=np.int64),
)
```

兼容旧字段名：`doppler` 会被当作 `mtd_echo`，`mtd` 会被当作 `pc_echo`。

## 训练

自动划分训练、验证、测试集：

```powershell
python -m radar_three_cls.train `
  --data path\to\echo_dataset.db `
  --out-dir radar_three_cls\runs\real_data `
  --epochs 50 `
  --batch-size 32 `
  --device auto
```

已经分好数据时：

```powershell
python -m radar_three_cls.train `
  --train-data path\to\train.db `
  --val-data path\to\val.db `
  --test-data path\to\test.db `
  --out-dir radar_three_cls\runs\real_data
```

训练输出：

- `best.pth`：验证集准确率最好的模型。
- `last.pth`：最后一轮模型。
- `metrics.json`：准确率、精确率、召回率、F1、混淆矩阵和训练历史。
- `train_config.json`：训练配置。

## 推理

```powershell
python -m radar_three_cls.infer `
  --checkpoint radar_three_cls\runs\real_data\best.pth `
  --input path\to\echo_dataset.db `
  --device auto `
  --output radar_three_cls\runs\real_data\infer.json
```

## 点迹可视化界面

启动交互式查看器：

```powershell
python -m radar_three_cls.echo_dataset_viewer
```

也可以启动后直接加载指定数据集：

```powershell
python -m radar_three_cls.echo_dataset_viewer "D:\A3回波录取数据\第五版协议下点迹录取清洗后的数据\数据集26_06_06_15_35_14"
```

界面支持选择目录或 `.db/.sqlite/.npz` 文件，加载后可通过点迹列表、上一条/下一条、序号跳转查看单个点迹的两张图和波形参数。

## 自检

```powershell
python -m radar_three_cls.make_example_data --output radar_three_cls\example_data_dcls.npz
python -m radar_three_cls.train --data radar_three_cls\example_data_dcls.npz --out-dir radar_three_cls\runs\demo --epochs 3
python -m radar_three_cls.infer --checkpoint radar_three_cls\runs\demo\best.pth --input radar_three_cls\example_data_dcls.npz
```

合成数据只用于确认代码链路能跑通，不代表真实雷达数据效果。
