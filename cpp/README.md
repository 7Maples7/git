# RadarThreeRecognizer C++/Qt 桥接说明

本目录新增了一套 C++/Qt 调用 `radar_three_cls` 三分类模型的桥接代码。C++ 侧通过 `QProcess` 启动 Python worker，双方使用 JSON Lines 协议通信，业务侧只需要把现有 `DClsEcho::data()` 返回的二进制点迹传入桥接类即可获得模型预测结果。

## 新增文件

- `../radar_three_recognizer_worker.py`：Python 常驻识别 worker，负责加载 `best.pth` 或 `last.pth` 并执行模型推理。
- `../radar_three_recognizer_worker.spec`：PyInstaller 打包配置，可将 worker 打包成独立可执行文件。
- `RadarThreeRecognizerBridge.h/.cpp`：C++/Qt 封装类，负责启动 worker、发送命令、读取结果、处理超时与错误信息。
- `example_main.cpp`：命令行示例程序，用于快速验证 checkpoint、worker 和 `DClsEcho` 二进制输入是否能连通。
- `radar_three_cls_client.pro`：qmake 示例工程。

## 通信协议

C++ 侧向 worker 写入一行紧凑 JSON，worker 也返回一行紧凑 JSON。

支持命令：

- `ping`：检测 worker 是否存活。
- `init`：加载模型 checkpoint，支持 `device=auto/cpu/cuda` 和 `strict_model_load`。
- `recognize_echo`：执行单条点迹识别，参数为 `echo_blob_b64`，内容是 `DClsEcho::data()` 的 base64 编码。
- `shutdown`：结束 worker。

`recognize_echo` 返回的主要字段：

- `pred_target_type` / `pred_target_type_hex`：预测目标类型，十进制和十六进制同时返回。
- `pred_class_id` / `pred_class_id_hex`：模型内部类别编号。
- `pred_label`：训练时保存的标签文本。
- `score`、`top1_score`、`top2_score`、`margin`：置信度和前两名分数差。
- `probabilities`、`probabilities_by_class_id`、`topk`：完整概率分布和 Top-K 结果。

## qmake 示例

```qmake
QT += core
CONFIG += console c++11

SOURCES += \
    cpp/RadarThreeRecognizerBridge.cpp \
    cpp/example_main.cpp

HEADERS += \
    cpp/RadarThreeRecognizerBridge.h
```

## 命令行验证

```powershell
radar_three_cls_client.exe `
  C:\path\to\best.pth `
  C:\path\to\radar_three_recognizer_worker.py `
  C:\path\to\dcls_echo_blob.bin `
  python `
  auto
```

如果 worker 已经通过 PyInstaller 打包成 exe，第二个参数可以直接传入 `radar_three_recognizer_worker.exe`。

## 集成提示

在主 Qt 工程中创建 `RadarThreeRecognizerBridge`，初始化时传入 checkpoint 路径、worker 脚本或 exe 路径、Python 路径和推理设备。识别时直接调用：

```cpp
RadarThreeCls::RadarThreeRecognizeResult result = bridge.recognize(dclsEcho.data());
```

若 `result.ok` 为 `false`，可读取 `result.errorCode` 和 `result.errorMsg` 定位启动、模型加载、超时或输入数据问题。
