[English](./README_EN.md) | [简体中文](./README.md)

# BEVFormer 推理

BEVFormer DEMO on Axera

## 支持平台

- [x] AX650
- [ ] AX637

## 项目结构

```
bevformer/
├── CMakeLists.txt          # 构建配置文件
├── build650.sh             # AX650 构建脚本
├── build637.sh             # AX637 构建脚本
├── README.md               # 本文档（中文版）
├── README_EN.md            # 英文版文档
├── toolchains/
│   └── aarch64-none-linux-gnu.toolchain.cmake  # 交叉编译工具链
├── include/                # 头文件
│   ├── bevformer_common.hpp
│   ├── data_loader.hpp
│   ├── preprocess.hpp
│   ├── postprocess.hpp
│   ├── visualization.hpp
│   ├── utils.hpp
│   └── timer.hpp
└── src/                    # 源文件
│   ├── main.cpp
│   ├── data_loader.cpp
│   ├── preprocess.cpp
│   ├── postprocess.cpp
│   ├── visualization.cpp
│   └── utils.cpp
├── script/                 # Python 参考实现
│   └── inference_axmodel.py  # 使用 axengine 的 Python 推理脚本
└── bevformer_onnx_export/ 
```

## 依赖项

- OpenCV (>= 3.0)
- AXERA BSP (msp/out 目录) - 芯片特定（AX650 或 AX637）
- CMake (>= 3.13)
- C++14 编译器
- jsoncpp（可选，用于 JSON 解析 - 可使用简化解析器）
- 交叉编译工具链（用于在 x86_64 主机上编译 aarch64 目标）

## 构建

### 自动化构建（推荐）

项目为不同的芯片类型提供了独立的构建脚本：

#### 对于 AX650：
```bash
./build650.sh
```

#### 对于 AX637：
```bash
./build637.sh
```

构建脚本将自动：
1. 检查并验证系统依赖项（cmake、wget、unzip、tar、git、make）
2. 下载并设置适用于 aarch64 的 OpenCV 库
3. 克隆并设置目标芯片的 BSP SDK
4. 下载并设置交叉编译工具链（适用于 x86_64 主机）
5. 使用适当的芯片类型配置 CMake 并构建项目

**注意**： 
- 首次运行时，脚本将下载约 500MB 的依赖项。后续运行将重用缓存文件。
- 构建输出存储在单独的目录中：`build_ax650/` 和 `build_ax637/`
- 每个构建脚本专用于其特定的芯片类型（build650.sh 用于 AX650，build637.sh 用于 AX637）

### 手动构建

如果您更喜欢手动构建：

```bash
# 对于 AX650
mkdir build_ax650 && cd build_ax650
cmake -DBSP_MSP_DIR=/path/to/ax650/msp/out -DAXERA_TARGET_CHIP=ax650 ..
make -j$(nproc)

# 对于 AX637
mkdir build_ax637 && cd build_ax637
cmake -DBSP_MSP_DIR=/path/to/ax637/msp/out -DAXERA_TARGET_CHIP=ax637 ..
make -j$(nproc)
```

#### 手动依赖项设置

1. **OpenCV**：从[这里](https://github.com/AXERA-TECH/ax-samples/releases/download/v0.1/opencv-aarch64-linux-gnu-gcc-7.5.0.zip)下载并解压到 `3rdparty/`
2. **BSP SDK**： 
   - 对于 AX650：从 `https://github.com/AXERA-TECH/ax650n_bsp_sdk.git` 克隆
   - 对于 AX637：从 `https://github.com/AXERA-TECH/ax637_bsp_sdk.git` 克隆（即将推出...）
3. **工具链**：从 [ARM](https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz) 下载并解压

#### CMake 配置选项

- `AXERA_TARGET_CHIP`：目标芯片类型（ax650 或 ax637，默认：ax650）
- `BSP_MSP_DIR`：BSP msp/out 目录的路径
- `CMAKE_TOOLCHAIN_FILE`：交叉编译工具链文件的路径（用于交叉编译）


## 使用方法

```bash
./bevformer_inference <model> <config_json> <data_dir> [options]
```

### 参数

- `model`：BEVFormer AXModel 文件路径（.axmodel）
- `config_json`：配置文件 JSON 路径
- `data_dir`：提取的数据目录路径（应包含 scene_index.json 和场景文件夹）

### 选项

- `--output-dir <dir>`：输出目录（默认：./inference_results_extracted）
- `--score-thr <float>`：分数阈值（默认：0.1）
- `--fps <int>`：视频帧率（默认：3）
- `--start-scene <int>`：起始场景索引（默认：0）
- `--end-scene <int>`：结束场景索引（默认：全部）

### 示例

```bash
./bevformer_inference \
    models/bevformer.axmodel \
    config/bevformer_config.json \
    data/extracted_data \
    --output-dir ./results \
    --score-thr 0.3 \
    --fps 3
```

## Python 参考实现

项目还包含一个 Python 参考实现，位于 `script/inference_axmodel.py`，使用 `axengine` Python 库进行推理。该脚本用作：

- **参考实现**：演示完整推理流程的工作 Python 版本
- **验证工具**：可通过比较输出来验证 C++ 实现的正确性
- **开发辅助**：在开发过程中更容易修改和调试

### Python 脚本使用方法

```bash
python3 script/inference_axmodel.py <model> <config_json> <data_dir> [options]
```

**参数和选项**（与 C++ 版本相同）：
- `model`：BEVFormer AXModel 文件路径（.axmodel）
- `config_json`：配置文件 JSON 路径
- `data_dir`：提取的数据目录路径
- `--output-dir <dir>`：输出目录（默认：./inference_results_extracted）
- `--score-thr <float>`：分数阈值（默认：0.1）
- `--fps <int>`：视频帧率（默认：3）
- `--start-scene <int>`：起始场景索引（默认：0）
- `--end-scene <int>`：结束场景索引（默认：全部）

### Python 依赖项

```bash
opencv-python numpy axengine tqdm
```

### 示例

```bash
python3 script/inference_axmodel.py \
    models/bevformer.axmodel \
    config/bevformer_config.json \
    data/extracted_data \
    --output-dir ./results \
    --score-thr 0.3 \
    --fps 3
```

**注意**：Python 脚本产生与 C++ 版本相同的输出格式，便于比较结果和验证 C++ 实现。

## 数据格式

数据目录应具有以下结构：

```
data_dir/
├── scene_index.json
└── scene_xxx/
    ├── meta_000000.json
    ├── cam_00_000000.png
    ├── cam_01_000000.png
    ├── cam_02_000000.png
    ├── cam_03_000000.png
    ├── cam_04_000000.png
    └── cam_05_000000.png
    ...
```

## 模型和数据集

### 预转换模型

预转换的 AXModel 文件可从以下地址下载：
- **模型**：[https://huggingface.co/AXERA-TECH/bevformer](https://huggingface.co/AXERA-TECH/bevformer)
- **示例数据集**：[https://huggingface.co/AXERA-TECH/bevformer](https://huggingface.co/AXERA-TECH/bevformer)
- **推理 Json 文件**：[https://huggingface.co/AXERA-TECH/bevformer](https://huggingface.co/AXERA-TECH/bevformer)

### 模型转换

如果您想自己转换模型，请参考 `bevformer_onnx_export` 文件夹，其中包含：

- ONNX 导出脚本
- 量化数据集准备脚本
- 针对 AXERA 芯片兼容性的模型修改
- 使用 onnxruntime 工具的 ONNX 模型推理（inference_onnx.py）

**模型转换要求：**
1. 按照 `bevformer_onnx_export` 文件夹中的说明操作
2. 根据要求配置环境
3. 准备相应的数据集

**ONNX 到 AXModel 转换：**
有关将 ONNX 模型转换为 AXModel 格式的信息，请参考 [Pulsar2 文档](https://pulsar2-docs.readthedocs.io/en/latest/index.html)。

## 输出

### 输出结构

运行推理后，程序会生成以下输出结构：

```
<output_dir>/
├── <scene_id>/
│   ├── images/
│   │   ├── frame_000000.png    # 第 0 帧可视化
│   │   ├── frame_000001.png    # 第 1 帧可视化
│   │   └── ...
│   └── <scene_id>_result.avi   # 合并的视频文件
└── ...
```

**输出文件：**
- **`images/frame_XXXXXX.png`**：单个帧的可视化图像，显示：
  - 6 个相机视图，每个图像上投影了 3D 边界框
  - BEV（鸟瞰图）地图，从俯视角度可视化 3D 检测结果
- **`<scene_id>_result.avi`**：包含所有帧连接在一起的视频文件（MJPG 编解码器）

### 输出示例

运行推理命令：

```bash
./bevformer_inference ax650/compiled.axmodel config.json inference_data/ --output-dir results
```

**控制台输出：**

```
Loaded configuration:
  bev_h=200, bev_w=200, embed_dims=256
  max_num=100, score_threshold=0.1
  pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3]
  post_center_range=[-61.2, -61.2, -10, 61.2, 61.2, 10]

Found 2 scenes by directory scan
Total scenes to process: 2
  - 325cef682f064c55a255f2625c533b75: 41 frames
  - fcbccedd61424f1b85dcbf8f897f9754: 40 frames

Processing scene 1/2: 325cef682f064c55a255f2625c533b75 (41 frames)
Processing: [========================================] 100% [41/41] 55.1fps, ETA: 00:00

Performance Statistics (Scene: 325cef682f064c55a255f2625c533b75, 41 frames):
  Load:       avg=107.22 ms
  Preprocess: avg=29.59 ms
  Inference:  avg=89.83 ms (min=89.60, max=90.05)
  Postprocess: avg=1.86 ms
  Visualize:  avg=153.40 ms
  Total:      avg=744.28 ms (min=701.33, max=795.70), FPS=1.34

Created video with 41 frames using codec MJPG (AVI): results/325cef682f064c55a255f2625c533b75/325cef682f064c55a255f2625c533b75_result.avi
Scene 325cef682f064c55a255f2625c533b75 completed
```

**性能统计说明：**
- **Load**：从磁盘加载相机图像的时间
- **Preprocess**：图像预处理时间
- **Inference**：NPU 模型推理时间（在 AX650 上通常每帧约 90ms）
- **Postprocess**：检测解码和过滤时间
- **Visualize**：渲染 3D 框和创建可视化图像的时间
- **Total**：每帧的端到端处理时间
- **FPS**：有效帧率（平均总时间的倒数）

### 可视化结果

每帧可视化包括：
- **6 个相机视图**（顶部行）：前右、前、前左、后右、后、后左相机，带有投影的 3D 边界框
- **BEV 地图**（底部）：俯视图，显示检测到的对象及其边界框、标签和方向箭头

![可视化示例](./asset/output.gif)


## 注意事项

- 当前实现使用简化的 JSON 解析器。对于生产使用，请考虑集成 jsoncpp 库。
- 确保 BSP 目录包含所需的库（ax_engine、ax_interpreter、ax_sys、ax_ivps）。
- 模型输入应匹配：img、can_bus、lidar2img、prev_bev
- 模型输出应包括：cls_scores、bbox_preds、bev_embed

## 技术讨论

- Github issues
- QQ 群: 139953715