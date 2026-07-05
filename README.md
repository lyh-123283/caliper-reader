# 游标卡尺自动读数

这是一个基于 Python、OpenCV、Tkinter 和 OCR 的游标卡尺读数工具。当前版本只面向 **0.02 mm 精度游标卡尺**，流程是：自动提取卡尺读数区域、全局方向校正、主尺/游标尺区域分离、刻线检测、主尺数字 OCR、游标 0 刻线定位、游标对齐，最后输出读数和可视化调试图。

当前版本不再包含手动框选刻度区、精度切换、跳过 OCR 的算法调试模式，也不再使用几何回退法推断主尺读数。主尺整数读数必须依赖 OCR 成功识别主尺 cm 数字。

## 当前能力

- 自动读取图片并运行完整识别 pipeline。
- GUI 通过进度回调逐步刷新中间结果。
- 固定支持 0.02 mm 游标尺。
- 主尺数字支持 OCR 识别 1-15 cm 标记，并支持 10-15 的双字符组合。
- 主尺 OCR 数字会绑定到对应 cm 长刻线，再结合游标 0 线计算主尺读数。
- 游标尺先裁剪出本体区域，再在窄刻线带上检测游标刻线和 0 刻线。
- 游标 0 刻线使用左侧起点、右侧稳定网格和近整数吸附共同约束。
- GUI 会显示预处理、ROI、方向校正、区域分离、主尺刻线、OCR、游标刻线、游标对齐、最终标注和读数推导图。

当前回归结果：

| 图片 | 期望读数 | 当前结果 |
| --- | ---: | ---: |
| `youbiaokachi.jpg` | 30.00 mm | 30.00 mm |
| `测试图2.0.jpg` | 50.00 mm | 50.00 mm |

## 环境要求

建议使用当前项目测试过的 Python：

```powershell
C:\Users\lenovo\AppData\Local\Programs\Python\Python313\python.exe
```

安装 Python 依赖：

```powershell
pip install -r requirements.txt
```

`requirements.txt` 当前包含：

- `opencv-python`
- `numpy`
- `Pillow`
- `pytesseract`
- `easyocr`

OCR 说明：

- 推荐安装 Tesseract OCR 二进制程序，仅安装 `pytesseract` Python 包还不够。
- Windows 推荐使用 UB Mannheim Tesseract 发行版。
- 如果 Tesseract 不可用，代码会尝试 EasyOCR。
- 如果没有可用 OCR 引擎，主尺读数会失败，最终结果不应作为有效测量值。

## 启动 GUI

在项目根目录运行：

```powershell
& C:\Users\lenovo\AppData\Local\Programs\Python\Python313\python.exe main.py
```

也可以在 VSCode 中直接运行 `main.py`。项目中保留了 Windows Tcl/Tk 路径修复逻辑，用于处理 Python 3.13 下 Tkinter 找不到 Tcl/Tk 脚本的问题。

## GUI 使用

1. 点击“打开图片文件”。
2. 选择卡尺图片。
3. 程序自动运行识别流程。
4. 左侧显示主尺读数、游标读数、总读数、置信度和 OCR 状态。
5. 右侧标签页显示各步骤调试图。
6. 需要时点击“保存结果图”保存最终标注图。

## Pipeline 流程

### 0. 图像预处理

文件：`caliper/preprocess.py`

预处理输出两类图像：

- 识别用增强图和二值图：尽量保留刻线，不做过强滤波。
- ROI 定位用结构图：使用更稳定的结构增强，服务于 ROI 提取。

主要步骤包括灰度化、Gamma 校正、轻量双边滤波、CLAHE、锐化、自适应二值化和小连通域过滤。GUI 中只显示关键预处理视图，避免大量小图导致看不清。

### 1. ROI 提取和全局方向校正

文件：`caliper/roi_extract.py`

ROI 提取自动定位包含主尺和游标尺的读数区域。当前没有手动框选入口。

方向校正是全局 ROI 校正，不是游标尺局部旋转。流程是 Canny 边缘、HoughLinesP 检测近似竖直刻线、统计角度，然后对彩色图、灰度图和二值图同步旋转。角度太小时会跳过旋转，避免微小噪声导致反向修正。

### 2. 主尺/游标尺区域分离

文件：`caliper/region_split.py`

目标是在校正后的 ROI 中找到主尺和游标尺的真实接缝。当前逻辑优先使用游标刻线带相关信息定位接缝，并保留灰度接缝和候选扫描作为兜底。`split_y` 贴合真实接缝，不再人为向上偏移。

输出：

- `split_y`
- `region_main`
- `region_vernier`
- `split_vis`

### 3. 主尺刻线检测

文件：`caliper/main_scale.py`

主尺区域会重新二值化并检测竖直刻线。流程是垂直投影找候选峰、等间距约束补全/去重、再在局部二值图中提取真实刻线段。主尺刻线检测本身不计算最终主尺读数，最终主尺读数在合并阶段结合 OCR 和游标 0 线计算。

### 4. 游标尺本体裁剪

文件：`caliper/vernier_rectify.py`

这个模块保留了历史命名，但当前作用是裁剪游标尺本体区域，而不是做局部角度校正。它会裁掉左右无关背景和主尺残留干扰，输出给后续游标刻线检测。

当前不做：

- 游标局部 Hough 检测展示。
- 游标尺整体旋转。
- 游标角度校正 UI。

### 5. 游标刻线和 0 刻线检测

文件：`caliper/vernier_scale.py`

当前重点是在游标本体图中先找到窄刻线带，只在这条窄带里做刻线检测。这样可以减少数字、污点和主尺延伸线对垂直投影的干扰。

主要流程：

1. 在接缝下方寻找游标刻线主要出现的窄纵向范围。
2. 在窄刻线带中做垂直投影，生成刻线候选。
3. 对候选刻线做局部像素拟合，细化 x 坐标。
4. 使用固定 0.02 mm 游标尺的网格间距过滤伪刻线。
5. 从靠左的真实起点开始验证右侧稳定刻线序列，定位游标 0 刻线。
6. 对接近整数读数的情况做近整数吸附，修正小的系统偏移。
7. 与主尺刻线对齐，得到游标小数读数。

GUI 中“游标刻度线”图直接标注在窄刻线带上，便于检查 0 线和候选刻线。

### 6. 主尺 OCR 和读数合并

文件：`caliper/merger.py`、`caliper/ocr.py`

主尺整数读数走 OCR 必需路径：

1. 根据游标 0 线 `zero_x`，在其上方左侧截取主尺数字候选区域。
2. 在候选区域里找数字连通域。
3. 对每个候选数字 patch 做 OCR。
4. 将 OCR 数字绑定到对应的 cm 长刻线。
5. 支持把相邻字符组合为 10-15 的两位数 cm 标记。
6. 只保留绑定刻线在 `zero_x` 左侧或非常接近 `zero_x` 的候选。
7. 选择最靠近 `zero_x` 的有效 cm 数字。
8. 计算主尺读数：

```text
main_reading = digit_value * 10 + extra_ticks
```

其中 `digit_value` 是 OCR 识别到的 cm 数字，例如 `3` 表示 30 mm；`extra_ticks` 是该 cm 长刻线到游标 0 线之间的主尺 mm 刻线数。

### 7. 最终结果

最终读数：

```text
total = main_scale + vernier_scale
```

`CaliperResult` 主要字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `main_scale` | float | 主尺读数，单位 mm |
| `vernier_scale` | float | 游标小数读数，单位 mm |
| `total` | float | 总读数，单位 mm |
| `precision` | float | 固定为 `0.02` |
| `confidence` | float | 置信度，0 到 1 |
| `image_annotated` | ndarray | 最终标注图，BGR |
| `debug_images` | dict | GUI 使用的中间调试图 |
| `extra_info` | dict | OCR、刻线、0 线和对齐诊断信息 |

## Python API

读取单张图片：

```python
from caliper.pipeline import read_caliper

result = read_caliper("youbiaokachi.jpg")
print(result.main_scale, result.vernier_scale, result.total)
```

从 OpenCV 图像数组读取：

```python
import cv2
import numpy as np
from caliper.pipeline import CaliperPipeline

img = cv2.imdecode(np.fromfile("youbiaokachi.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)

pipeline = CaliperPipeline()
result = pipeline.run(img)

print(result.total)
print(result.extra_info)
```

进度回调：

```python
def on_progress(step_key, image, status):
    print(step_key, status, image.shape)

pipeline = CaliperPipeline()
result = pipeline.run(img, progress_callback=on_progress)
```

GUI 通过这个回调逐步刷新中间图。

## 调试标签页

当前主要调试图：

| 标签 | 内容 |
| --- | --- |
| `0_预处理` | 原图、增强图、自适应二值图、最终过滤图 |
| `1a_ROI提取` | 自动裁剪出的 ROI |
| `1b_方向矫正` | 全局方向校正结果 |
| `2_区域分离` | 主尺/游标尺分界线 |
| `3a_主尺刻度线` | 主尺刻线和游标 0 线总览 |
| `3b_主尺数字OCR` | OCR 候选框、选中数字和 OCR 状态 |
| `4b_游标刻度线` | 窄刻线带上的游标刻线和 0 线 |
| `4c_游标对齐` | 主尺网格、游标刻线和最佳对齐线 |
| `5_最终标注` | 最终读数标注图 |
| `5b_读数推导` | 主尺、游标和总读数推导图 |

## 代码结构

```text
main.py                     Tkinter GUI
requirements.txt            Python 依赖
youbiaokachi.jpg            示例/回归图片
测试图2.0.jpg               本地回归图片
caliper/
  __init__.py               包导出
  config.py                 集中配置
  result.py                 CaliperResult / TickInfo / DigitInfo
  pipeline.py               主 pipeline
  preprocess.py             图像预处理
  roi_extract.py            ROI 提取和全局方向校正
  region_split.py           主尺/游标尺区域分离
  main_scale.py             主尺刻线检测、OCR 候选区域和数字连通域
  vernier_rectify.py        游标尺本体裁剪
  vernier_scale.py          游标刻线、0 线、对齐和 0.02 mm 小数读数
  ocr.py                    Tesseract/EasyOCR 封装
  merger.py                 主尺 OCR 读数、读数合并和最终标注
  utils.py                  通用图像和刻线工具
```

## 当前不支持的功能

- 不支持手动框选刻度区域。
- 不支持 0.05 mm 或 0.1 mm 精度卡尺。
- 不支持精度自动推断，固定 0.02 mm。
- 不使用几何法回退主尺读数。
- 不提供跳过 OCR 的算法调试模式。
- 不对游标尺局部区域做整体旋转。

## 已知限制

- 图片需要能看清主尺数字、游标 0 线附近刻线和主尺/游标尺接缝。
- OCR 失败会直接影响主尺读数。
- 当前鲁棒性主要围绕现有两张测试图迭代，更多光照、角度、反光、污渍场景还需要扩充测试集。
- 如果游标尺左边缘裁剪不准，游标 0 刻线可能被裁掉或误识别。
- 如果主尺长刻线与普通刻线长度差异不明显，OCR 数字和 cm 刻线绑定可能失败。

## 验证命令

编译检查：

```powershell
& C:\Users\lenovo\AppData\Local\Programs\Python\Python313\python.exe -m compileall caliper main.py
```

两张回归图：

```powershell
& C:\Users\lenovo\AppData\Local\Programs\Python\Python313\python.exe -c "import cv2,numpy as np; from caliper.pipeline import CaliperPipeline; paths=['youbiaokachi.jpg','测试图2.0.jpg'];
for path in paths:
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    p=CaliperPipeline(); r=p.run(img)
    print(path, r.main_scale, r.vernier_scale, r.total, r.extra_info.get('main_derivation'))"
```
