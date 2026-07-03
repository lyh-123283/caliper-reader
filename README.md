# 游标卡尺自动读数

基于 Python、OpenCV 和 Tkinter 的游标卡尺读数工具。当前项目面向 **0.02 mm 精度游标卡尺**，核心流程是：自动定位卡尺 ROI、方向校正、主尺/游标尺区域分离、刻度线检测、主尺数字 OCR、游标 0 刻线定位、游标对齐，最后输出读数和调试图。

> 重要说明：当前版本不再支持手动框选 ROI、不再支持 0.05/0.1 mm 精度切换，也不再使用几何法推断主尺读数。正常读数需要 OCR 成功识别主尺 cm 数字。

## 当前能力

- 自动打开图片并运行完整 pipeline。
- GUI 中逐步显示中间结果，不必等所有步骤结束后才一次性显示。
- 支持“跳过 OCR（算法调试）”模式，用于快速看 ROI、分割、刻线、游标 0 线和对齐结果。
- 固定支持 0.02 mm 游标尺。
- 主尺数字使用 OCR 识别，并将 OCR 数字绑定到最近的 cm 长刻线。
- 游标尺不做整体局部旋转，直接在裁剪出的游标本体和窄刻线带上检测刻线。
- 结果图会标注主尺刻线、游标刻线、游标 0 线、对齐线、读数公式和 OCR 信息。

已用当前测试图验证：

| 图片 | 期望读数 | 当前回归结果 |
| --- | ---: | ---: |
| `youbiaokachi.jpg` | 30.00 mm | 30.00 mm |
| `测试图2.0.jpg` | 50.00 mm | 50.00 mm |

## 环境要求

建议使用项目当前测试过的 Python 解释器：

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

- 推荐安装 Tesseract OCR 二进制程序。仅安装 `pytesseract` Python 包还不够。
- Windows 推荐使用 UB Mannheim Tesseract 发行版。
- 如果 Tesseract 不可用，代码会尝试 EasyOCR。
- 如果没有可用 OCR 引擎，主尺读数会失败，GUI 会显示 OCR 状态。此时不要把最终读数当作有效测量结果。

## 启动

在项目根目录运行：

```powershell
& C:\Users\lenovo\AppData\Local\Programs\Python\Python313\python.exe main.py
```

你也可以在 VSCode 中直接运行 `main.py`。项目里有 Windows Tcl/Tk 路径修复逻辑，用于解决 Python 3.13 下 Tkinter 找不到 Tcl/Tk 脚本的问题。

## GUI 使用

1. 点击“打开图片文件”。
2. 选择卡尺图片。
3. 程序自动运行识别流程。
4. 左侧显示最终读数、置信度、OCR 状态。
5. 右侧标签页显示每一步调试图。

常用操作：

| 功能 | 说明 |
| --- | --- |
| 打开图片 | 支持常见图片格式，使用 `cv2.imdecode` 读取，兼容中文路径。 |
| 跳过 OCR | 只用于算法调试。会跳过主尺数字 OCR，最终读数通常不可信。 |
| 缩放查看 | 支持放大、缩小、适应窗口、100% 显示。 |
| 保存结果 | 保存最终标注图。 |
| 中间图标签页 | 查看预处理、ROI、方向校正、区域分离、主尺刻线、OCR、游标刻线、游标对齐、最终标注、读数推导。 |

## Pipeline 真实流程

### 0. 预处理

文件：`caliper/preprocess.py`

输入原图，输出增强图和二值图。主要步骤：

1. BGR 转灰度。
2. Gamma 校正。
3. 可选高斯滤波。
4. 双边滤波保边降噪。
5. 中值滤波去噪。
6. CLAHE 局部对比度增强。
7. 非锐化增强刻线边缘。
8. 自适应阈值二值化。
9. 形态学开运算。
10. 小连通域过滤。

GUI 中显示的是压缩后的关键预处理视图，不再展示大量很小、难以观察的分散小图。

### 1. ROI 提取与全局方向校正

文件：`caliper/roi_extract.py`

ROI 提取会自动定位包含主尺和游标尺的读数区域。当前没有手动框选入口。

主要策略：

- 优先使用预处理二值图的投影信息定位读数区域。
- 失败时使用 Sobel X 垂直边缘和轮廓评分辅助定位。
- ROI 提取后，对整张 ROI 做全局方向校正。

方向校正仍然使用全局 HoughLinesP：

1. Canny 提取边缘。
2. HoughLinesP 找近似垂直的刻线。
3. 统计刻线角度。
4. 角度小于阈值时不旋转，避免微小噪声导致反向修正。
5. 对 ROI 的彩色图、灰度图、二值图同步旋转。

注意：这里是 **全局 ROI 方向校正**，不是游标尺局部旋转。

### 2. 主尺/游标尺区域分离

文件：`caliper/region_split.py`

目标是在全局校正后的 ROI 中找到主尺和游标尺的真实接缝线。

当前逻辑综合使用：

- 水平方向边缘强度。
- 灰度梯度。
- 二值图闭运算后的水平投影。
- 上下两侧刻线密度和等间距覆盖情况。

输出：

- `split_y`
- `region_main`
- `region_vernier`
- `split_vis`

`split_y` 现在贴合接缝，不再人为向上偏移。

### 3. 主尺刻线检测

文件：`caliper/main_scale.py`

主尺区域会重新二值化并检测竖直刻线。

流程：

1. 自适应阈值，必要时回退 OTSU。
2. 垂直投影找候选峰值。
3. 使用等间距约束补全漏检刻线、去除重复刻线。
4. 对每个候选 x，在二值图附近提取真实竖直刻线段。
5. 根据长度判断长刻线，长刻线用于 cm 数字绑定。

主尺刻线检测本身不负责最终数字读数，OCR 被放到后续合并阶段，因为它需要知道游标 0 线位置。

### 4. 游标尺本体裁剪

文件：`caliper/vernier_rectify.py`

这个模块名字保留了历史命名，但当前真实作用是 **裁剪游标尺本体**：

- 在区域分离后的游标尺区域里，利用上部灰度变化找到游标尺金属本体的左右范围。
- 裁掉左、右无关背景和主尺残留干扰。
- 不做 Hough。
- 不做游标局部旋转。
- 不生成角度校正结果。

输出给后续游标刻线识别的图，是这个裁剪后的游标尺本体区域。

### 5. 游标刻线和 0 刻线检测

文件：`caliper/vernier_scale.py`

当前游标刻线识别重点不是整块图投影，而是先找一条窄刻线带。

流程概况：

1. 在游标本体裁剪图中寻找刻线主要出现的窄纵向范围。
2. 只在这个窄刻线带内做垂直投影。
3. 生成候选刻线。
4. 对每条候选刻线做局部像素拟合，细化 x 坐标。
5. 使用固定 0.02 mm 卡尺的等间距约束过滤刻线序列。
6. 定位游标 0 刻线。
7. 对接近整数读数的情况做近整数吸附修正。

游标 0 刻线当前依赖以下信息共同约束：

- 游标尺本体左边缘。
- 窄刻线带中的真实刻线序列。
- 右侧是否能形成稳定的 0.02 mm 游标刻线间距。
- 与主尺长刻线的近整数关系。

GUI 中的“游标刻度线”图直接基于窄刻线带标注，避免在整块灰度图上看起来发糊或难判断。

### 6. 主尺 OCR 与读数合并

文件：`caliper/merger.py`、`caliper/ocr.py`

主尺读数不是几何回退，而是 OCR 必需路径。

当前主尺 OCR 流程：

1. 根据游标 0 线 `zero_x`，在其上方左侧截取主尺数字候选区域。
2. 在候选区域内找多个可能的数字连通域。
3. 对每个候选数字 patch 做 OCR。
4. 把 OCR 数字绑定到最近的 cm 长刻线。
5. 只保留绑定刻线在 `zero_x` 左侧或非常接近 `zero_x` 的候选。
6. 选择最靠近 `zero_x` 左侧的 cm 数字。
7. 用公式计算主尺读数：

```text
main_reading = digit_value * 10 + extra_ticks
```

其中：

- `digit_value` 是 OCR 识别到的 cm 数字，例如 `3` 表示 30 mm。
- `extra_ticks` 是该 cm 长刻线到游标 0 线之间的主尺 mm 刻线数。

如果 OCR 失败：

- `strategy` 会是 `ocr_failed`。
- 主尺读数返回 0。
- 最终读数不应作为有效测量。

如果启用“跳过 OCR”：

- `strategy` 会是 `skipped`。
- 主尺读数返回 0。
- 该模式仅用于看算法中间图，不用于最终读数。

### 7. 游标对齐和小数读数

文件：`caliper/vernier_scale.py`

游标小数读数固定按 0.02 mm 计算：

```text
vernier_reading = aligned_tick_index * 0.02
```

对齐线选择逻辑：

- 对每条游标刻线，寻找最近的主尺刻线。
- 只考虑合法的 0.02 mm 游标线范围。
- 选择误差最小的对齐线。
- 返回对齐刻线、游标小数读数和对齐置信度。

### 8. 最终结果和标注图

文件：`caliper/merger.py`

最终输出：

```text
total = main_scale + vernier_scale
```

标注图包含：

- 主尺刻线。
- 游标刻线。
- 主尺/游标分割线。
- 游标 0 线。
- OCR 数字候选和选中结果。
- 对齐线。
- 主尺读数、游标读数、总读数。
- 读数推导图。

## 调试图标签页

当前 pipeline 会产生这些主要调试图：

| 标签 | 内容 |
| --- | --- |
| `0_预处理` | 预处理关键图组合。 |
| `1a_ROI提取` | 自动裁剪出的 ROI 彩色图。 |
| `1b_方向矫正` | 全局校正前后对比。 |
| `2_区域分离` | 主尺/游标尺分界线和辅助投影。 |
| `3a_主尺刻度线` | 主尺刻线与游标 0 线总览。 |
| `3b_主尺数字OCR` | OCR 候选框、连通域、选中数字和 OCR 引擎信息。 |
| `4b_游标刻度线` | 窄刻线带上的游标刻线和 0 线。 |
| `4c_游标对齐` | 主尺网格、游标刻线和最佳对齐线。 |
| `5_最终标注` | 最终读数标注图。 |
| `5b_读数推导` | 主尺、游标和总读数的推导视图。 |

## 代码结构

```text
main.py                     Tkinter GUI
requirements.txt            Python 依赖
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
  vernier_scale.py          游标刻线、0 线、对齐和 0.02mm 小数读数
  ocr.py                    Tesseract/EasyOCR OCR 封装
  merger.py                 主尺 OCR 读数、读数合并、最终标注
  utils.py                  通用图像和刻线工具
```

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
pipeline.set_skip_ocr(False)
result = pipeline.run(img)

print(result.total)
print(result.extra_info)
```

算法调试模式：

```python
pipeline = CaliperPipeline()
pipeline.set_skip_ocr(True)
result = pipeline.run(img)
```

注意：`skip_ocr=True` 不适合验证最终读数，因为主尺数字不会被识别。

进度回调：

```python
def on_progress(step_key, image, status):
    print(step_key, status, image.shape)

pipeline = CaliperPipeline()
result = pipeline.run(img, progress_callback=on_progress)
```

GUI 就是通过这个回调逐步刷新中间图。

## 结果结构

`CaliperResult` 主要字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `main_scale` | float | 主尺读数，单位 mm。 |
| `vernier_scale` | float | 游标小数读数，单位 mm。 |
| `total` | float | 总读数，单位 mm。 |
| `precision` | float | 当前固定为 `0.02`。 |
| `confidence` | float | 置信度，0 到 1。 |
| `image_annotated` | ndarray | 最终标注图，BGR。 |
| `debug_images` | dict | 所有中间调试图。 |
| `extra_info` | dict | OCR、刻线、分割、对齐等诊断信息。 |

`extra_info["main_derivation"]` 常见字段：

| 字段 | 说明 |
| --- | --- |
| `strategy` | `ocr`、`ocr_failed` 或 `skipped`。 |
| `nearest_digit` | 被选中的 OCR 数字信息。 |
| `ocr_text` | OCR 识别文本。 |
| `ocr_confidence` | OCR 置信度。 |
| `ocr_engine` | 当前 OCR 引擎。 |
| `ref_tick_x` | OCR 数字绑定到的 cm 长刻线 x 坐标。 |
| `extra_ticks` | 从 cm 长刻线到游标 0 线之间的 mm 刻线数。 |
| `ocr_candidates` | OCR 候选数字列表，含是否选中。 |

## 集中配置

配置入口：

```python
from caliper.config import config
```

常用配置组：

| 配置类 | 对应模块 | 说明 |
| --- | --- | --- |
| `PreprocessConfig` | `preprocess.py` | Gamma、滤波、CLAHE、二值化、形态学、小连通域。 |
| `ROIExtractConfig` | `roi_extract.py` | ROI 搜索范围、轮廓评分、边界 padding。 |
| `OrientConfig` | `roi_extract.py` | 全局 Hough 方向校正参数。 |
| `RegionSplitConfig` | `region_split.py` | 分界线搜索范围、fallback 比例。 |
| `MainScaleConfig` | `main_scale.py` | 主尺峰值、等间距补全、长刻线判断。 |
| `VernierScaleConfig` | `vernier_scale.py` | 游标峰值、0 线验证、等间距补全、对齐阈值。 |
| `OCRConfig` | `ocr.py` | OCR 搜索窗口、Tesseract/EasyOCR 参数。 |
| `MergerConfig` | `merger.py` | 置信度和最终绘图参数。 |

示例：

```python
from caliper.config import config

config.main_scale.spacing_refine_enabled = True
config.vernier_scale.spacing_refine_enabled = True
config.preprocess.gamma = 1.5
config.reset()
```

## 当前不支持或已删除的功能

- 不支持手动框选刻度区域。
- 不支持 0.05 mm 或 0.1 mm 精度卡尺。
- 不支持精度自动推断，固定 0.02 mm。
- 不使用几何法回退主尺读数。
- 不对游标尺局部区域做整体旋转。
- 不在主流程中运行游标局部 Hough 刻线检测。

## 已知限制

- 图片需要能看清主尺数字和游标 0 线附近刻线。
- OCR 失败会直接影响主尺读数。
- 目前鲁棒性主要围绕现有两张测试图迭代，更多拍摄角度、光照、反光、污渍情况还需要继续扩充测试集。
- 如果主尺/游标尺接缝被遮挡或反光严重，区域分离可能失败。
- 如果游标尺左边缘裁剪不准，游标 0 刻线可能被裁掉或误识别。

## 验证命令

编译检查：

```powershell
& C:\Users\lenovo\AppData\Local\Programs\Python\Python313\python.exe -m compileall caliper main.py
```

两张当前回归图：

```powershell
& C:\Users\lenovo\AppData\Local\Programs\Python\Python313\python.exe -c "import cv2,numpy as np; from caliper.pipeline import CaliperPipeline; paths=['youbiaokachi.jpg','测试图2.0.jpg'];
for path in paths:
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    p=CaliperPipeline(); p.set_skip_ocr(False); r=p.run(img)
    print(path, r.main_scale, r.vernier_scale, r.total, r.extra_info.get('main_derivation'))"
```
