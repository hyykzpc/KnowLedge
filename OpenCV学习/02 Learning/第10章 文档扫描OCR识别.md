---
title: 第10章 文档扫描OCR识别
created: 2026-07-28
updated: 2026-07-28
type: note
tags:
  - #opencv
  - #项目实战
  - #OCR
  - #透视变换
status: developing
---

# 第10章：项目实战-文档扫描OCR识别

> **对应视频**: P41-P45 | **时长**: 约35分钟 | **难度**: ★★★

## 一、核心摘要

> 拍照扫描文档的完整流程：找到文档轮廓 → 透视变换矫正 → 二值化增强 → Tesseract OCR 识别文字。核心难点在于轮廓检测和透视变换的坐标排序。

## 二、项目 Pipeline

```
输入: 倾斜的文档照片
    ↓
┌─────────────────────────────────────┐
│ 1. 图像预处理                        │
│    灰度 → 高斯滤波 → Canny 边缘检测   │
├─────────────────────────────────────┤
│ 2. 轮廓检测与筛选                    │
│    findContours → 按面积排序         │
│    → 取最大轮廓 → 多边形近似          │
│    → 筛选四边形（4个顶点）            │
├─────────────────────────────────────┤
│ 3. 透视变换矫正                      │
│    顶点排序(左上/右上/右下/左下)      │
│    → getPerspectiveTransform        │
│    → warpPerspective                │
├─────────────────────────────────────┤
│ 4. 图像增强                          │
│    灰度 → 自适应阈值/二值化           │
│    → 使文字更清晰                    │
├─────────────────────────────────────┤
│ 5. OCR 识别                         │
│    Tesseract 识别文字                │
└─────────────────────────────────────┘
    ↓
输出: 矫正后的文档 + 识别的文字
```

## 三、核心步骤详解

### 3.1 轮廓检测策略

**为什么用面积排序？** 文档是照片中最大的物体，面积最大的轮廓通常就是文档。

```python
cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
# 取前5个面积最大的轮廓，逐一检查
```

### 3.2 顶点排序（关键中的关键）

透视变换需要按顺序指定四个顶点：**左上、右上、右下、左下**。

```python
# 排序算法：
# 1. 按 y 坐标分成两组（上两个、下两个）
# 2. 每组内按 x 坐标排序（左小右大）

def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    # y 坐标和最小的 → 左上角
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]  # y 坐标和最大的 → 右下角
    # x 坐标差最小的 → 右上角
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]  # x 坐标差最大的 → 左下角
    return rect
```

### 3.3 透视变换

**核心思想**：将任意四边形映射到矩形。

```
倾斜的文档:                      矫正后:
    ╱──────────╲                   ┌──────────┐
   ╱            ╲                  │          │
  ╱              ╲     →           │  文档    │
 ╱                ╲                │          │
╱──────────────────╲               └──────────┘
```

**变换矩阵**：`getPerspectiveTransform` 计算 3×3 变换矩阵，`warpPerspective` 执行变换。

### 3.4 原理深入：透视变换的数学原理

**为什么 4 个点就能确定透视变换？**

透视变换矩阵有 8 个自由度（3×3 矩阵，最后一个元素固定为 1），每个点对应两个方程（x 和 y），4 个点正好提供 8 个方程，唯一确定变换矩阵。

$$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

最终坐标: $x_{out} = x'/w'$, $y_{out} = y'/w'$

### 3.5 原理深入：Tesseract OCR 简介

Tesseract 是 Google 维护的开源 OCR 引擎，能识别 100+ 种语言。

**安装与使用**：

```bash
# 安装
pip install pytesseract
# 安装 Tesseract 引擎本体（Windows 需单独下载）
# https://github.com/UB-Mannheim/tesseract/wiki

# 中文支持：下载 chi_sim.traineddata 放到 tessdata 目录
```

## 四、核心 API

| API | 函数签名 | 功能 | 本章用法 |
|-----|----------|------|----------|
| `cv2.getPerspectiveTransform()` | `getPerspectiveTransform(src, dst)` | 计算透视变换矩阵 | 输入 4 个原始点和 4 个目标点 |
| `cv2.warpPerspective()` | `warpPerspective(src, M, dsize)` | 执行透视变换 | 输出矫正后的图像 |
| `cv2.approxPolyDP()` | `approxPolyDP(cnt, epsilon, closed)` | 多边形近似 | 检测文档是否为四边形 |
| `pytesseract.image_to_string()` | `image_to_string(img)` | OCR 识别 | 识别图像中的文字 |

## 五、代码实战

### 5.1 完整流程

```python
import cv2
import numpy as np
import pytesseract

# 配置 tesseract 路径（Windows 需要）
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def order_points(pts):
    """按 左上/右上/右下/左下 顺序排列四个顶点"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 左上
    rect[2] = pts[np.argmax(s)]   # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # 右上
    rect[3] = pts[np.argmax(diff)] # 左下
    return rect

def four_point_transform(img, pts):
    """透视变换矫正"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算变换后的宽度和高度
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = max(int(height_left), int(height_right))

    # 目标矩形
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))
    return warped

# ====== 主流程 ======
img = cv2.imread("document.jpg")
orig = img.copy()
ratio = img.shape[0] / 500.0  # 缩放比例，用于还原坐标

# 1. 预处理
img_resized = cv2.resize(img, (int(img.shape[1]*500/img.shape[0]), 500))
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# 2. 找轮廓
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:  # 找到四边形
        screen_cnt = approx
        break

# 3. 透视变换
warped = four_point_transform(orig, screen_cnt.reshape(4, 2) * ratio)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# 4. 图像增强
warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

# 5. OCR 识别
text = pytesseract.image_to_string(warped, lang="eng")
print("识别结果:\n", text)

cv2.imshow("Scanned", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 5.2 效果对比

| 步骤 | 效果 |
|------|------|
| 原图 → Canny | 提取文档边缘，为轮廓检测做准备 |
| 轮廓筛选 → 近似多边形 | 找到文档的 4 个角点 |
| 透视变换 | 倾斜文档变成矩形，矫正视角 |
| 自适应阈值 | 增强文字对比度，去除背景 |
| OCR | 输出可编辑的文本 |

## 六、调试经验与避坑指南

| 问题现象 | 原因 | 解决方案 |
|----------|------|----------|
| 找不到四边形轮廓 | 轮廓近似精度太高 | 增大 epsilon（如 0.05*周长） |
| 透视变换后图像变形 | 顶点排序错误 | 检查 order_points 逻辑 |
| OCR 识别结果乱码 | 语言包未安装 | 安装对应语言的 traineddata |
| 自适应阈值后文字模糊 | blockSize 不合适 | 调整 blockSize 和 C 参数 |

### 常见坑点
- **坑点1**: 透视变换的目标点坐标顺序必须与源点一致 → 左上对左上，不能错位
- **坑点2**: Tesseract 中文识别需额外配置 → `lang="chi_sim+eng"` 同时识别中英文
- **坑点3**: 缩比检测轮廓后，坐标需乘以 ratio 还原到原始图像尺寸

## 七、本章小结

- **必掌握**: 轮廓检测 → 面积排序 → 多边形近似 → 顶点排序 → 透视变换 → 自适应阈值 → OCR
- **选了解**: Tesseract 多语言配置、透视变换的数学推导
- **一句话总结**: 拍照 → 找文档 → 矫正 → 增强 → OCR，这是文档扫描的完整 pipeline

## 八、知识连接

- [[第6章 边缘检测]] — Canny 边缘检测
- [[第7章 图像金字塔与轮廓检测]] — 轮廓检测与多边形近似
- [[第3章 阈值与平滑处理]] — 自适应阈值
- [[第9章 信用卡数字识别]] — 上一个 OCR 项目（模板匹配方式）

## 九、课后实践

- [ ] 基础练习: 用手机拍一张倾斜的文档，完成透视矫正
- [ ] 进阶挑战: 加入中文 OCR 支持，识别中文文档
- [ ] 思考题: 如果文档有弯曲（非平面），透视变换还能矫正吗？如果不能，该怎么办？