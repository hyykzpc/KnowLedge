---
title: 第18章 OpenCV的DNN模块
created: 2026-07-28
updated: 2026-07-28
type: note
tags:
  - #opencv
  - #深度学习
  - #DNN
  - #模型推理
status: developing
---

# 第18章：OpenCV的DNN模块

> **对应视频**: P78-P79（DNN模块部分） | **时长**: 约20分钟 | **难度**: ★★★

## 一、核心摘要

> OpenCV 的 DNN（Deep Neural Network）模块允许在不安装 PyTorch/TensorFlow 的情况下加载和推理深度学习模型。支持 Caffe、TensorFlow、ONNX、Darknet 等主流框架的模型格式。

## 二、知识图谱

```
OpenCV DNN 模块
├── 支持的框架
│   ├── Caffe (.caffemodel + .prototxt)
│   ├── TensorFlow (.pb)
│   ├── ONNX (.onnx)
│   ├── Darknet (.cfg + .weights)
│   └── Torch (.t7)
│
├── 推理流程
│   ├── readNet → 加载模型
│   ├── blobFromImage → 预处理（缩放/减均值/通道交换）
│   ├── setInput → 设置输入
│   ├── forward → 前向推理
│   └── 后处理 → 解析输出
│
└── 应用场景
    ├── 图像分类
    ├── 目标检测（SSD, YOLO）
    ├── 人脸识别
    └── 姿态估计
```

## 三、核心概念

### 3.1 blobFromImage

**为什么需要 blobFromImage？**

深度学习模型对输入有严格要求，blobFromImage 自动完成：

| 操作 | 参数 | 说明 |
|------|------|------|
| 缩放 | size | 统一到模型要求的输入尺寸 |
| 减均值 | mean | 减去训练集的均值（BGR 顺序） |
| 通道交换 | swapRB | BGR→RGB 或反之 |
| 归一化 | scalefactor | 像素值缩放（如 1/255） |

```python
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                              mean=(104.0, 177.0, 123.0), swapRB=True)
```

### 3.2 推理流程

```
模型文件 (.caffemodel + .prototxt)
    ↓ readNet
加载到内存
    ↓ blobFromImage
预处理输入图像
    ↓ setInput + forward
前向推理
    ↓
输出结果（分类概率 / 检测框 / ...）
    ↓ 后处理
解析为可读结果
```

### 3.3 原理深入：为什么 DNN 模块有用？

**传统方式**：需要安装 PyTorch/TensorFlow → 加载模型 → 推理 → 结果。环境复杂，依赖重。

**OpenCV DNN**：只需 opencv-contrib-python → 一行 readNet → 推理。适合部署和轻量级应用。

**局限**：DNN 模块只支持**推理**，不支持训练。模型需要在其他框架中训练好后导出。

### 3.4 原理深入：均值减去的意义

训练深度学习模型时，数据预处理通常包括**减均值**（mean subtraction）。推理时也必须做同样的预处理，否则结果会错误。

不同模型的均值不同：

| 模型 | 均值 (BGR) |
|------|-----------|
| Caffe 默认 | (104.0, 177.0, 123.0) |
| TensorFlow 默认 | (127.5, 127.5, 127.5) |
| ImageNet 标准 | (123.68, 116.78, 103.94) |

## 四、核心 API

| API | 函数签名 | 功能 |
|-----|----------|------|
| `cv2.dnn.readNetFromCaffe()` | `readNetFromCaffe(prototxt, caffeModel)` | 加载 Caffe 模型 |
| `cv2.dnn.readNetFromTensorflow()` | `readNetFromTensorflow(model[, config])` | 加载 TensorFlow 模型 |
| `cv2.dnn.readNetFromONNX()` | `readNetFromONNX(onnxFile)` | 加载 ONNX 模型 |
| `cv2.dnn.readNetFromDarknet()` | `readNetFromDarknet(cfgFile, darknetModel)` | 加载 Darknet 模型 |
| `cv2.dnn.readNet()` | `readNet(model[, config[, framework]])` | 通用加载 |
| `cv2.dnn.blobFromImage()` | `blobFromImage(image, scalefactor, size, mean, swapRB, crop)` | 图像预处理 |
| `net.setInput()` | `setInput(blob)` | 设置输入 |
| `net.forward()` | `forward([outputName])` | 前向推理 |

## 五、代码实战

### 5.1 图像分类（Caffe 模型）

```python
import cv2
import numpy as np

# 加载模型（需要下载模型文件）
prototxt = "bvlc_googlenet.prototxt"
caffemodel = "bvlc_googlenet.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

# 加载类别标签
with open("synset_words.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 预处理
img = cv2.imread("dog.jpg")
blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(224, 224),
                              mean=(104, 117, 123), swapRB=False)

# 推理
net.setInput(blob)
output = net.forward()

# 解析结果
class_id = np.argmax(output)
confidence = output[0][class_id]
print(f"类别: {classes[class_id]}, 置信度: {confidence:.2%}")
```

### 5.2 SSD 目标检测

```python
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                                "MobileNetSSD_deploy.caffemodel")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

img = cv2.imread("street.jpg")
h, w = img.shape[:2]
blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)

net.setInput(blob)
detections = net.forward()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        class_id = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{CLASSES[class_id]}: {confidence:.2f}"
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

cv2.imshow("Detection", img)
cv2.waitKey(0)
```

### 5.3 效果对比

| 模型 | 检测速度 | 精度 | 适用场景 |
|------|---------|------|----------|
| SSD MobileNet | 快 | 中 | 实时检测 |
| SSD ResNet | 中 | 高 | 精度要求高 |
| YOLO (Darknet) | 快 | 高 | 实时检测（推荐） |

## 六、调试经验与避坑指南

| 问题现象 | 原因 | 解决方案 |
|----------|------|----------|
| 模型加载失败 | 文件路径错误或格式不匹配 | 检查 prototxt/caffemodel 路径 |
| 推理结果全错 | 均值/归一化参数不对 | 确认训练时的预处理参数 |
| 输出为空 | 网络层名称错误 | 检查 forward 的 outputName |
| 检测框位置偏移 | 坐标缩放没做 | 将归一化坐标乘回原图尺寸 |

### 常见坑点
- **坑点1**: `blobFromImage` 的 mean 参数是 BGR 顺序，不是 RGB
- **坑点2**: `swapRB=True` 会将 BGR 转成 RGB，具体看模型训练时用的什么顺序
- **坑点3**: 需要 `opencv-contrib-python` 才能使用 DNN 模块

## 七、本章小结

- **必掌握**: readNet + blobFromImage + setInput + forward 四步推理流程
- **选了解**: 不同框架模型的加载方式
- **一句话总结**: DNN 模块让 OpenCV 能直接运行深度学习模型推理，无需安装 TensorFlow/PyTorch

## 八、知识连接

- [[第19章 目标追踪]] — 用 DNN 模块加载检测模型做追踪
- [[第20章 卷积原理与操作]] — 理解 DNN 内部原理
- [[第21章 疲劳检测]] — DNN 模块的实际应用

## 九、课后实践

- [ ] 基础练习: 下载 MobileNet SSD 模型，检测图片中的常见物体
- [ ] 进阶挑战: 用 ONNX 格式加载一个自定义模型并推理