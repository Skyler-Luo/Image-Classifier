# 🖼️ Image-Classifier

<p align="center">
  <b>一个功能丰富、易于使用的 PyTorch 图像分类框架</b>
</p>

<p align="center">
  支持 20+ 种模型架构 | 🔥 知识蒸馏 | 🎨 丰富数据增强 | ⚡ 混合精度训练 | 📦 多格式导出
</p>

## 目录

- [✨ 特性亮点](#-特性亮点)
- [🏗️ 支持的模型](#️-支持的模型)
- [📁 项目结构](#-项目结构)
- [🚀 快速开始](#-快速开始)
- [⚙️ 参数详解](#️-参数详解)
- [🔧 高级功能](#-高级功能)
- [📊 训练监控与可视化](#-训练监控与可视化)
- [📦 模型导出](#-模型导出)
- [🔬 模型评估](#-模型评估)
- [💡 最佳实践](#-最佳实践)
- [📋 依赖环境](#-依赖环境)
- [📄 License](#-license)

## ✨ 特性亮点

| 特性 | 说明 |
|------|------|
| 🧠 **模型支持** | 20+ 种模型架构，支持 ImageNet 预训练权重与自定义权重加载 |
| 🔥 **知识蒸馏** | SoftTarget、MGD、SP、AT 等多种蒸馏方法 |
| 🎨 **数据增强** | 基础增强、混合增强（MixUp、CutMix）、TTA 测试时增强 |
| ⚡ **训练技巧** | AMP、EMA、Gradient Accumulation、Early Stop、R-Drop、Label Smoothing 等训练技巧 |
| 🎯 **损失函数** | CrossEntropy、FocalLoss、PolyLoss 等多种损失函数 |
| 📊 **可视化**   | TensorBoard、Grad-CAM、t-SNE |
| 📦 **模型导出** | TorchScript、ONNX、TensorRT 多格式支持 |

## 🏗️ 支持的模型

### 📊 模型列表

| 模型名称 | 可选版本 |
|----------|----------|
| ResNet | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` |
| ResNeXt | `resnext50`, `resnext101` |
| Wide ResNet | `wide_resnet50`, `wide_resnet101` |
| ResNeSt | `resnest50`, `resnest101`, `resnest200`, `resnest269` |
| MobileNetV2 | `mobilenetv2` |
| MobileNetV3 | `mobilenetv3_large`, `mobilenetv3_small` |
| ShuffleNetV2 | `shufflenet_v2_x0_5`, `shufflenet_v2_x1_0` |
| GhostNet | `ghostnet` |
| RepGhost | `repghostnet_*` |
| EfficientNet | `efficientnet_b0` ~ `efficientnet_b7` |
| EfficientNetV2 | `efficientnet_v2_s`, `efficientnet_v2_m`, `efficientnet_v2_l` |
| ConvNeXt | `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`, `convnext_xlarge` |
| RepVGG | `RepVGG-A0`, `RepVGG-A1`, `RepVGG-A2`, `RepVGG-B0`, `RepVGG-B1`, `RepVGG-B2`, `RepVGG-B3` |
| VGG | `vgg11`, `vgg13`, `vgg16`, `vgg19` (含 `_bn` 变体) |
| DenseNet | `densenet121`, `densenet161`, `densenet169`, `densenet201` |
| DPN | `dpn68`, `dpn98`, `dpn131` |
| CSPNet | `cspresnet50`, `cspdarknet53`, `darknet53` |
| VoVNet | `vovnet39`, `vovnet57` |
| MNASNet | `mnasnet` |
| Sequencer2D | `sequencer2d_s`, `sequencer2d_m`, `sequencer2d_l` |

> [!NOTE]
> ViT相关模型待集成

## 📁 项目结构

```
Image-Classifier/
├── 📁 config/
│   ├── config.py           # 默认训练配置
│   └── sgd_config.py       # SGD 优化器配置示例
│
├── 📁 model/
│   ├── resnet.py           # ResNet 系列
│   ├── efficientnetv2.py   # EfficientNet 系列
│   ├── mobilenetv2.py      # MobileNetV2
│   ├── mobilenetv3.py      # MobileNetV3
│   ├── convnext.py         # ConvNeXt
│   ├── densenet.py         # DenseNet
│   ├── vgg.py              # VGG
│   ├── ghostnet.py         # GhostNet
│   ├── repvgg.py           # RepVGG
│   ├── shufflenetv2.py     # ShuffleNetV2
│   └── ...                 # 更多模型
│
├── 📁 utils/
│   ├── utils.py            # 通用工具函数
│   ├── utils_aug.py        # 数据增强
│   ├── utils_loss.py       # 损失函数
│   ├── utils_fit.py        # 训练循环
│   ├── utils_distill.py    # 知识蒸馏
│   └── utils_model.py      # 模型选择器
│
├── main.py                 # 训练主程序
├── predict.py              # 单张/批量图片预测
├── metrics.py              # 模型评估与指标计算
├── export.py               # 模型导出 (ONNX/TorchScript/TensorRT)
├── processing.py           # 数据集划分工具
└── requirements.txt        # 依赖列表
```

## 🚀 快速开始

### 1️⃣ 环境安装

```bash
# 克隆项目
git clone https://github.com/your-repo/Image-Classifier.git
cd Image-Classifier

# 安装 PyTorch (https://pytorch.org/get-started/previous-versions/)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# 安装其余依赖
pip install -r requirements.txt
```

### 2️⃣ 准备数据集

数据集需要按照以下示例结构组织：

```
dataset/
├── 📁 train/                    # 训练集
│   ├── 📁 cat/                  # 类别1
│   │   ├── cat_001.jpg
│   │   ├── cat_002.jpg
│   │   └── ...
│   ├── 📁 dog/                  # 类别2
│   │   ├── dog_001.jpg
│   │   └── ...
│   └── 📁 bird/                 # 类别3
│       └── ...
│
├── 📁 val/                      # 验证集 (结构同上)
│   ├── 📁 cat/
│   ├── 📁 dog/
│   └── 📁 bird/
│
├── 📁 test/                     # 测试集 (结构同上)
│   ├── 📁 cat/
│   ├── 📁 dog/
│   └── 📁 bird/
│
└── 📄 label.txt                 # 类别标签文件
```

**label.txt 格式：**
```
cat
dog
bird
```

#### 使用数据集划分工具

如果你只有一个包含所有图片的文件夹，可以使用 `processing.py` 自动划分：

```bash
python processing.py --data_path dataset/train --val_size 0.1 --test_size 0.2
```

这将自动：
- ✅ 生成 `label.txt`
- ✅ 按比例划分训练集、验证集、测试集
- ✅ 重命名类别文件夹为数字编号

### 3️⃣ 开始训练

```bash
python main.py \
    --model_name resnet18 \
    --pretrained \
    --device 0 \
    --batch_size 32 \
    --epoch 100 \
    --loss FocalLoss \
    --optimizer AdamW \
    --lr 1e-3 \
    --Augment RandAugment \
    --label_smoothing 01 \
    --mixup cutmix \
    --label_smoothing 0.1 \
    --amp \
    --ema \
    --warmup
```

### 4️⃣ 推理预测

```bash
# 单张图片预测
python predict.py --source image.jpg --save_path runs/exp

# 批量预测
python predict.py --source images_folder/ --save_path runs/exp

# 使用 Grad-CAM 可视化
python predict.py --source image.jpg --save_path runs/exp --cam_visual --cam_type GradCAM --device cpu
```

### 5️⃣ 模型评估

```bash
# 在测试集上评估
python metrics.py --test_path dataset/test --save_path runs/exp --task test

# 测试推理速度 (FPS)
python metrics.py --save_path runs/exp --task fps --batch_size 32
```

## ⚙️ 参数详解

### 📌 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_name` | str | `resnet18` | 模型名称，见支持的模型列表 |
| `--pretrained` | flag | `False` | 使用 ImageNet 预训练权重 |
| `--weight` | str | `''` | 自定义权重文件路径 |
| `--config` | str | `config/config.py` | 配置文件路径 |
| `--device` | str | `''` | GPU 设备，如 `0` 或 `0,1` 或 `cpu` |

### 📂 数据参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train_path` | str | `dataset/train` | 训练集路径 |
| `--val_path` | str | `dataset/val` | 验证集路径 |
| `--test_path` | str | `dataset/test` | 测试集路径 |
| `--label_path` | str | `dataset/label.txt` | 类别标签文件 |
| `--image_size` | int | `224` | 输入图像尺寸 |
| `--image_channel` | int | `3` | 图像通道数 |
| `--workers` | int | `4` | DataLoader 工作进程数 |
| `--batch_size` | int | `64` | 批次大小 (`-1` 自动计算最优值) |

### 🎯 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epoch` | int | `100` | 训练轮数 |
| `--lr` | float | `1e-3` | 初始学习率 |
| `--optimizer` | str | `AdamW` | 优化器：`SGD`, `AdamW`, `RMSProp` |
| `--weight_decay` | float | `5e-4` | 权重衰减 |
| `--momentum` | float | `0.9` | 动量 (SGD/RMSProp) |
| `--accumulate` | int | `1` | 梯度累积步数 |
| `--grad_clip` | float | `0.0` | 梯度裁剪阈值 (`0` 禁用) |
| `--save_path` | str | `runs/exp` | 模型和日志保存路径 |
| `--resume` | flag | `False` | 从断点继续训练 |

### 📉 损失函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--loss` | str | `CrossEntropyLoss` | 损失函数：`CrossEntropyLoss`, `FocalLoss`, `PolyLoss` |
| `--label_smoothing` | float | `0.1` | 标签平滑系数 |
| `--class_balance` | flag | `False` | 启用类别平衡权重 |

### 🎨 数据增强参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--Augment` | str | `none` | 自动增强策略：`RandAugment`, `AutoAugment`, `TrivialAugmentWide`, `AugMix`, `none` |
| `--mixup` | str | `none` | 混合增强：`mixup`, `cutmix`, `none` |
| `--imagenet_meanstd` | flag | `False` | 使用 ImageNet 均值和标准差 |
| `--test_tta` | flag | `False` | 测试时增强 (TenCrop) |

### ⚡ 训练技巧参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--amp` | flag | `False` | 混合精度训练 (FP16) |
| `--ema` | flag | `False` | 指数移动平均 |
| `--warmup` | flag | `False` | 学习率预热 |
| `--warmup_ratios` | float | `0.05` | 预热轮数比例 |
| `--warmup_minlr` | float | `1e-6` | 预热最小学习率 |
| `--rdrop` | flag | `False` | R-Drop 正则化 |
| `--freeze_backbone` | flag | `False` | 冻结骨干网络 |
| `--freeze_epochs` | int | `0` | 冻结轮数 (`0` 表示全程冻结) |
| `--patience` | int | `30` | 早停耐心值 |
| `--metric` | str | `acc` | 最佳模型保存指标：`loss`, `acc`, `mean_acc`, `f1` |

### 🔥 知识蒸馏参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--kd` | flag | `False` | 启用知识蒸馏 |
| `--teacher_path` | str | `''` | 教师模型路径 |
| `--kd_method` | str | `SoftTarget` | 蒸馏方法：`SoftTarget`, `MGD`, `SP`, `AT` |
| `--kd_ratio` | float | `0.7` | 蒸馏损失权重 |

## 🔧 高级功能

### 知识蒸馏详解

知识蒸馏可以将大模型的知识迁移到小模型，实现模型压缩。

#### 支持的蒸馏方法

| 方法 | 论文 | 说明 |
|------|------|------|
| **SoftTarget** | Hinton et al. | 经典软标签蒸馏，使用教师模型的软化输出 |
| **MGD** | Masked Generative Distillation | 基于掩码的特征蒸馏 |
| **SP** | Similarity-Preserving | 保持样本间相似性关系 |
| **AT** | Attention Transfer | 注意力图迁移 |

#### 蒸馏训练流程

```bash
# Step 1: 训练教师模型 (大模型)
python main.py \
    --model_name resnet101 \
    --pretrained \
    --train_path dataset/train \
    --val_path dataset/val \
    --epoch 100 \
    --save_path runs/teacher

# Step 2: 蒸馏训练学生模型 (小模型)
python main.py \
    --model_name mobilenetv3_small \
    --pretrained \
    --train_path dataset/train \
    --val_path dataset/val \
    --kd \
    --teacher_path runs/teacher \
    --kd_method SoftTarget \
    --kd_ratio 0.7 \
    --save_path runs/student
```

### 自定义数据增强

编辑 `config/config.py` 添加自定义增强：

```python
import torchvision.transforms as transforms
from utils.utils_aug import Create_Albumentations_From_Name, CutOut

class Config:
    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    lr_scheduler_params = {
        'T_max': 10,
        'eta_min': 1e-6
    }
    
    # 随机种子
    random_seed = 42
    
    # 训练批次可视化数量
    plot_train_batch_count = 5
    
    # 自定义数据增强
    custom_augment = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(45),
        CutOut(n_holes=4, length=16),  # CutOut 增强
        # 使用 Albumentations 增强
        Create_Albumentations_From_Name('RandomGridShuffle', grid=(4, 4)),
        Create_Albumentations_From_Name('PixelDropout', p=0.1),
    ])
```

### Grad-CAM 可视化

支持 8 种 CAM 可视化方法：

| 方法 | 说明 |
|------|------|
| `GradCAM` | 经典梯度加权类激活映射 |
| `GradCAMPlusPlus` | 改进版 GradCAM |
| `HiResCAM` | 高分辨率 CAM |
| `ScoreCAM` | 基于分数的 CAM |
| `AblationCAM` | 消融 CAM |
| `XGradCAM` | 扩展 GradCAM |
| `EigenCAM` | 特征值 CAM |
| `FullGrad` | 完整梯度 CAM |

```bash
# 使用不同的 CAM 方法
python predict.py \
    --source image.jpg \
    --save_path runs/exp \
    --cam_visual \
    --cam_type GradCAMPlusPlus \
    --device cpu
```

> [!NOTE]
> CAM 可视化仅支持 CPU 和 FP32 模式

## 📊 训练监控与可视化

### TensorBoard

训练过程中自动记录到 TensorBoard：

```bash
# 启动 TensorBoard
tensorboard --logdir=runs/exp/tensorboard

# 浏览器访问
# http://localhost:6006
```

| 指标 | 说明 |
|------|------|
| Loss | 训练/验证损失 |
| Accuracy | 训练/验证准确率 |
| Mean Accuracy | 训练/验证平均类别准确率 |
| F1 Score | 训练/验证 F1 分数 |
| Learning Rate | 学习率变化 |
| KD Loss | 知识蒸馏损失（蒸馏时） |

### 输出文件

训练完成后，`save_path` 目录包含：

```
runs/exp/
├── best.pt                  # 最佳模型权重
├── last.pt                  # 最后一轮模型权重
├── train.log                # 训练日志 (CSV 格式)
├── param.yaml               # 训练参数配置
├── preprocess.transforms    # 数据预处理参数
├── main.py                  # 训练脚本备份
├── conafig.py                # 配置文件备份
├── train_batch1.png         # 训练批次可视化
├── iterative_curve.png      # Loss/Accuracy 曲线
├── lesarning_rate_curve.png  # 学习率曲线
└── tensorboard/             # TensorBoard 日志
```

## 📦 模型导出

### 导出为 ONNX

```bash
# 基础导出
python export.py --save_path runs/exp --export onnx

# 简化 ONNX 模型
python export.py --save_path runs/exp --export onnx --simplify

# 动态 batch size
python export.py --save_path runs/exp --export onnx --dynamic
```

### 导出为 TorchScript

```bash
python export.py --save_path runs/exp --export torchscript
```

### 导出为 TensorRT

```bash
# FP32 精度
python export.py --save_path runs/exp --export tensorrt --device 0

# FP16 精度 (更快)
python export.py --save_path runs/exp --export tensorrt --device 0 --half
```

### 导出参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--save_path` | str | `runs/exp` | 模型路径 |
| `--export` | str | `torchscript` | 导出格式：`onnx`, `torchscript`, `tensorrt` |
| `--image_size` | int | `224` | 输入图像尺寸 |
| `--batch_size` | int | `1` | 批次大小 |
| `--dynamic` | flag | `False` | 动态 batch size (ONNX) |
| `--simplify` | flag | `False` | 简化 ONNX 模型 |
| `--half` | flag | `False` | FP16 精度 (TensorRT) |

## 🔬 模型评估

### 评估指标

`metrics.py` 提供全面的模型评估：

```bash
# 在测试集上评估
python metrics.py \
    --test_path dataset/test \
    --save_path runs/exp \
    --task test \
    --batch_size 64

# 使用 TTA 提升精度
python metrics.py \
    --test_path dataset/test \
    --save_path runs/exp \
    --task test \
    --test_tta

# 可视化预测结果
python metrics.py \
    --test_path dataset/test \
    --save_path runs/exp \
    --task test \
    --visual

# t-SNE 特征可视化
python metrics.py \
    --test_path dataset/test \
    --save_path runs/exp \
    --task test \
    --tsne
```

### 输出指标

| 类型 | 指标 |
|------|------|
| Per-Class | Precision、Recall、F0.5/F1/F2、AUC、AUPR、Accuracy |
| Overall | Accuracy、MPA、Kappa、Micro/Macro Precision、Micro/Macro Recall、Micro/Macro F1 |

### FPS 测试

```bash
# 测试推理速度
python metrics.py \
    --save_path runs/exp \
    --task fps \
    --batch_size 32 \
    --device 0
```

### 安装命令

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装 ONNX 导出支持
pip install onnx onnx-simplifier onnxruntime

# 安装 TensorRT 支持 (需要 NVIDIA GPU)
pip install nvidia-pyindex nvidia-tensorrt
```

## 📄 License

本项目采用 MIT License 开源协议。

## 🙏 致谢

感谢以下开源项目：


---

<p align="center">
  ⭐ 如果这个项目对你有帮助，请给一个 Star！⭐
</p>
