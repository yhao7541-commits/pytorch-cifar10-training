# PyTorch CIFAR10 训练系统

一个基于 PyTorch 的工程化 CIFAR10 图像分类训练项目，
实现了完整的模型训练、评估、日志记录与参数管理流程。

## 项目简介

本项目基于 PyTorch 框架，构建了一个规范、可复现的图像分类训练系统，
在 CIFAR10 数据集上完成模型训练与测试流程。
项目采用标准工程结构，支持数据增强、断点恢复训练、命令行参数配置以及 TensorBoard 可视化，
适合作为深度学习工程与大模型方向的基础项目。

---

## 项目特点

- 标准化项目结构（data / models / checkpoints / logs）
- 使用 Dataset + DataLoader 构建高效数据加载流程
- 引入数据增强策略（RandomCrop、RandomHorizontalFlip、Normalize）
- 支持 TensorBoard 训练过程可视化
- 实现模型参数与训练状态的断点保存与恢复（checkpoint + resume）
- 通过 argparse 支持命令行参数配置，提升实验灵活性与可复现性
- 支持最优模型自动保存与周期性权重保存
- 引入随机种子控制，保证实验结果可复现

## 目录结构

```text
.
├── data/                 # 数据集目录（本地生成，不上传）
├── models/               # 模型结构定义
├── checkpoints/          # 模型权重与训练断点（本地生成）
├── train.py              # 训练脚本
├── test.py               # 测试脚本
├── README.md
└── LICENSE
```
## **环境依赖**

Python >= 3.8
PyTorch
torchvision
TensorBoard

## 模型训练

使用默认参数进行训练：python train.py
自定义训练参数示例：python train.py --epochs 30 --batch_size 64 --lr 0.01

## 断点恢复训练
若训练过程中断，可通过以下命令继续训练：python train.py --resume

## 可视化训练过程
训练过程中会自动生成 TensorBoard 日志，使用以下命令查看：tensorboard --logdir logs

## License
本项目基于 MIT License 开源，详见 LICENSE 文件。
