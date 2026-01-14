# PyTorch CIFAR10 训练系统
基于 PyTorch 构建的工程化 CIFAR10 图像分类训练项目，
实现完整的模型训练、评估与实验管理流程。

## 项目特点
- 标准化深度学习项目结构（data / models / checkpoints / logs）
- 支持数据增强与 batch 训练（RandomCrop、RandomHorizontalFlip）
- 集成 TensorBoard 进行训练过程可视化
- 实现 checkpoint 与 resume 机制，支持断点恢复训练
- 通过 argparse 支持命令行参数配置，提升实验灵活性与可复现性
- 支持最优模型与周期性权重自动保存

### 训练模型
```bash
python train.py --epochs 30 --batch_size 64

### **断点恢复**


