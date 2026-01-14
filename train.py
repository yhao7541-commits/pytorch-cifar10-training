import os
import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from models import Tudui
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)                  # Python 自带随机数
    np.random.seed(seed)               # NumPy 随机数
    torch.manual_seed(seed)            # CPU 上的 PyTorch 随机数
    torch.cuda.manual_seed(seed)       # 当前 GPU
    torch.cuda.manual_seed_all(seed)   # 所有 GPU

    # 让 CUDA 卷积结果可复现（可能会稍微慢一点）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR10 Training")  # 解CIFAR10 Training是一个命令行参数解析器，用于解析命令行参数。

    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="batch 大小")  # 默认值64
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="L2 正则")
    parser.add_argument("--resume", action="store_true", help="是否从断点恢复训练")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")

    return parser.parse_args()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()

    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc


def main():
    set_seed(0)
    # ===== 路径与设备 =====
    project_root = os.path.dirname(os.path.abspath(__file__))# 获取当前脚本所在目录的父目录，D:\pystudy\Project1
    data_dir = os.path.join(project_root, "data")
    ckpt_dir = os.path.join(project_root, "checkpoints")
    log_dir = os.path.join(project_root, "logs", "run1")
    os.makedirs(data_dir, exist_ok=True)# 不存在则创建data目录
    os.makedirs(ckpt_dir, exist_ok=True)# 不存在则创建checkpoints目录
    os.makedirs(log_dir, exist_ok=True)# 不存在则创建logs/run1目录

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir)


    # ===== 数据集 =====
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

    args = parse_args()
    set_seed(args.seed)
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=False)
    test_set  = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=test_transform, download=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=0, drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    print(f"训练集数量：{len(train_set)}")
    print(f"测试集数量：{len(test_set)}")

    # ===== 模型/损失/优化器 =====
    model = Tudui().to(device)# 模型实例化并移动到指定设备
    criterion = nn.CrossEntropyLoss()# 损失函数实例化
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    best_acc = 0.0
    start_epoch = 0
    ckpt_path = os.path.join(ckpt_dir, "last_checkpoint.pth")

    if args.resume and os.path.exists(ckpt_path):
        print("发现 checkpoint，正在恢复训练...")
        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]

        print(f"从 epoch {start_epoch} 继续训练")
    else:
        print("未发现 checkpoint，从头开始训练")

    # ===== 训练循环 =====
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar("train/loss", avg_train_loss, epoch)
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}")

        # ===== 测试 =====
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/acc", test_acc, epoch)
        print(f"Epoch {epoch}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")



        # ===== 保存策略：best + 每N轮 =====
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
            # 每一轮都保存“最近一次”的状态

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"tudui_epoch_{epoch+1}.pth"))
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_acc": best_acc
        }, os.path.join(ckpt_dir, "last_checkpoint.pth"))
    writer.close()
    print(f"训练结束，best_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
