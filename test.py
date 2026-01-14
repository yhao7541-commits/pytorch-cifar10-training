import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from models import Tudui
from torchvision import transforms

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "data")
    ckpt_path = os.path.join(project_root, "checkpoints", "best_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    model = Tudui().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()

    print("Test loss:", total_loss / len(test_loader))
    print("Test acc:", correct / total)


if __name__ == "__main__":
    main()
