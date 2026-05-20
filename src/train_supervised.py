import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm


def save_checkpoint(path, model, optimizer, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )

def evaluate(model, loader, crit, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = crit(logits, y)

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_dir", type=str, default="./runs/supervised")
    parser.add_argument("--ckpt_path", type=str, default="./runs/supervised/checkpoints/last.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = Compose(
        [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
    )
    train_ds = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    test_ds = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    test_dl = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = resnet18(weights=None, num_classes=10).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    crit = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = SummaryWriter(log_dir=args.log_dir)

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = crit(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        train_loss = epoch_loss / len(train_dl)
        test_loss, test_acc = evaluate(model, test_dl, crit, device)

        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("test/loss_epoch", test_loss, epoch)
        writer.add_scalar("test/accuracy", test_acc * 100, epoch)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"test_loss={test_loss:.4f} | test_acc={test_acc * 100:.2f}% | time={dt:.1f}s"
        )

        if test_acc > best_acc or test_loss > train_loss * 1.1:
            best_acc = test_acc
            save_checkpoint(args.ckpt_path, model, optimizer, epoch)
            print(f"Checkpoint saved at epoch {epoch}.")

    writer.close()
    print("Training finished.")
    print(f"Best accuracy: {best_acc * 100:.2f}%")

if __name__ == "__main__":
    main()