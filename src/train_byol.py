import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10

from src.augmentations import simclr_transform_cifar10
from src.models import SimCLR

class EMAUpdate:
    """Exponential moving average (EMA) weight update."""

    def __init__(self, alpha=0.996):
        self.alpha = alpha

    def update(self, online_params, target_params):
        for o, t in zip(online_params, target_params):
            t.data = self.alpha * t.data + (1.0 - self.alpha) * o.data

def byol_loss_fn(z_online, z_target):
    """
    BYOL Loss: Align online and target projections.

    z_online, z_target: (B, D) projections for two augmented views.
    """
    loss = 2 - 2 * F.cosine_similarity(z_online, z_target, dim=1).mean()
    return loss

class BYOL:
    """Main BYOL training class."""

    def __init__(self, backbone="resnet18", proj_dim=128):
        self.online_network = SimCLR(backbone=backbone, proj_dim=proj_dim)
        self.target_network = SimCLR(backbone=backbone, proj_dim=proj_dim)
        self.target_network.load_state_dict(self.online_network.state_dict())  # initialize target network
        self.ema = EMAUpdate()

    def update_target_network(self):
        self.ema.update(self.online_network.parameters(), self.target_network.parameters())

    def forward(self, x1, x2):
        """Forward pass; compute BYOL outputs and loss."""
        h1_online = self.online_network(x1)
        h2_online = self.online_network(x2)

        with torch.no_grad():  # target network inference only
            h1_target = self.target_network(x1)
            h2_target = self.target_network(x2)

        # Symmetric loss: online-to-target and target-to-online
        loss = byol_loss_fn(h1_online, h2_target) + byol_loss_fn(h2_online, h1_target)
        return loss

def save_checkpoint(path, model, optimizer, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.online_network.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.996)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_dir", type=str, default="./runs/byol_cifar10")
    parser.add_argument("--ckpt_path", type=str, default="./runs/byol_cifar10/checkpoints/last.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    base_transform = simclr_transform_cifar10()
    from src.augmentations import TwoCropsTransform  # Add this at the top.

    train_ds = CIFAR10(root=args.data_dir, train=True, download=True, transform=TwoCropsTransform(base_transform))

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = BYOL(backbone="resnet18", proj_dim=args.proj_dim)
    model.online_network.to(device)
    model.target_network.to(device)

    optimizer = torch.optim.AdamW(model.online_network.parameters(), lr=args.lr, weight_decay=1e-4)

    writer = SummaryWriter(log_dir=args.log_dir)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.online_network.train()
        epoch_loss = 0.0
        t0 = time.time()

        for (x1, x2), _ in train_dl:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            loss = model.forward(x1, x2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            model.update_target_network()

            epoch_loss += loss.item()
            if global_step % 50 == 0:
                writer.add_scalar("train/loss_step", loss.item(), global_step)

            global_step += 1

        avg_loss = epoch_loss / len(train_dl)
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)

        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} | loss={avg_loss:.4f} | time={dt:.1f}s")

        save_checkpoint(args.ckpt_path, model, optimizer, epoch)

    writer.close()
    print("Training finished.")
    print("Checkpoint saved to:", args.ckpt_path)
    print("TensorBoard logs in:", args.log_dir)

if __name__ == "__main__":
    main()