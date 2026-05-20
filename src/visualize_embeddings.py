import argparse
import os

import matplotlib
matplotlib.use('Agg')  # Ensure non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from umap import UMAP
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose

from src.models import SimCLR


@torch.no_grad()
def extract_embeddings(encoder, loader, device):
    """Extract embeddings from the encoder."""
    encoder.eval()
    embeddings, labels = [], []

    for x, y in loader:
        x = x.to(device)
        h = encoder(x).cpu().numpy()
        embeddings.append(h)
        labels.append(y.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels


def visualize_with_tsne(embeddings, labels, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="tab10", s=5, alpha=0.7
    )
    plt.colorbar(scatter, ticks=range(10))
    plt.title("t-SNE Visualization of Embeddings")
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE visualization saved to {save_path}")


def visualize_with_umap(embeddings, labels, save_path):
    umap = UMAP(n_components=2, random_state=42)
    embeddings_2d = umap.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="tab10", s=5, alpha=0.7
    )
    plt.colorbar(scatter, ticks=range(10))
    plt.title("UMAP Visualization of Embeddings")
    plt.savefig(save_path)
    plt.close()
    print(f"UMAP visualization saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./results/embeddings")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load encoder
    model = SimCLR(backbone="resnet18", proj_dim=128)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    encoder = model.encoder.to(device)

    # Data
    transform = Compose(
        [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
    )
    test_ds = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Extract embeddings
    embeddings, labels = extract_embeddings(encoder, test_dl, device)

    # Visualize with t-SNE
    visualize_with_tsne(
        embeddings, labels, save_path=os.path.join(args.output_dir, "tsne.png")
    )

    # Visualize with UMAP
    visualize_with_umap(
        embeddings, labels, save_path=os.path.join(args.output_dir, "umap.png")
    )

if __name__ == "__main__":
    main()