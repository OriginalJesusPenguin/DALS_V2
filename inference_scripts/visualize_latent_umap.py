#!/usr/bin/env python3
"""Visualize latent vectors using a 2D UMAP embedding."""

import argparse
import os
from glob import glob
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from umap import UMAP


LATENT_DIR_DEFAULT = "/home/ralbe/DALS/mesh_autodecoder/inference_results/meshes_MeshDecoderTrainer_2025-11-06_12-00-26/latents"
CHECKPOINT_DEFAULT = "/home/ralbe/DALS/mesh_autodecoder/models/MeshDecoderTrainer_2025-11-06_12-00-26.ckpt"


def load_inference_latents(latent_dir: str) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    pattern = os.path.join(latent_dir, "*.pt")
    latent_paths = sorted(glob(pattern))
    if not latent_paths:
        raise FileNotFoundError(f"No latent files found under {latent_dir}")

    latents = []
    labels = []
    names = []
    for path in latent_paths:
        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, torch.Tensor):
            array = tensor.detach().cpu().numpy()
        else:
            array = np.asarray(tensor)
        array = np.reshape(array, -1)
        latents.append(array)

        fname = os.path.basename(path).lower()
        if "cirrhotic" in fname:
            labels.append("cirrhotic")
        elif "healthy" in fname:
            labels.append("healthy")
        else:
            labels.append("unknown")

        names.append(os.path.basename(path).replace("_latent.pt", ""))

    return np.stack(latents), labels, latent_paths, names


def load_train_latents(checkpoint_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    latent_module = checkpoint["latent_vectors"]

    if isinstance(latent_module, torch.nn.Embedding):
        latent_tensor = latent_module.weight.detach().cpu()
    elif isinstance(latent_module, torch.nn.Parameter):
        latent_tensor = latent_module.detach().cpu()
    else:
        latent_tensor = torch.as_tensor(latent_module).detach().cpu()

    latents = latent_tensor.numpy()

    filenames = checkpoint.get("train_filenames")
    if filenames is None:
        filenames = [f"train_{idx:04d}" for idx in range(latents.shape[0])]

    names = [os.path.basename(str(name)).replace(".obj", "") for name in filenames]

    labels = []
    for name in names:
        lower = name.lower()
        if "cirrhotic" in lower:
            labels.append("cirrhotic")
        elif "healthy" in lower:
            labels.append("healthy")
        else:
            labels.append("unknown")

    return latents, labels, names


def compute_umap(latents: np.ndarray) -> np.ndarray:
    reducer = UMAP(n_components=2, init="spectral", random_state=0)
    return reducer.fit_transform(latents)


def plot_embedding(
    embedding: np.ndarray,
    labels: List[str],
    names: List[str],
    dataset_types: List[str],
    output: str,
) -> None:
    color_map = {
        "cirrhotic": "red",
        "healthy": "blue",
        "unknown": "gray",
    }

    marker_map = {
        "inference": "o",
        "train": "^",
    }

    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    for dataset in sorted(set(dataset_types)):
        indices = [idx for idx, dtype in enumerate(dataset_types) if dtype == dataset]
        if not indices:
            continue
        coords = embedding[indices]
        subset_labels = [labels[idx] for idx in indices]
        colors = [color_map.get(label, "gray") for label in subset_labels]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors,
            marker=marker_map.get(dataset, "o"),
            alpha=0.85,
            edgecolors="k",
            linewidths=0.3,
            label=f"{dataset} latents",
        )

    for (x, y), label, name, dtype in zip(embedding, labels, names, dataset_types):
        if label == "unknown":
            continue
        if dtype == "train":
            continue
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(2, 2), fontsize=7)

    color_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[lbl], markersize=7)
        for lbl in ["cirrhotic", "healthy", "unknown"]
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker=marker_map[key], color="k", linestyle="None", markersize=7)
        for key in sorted(set(dataset_types))
    ]

    legend1 = ax.legend(color_handles, ["cirrhotic", "healthy", "unknown"], title="Label", loc="upper right")
    ax.add_artist(legend1)
    ax.legend(marker_handles, [f"{key} latents" for key in sorted(set(dataset_types))], title="Source", loc="lower right")

    plt.title("UMAP projection of latent codes (train + inference)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300)
        print(f"Saved plot to {output}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot UMAP of latent vectors")
    parser.add_argument(
        "--latent-dir",
        default=LATENT_DIR_DEFAULT,
        help="Directory containing *_latent.pt files",
    )
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT_DEFAULT,
        help="Path to MeshDecoder checkpoint containing train latent vectors",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save the plot (e.g., /tmp/latents_umap.png). Shows interactively if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference_latents, inference_labels, latent_paths, inference_names = load_inference_latents(args.latent_dir)
    train_latents, train_labels, train_names = load_train_latents(args.checkpoint)

    combined_latents = np.concatenate([inference_latents, train_latents], axis=0)
    combined_labels = inference_labels + train_labels
    combined_names = inference_names + train_names
    dataset_types = ["inference"] * len(inference_labels) + ["train"] * len(train_labels)

    embedding = compute_umap(combined_latents)
    plot_embedding(embedding, combined_labels, combined_names, dataset_types, args.output)


if __name__ == "__main__":
    main()


