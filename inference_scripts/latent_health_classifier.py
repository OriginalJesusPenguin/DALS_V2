import argparse
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover - optional dependency
    roc_auc_score = None


DEFAULT_CHECKPOINT = (
    '/home/ralbe/DALS/mesh_autodecoder/models/'
    'MeshDecoderTrainer_2025-11-06_12-00-26.ckpt'
)
DEFAULT_TEST_LATENT_DIR = (
    '/home/ralbe/DALS/mesh_autodecoder/inference_results/'
    'meshes_MeshDecoderTrainer_2025-11-06_12-00-26/latents'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a simple MLP to classify latent codes into healthy/cirrhotic.'
    )
    parser.add_argument(
        '--checkpoint',
        default=DEFAULT_CHECKPOINT,
        help='Path to MeshDecoder checkpoint containing latent vectors.',
    )
    parser.add_argument(
        '--test-latent-dir',
        default=DEFAULT_TEST_LATENT_DIR,
        help='Directory with .pt latent tensors for testing.',
    )
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden layer width.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs.')
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Fraction of training data to reserve for validation (stratified).',
    )
    parser.add_argument(
        '--learning-rate', type=float, default=1e-3, help='Optimizer learning rate.'
    )
    parser.add_argument('--seed', type=int, default=1337, help='Random seed.')
    parser.add_argument(
        '--output-dir',
        default='/home/ralbe/DALS/mesh_autodecoder/inference_results/latent_classifier_outputs',
        help='Directory to store generated figures.',
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_from_name(filename: str) -> int:
    name = os.path.basename(filename).lower()
    if 'healthy' in name:
        return 0
    if 'cirrhotic' in name:
        return 1
    raise ValueError(f'Unable to infer label from filename: {filename}')


def load_training_latents(
    checkpoint_path: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
    )

    if 'latent_vectors' not in checkpoint or 'train_filenames' not in checkpoint:
        raise KeyError(
            'Checkpoint missing required keys: latent_vectors or train_filenames.'
        )

    latent_weight = checkpoint['latent_vectors'].weight.detach().cpu()
    train_filenames: Sequence[str] = checkpoint['train_filenames']

    if len(latent_weight) != len(train_filenames):
        raise ValueError(
            f'Latent tensor length ({len(latent_weight)}) does not match '
            f'number of filenames ({len(train_filenames)}).'
        )

    labels = torch.tensor([label_from_name(name) for name in train_filenames])
    latents = latent_weight.float()
    return latents, labels.float(), list(train_filenames)


def stratified_split(
    latents: torch.Tensor,
    labels: torch.Tensor,
    val_ratio: float,
    seed: int,
) -> Tuple[TensorDataset, TensorDataset]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError('val_ratio must be in (0, 1).')

    indices: DefaultDict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels.int().tolist()):
        indices[label].append(idx)

    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for cls_label, cls_indices in indices.items():
        if not cls_indices:
            continue
        rng.shuffle(cls_indices)
        val_count = max(1, int(len(cls_indices) * val_ratio)) if len(cls_indices) > 1 else 0
        val_indices.extend(cls_indices[:val_count])
        train_indices.extend(cls_indices[val_count:])

    if not train_indices or not val_indices:
        raise RuntimeError(
            'Stratified split failed; ensure each class has sufficient samples.'
        )

    train_latents = latents[train_indices]
    train_labels = labels[train_indices]
    val_latents = latents[val_indices]
    val_labels = labels[val_indices]

    return (
        TensorDataset(train_latents, train_labels),
        TensorDataset(val_latents, val_labels),
    )


def load_test_latents(
    latent_dir: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    latents: List[torch.Tensor] = []
    labels: List[int] = []
    filenames: List[str] = []

    for path in sorted(Path(latent_dir).glob('*_latent.pt')):
        data = torch.load(
            path,
            map_location='cpu',
        )
        if isinstance(data, torch.Tensor):
            latent = data
        elif isinstance(data, dict):
            if 'latent' in data:
                latent = data['latent']
            elif 'latent_vector' in data:
                latent = data['latent_vector']
            else:
                raise ValueError(f'Unsupported latent dict keys in {path}')
        else:
            raise TypeError(f'Unsupported latent format in {path}: {type(data)}')

        latent = latent.squeeze()
        if latent.ndim != 1:
            raise ValueError(f'Latent tensor in {path} has unexpected shape {latent.shape}')

        latents.append(latent.float())
        labels.append(label_from_name(path.name))
        filenames.append(path.name)

    if not latents:
        raise RuntimeError(f'No latent files found in {latent_dir}')

    return torch.stack(latents), torch.tensor(labels, dtype=torch.float32), filenames


class LatentClassifier(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_value: float,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    labels_int = labels.long()

    tp = torch.sum((preds == 1) & (labels_int == 1)).item()
    tn = torch.sum((preds == 0) & (labels_int == 0)).item()
    fp = torch.sum((preds == 1) & (labels_int == 0)).item()
    fn = torch.sum((preds == 0) & (labels_int == 1)).item()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    metrics = {
        'loss': loss_value,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }

    if roc_auc_score is not None and len(torch.unique(labels_int)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(labels_int.numpy(), probs.numpy())
        except ValueError:
            metrics['roc_auc'] = float('nan')
    else:
        metrics['roc_auc'] = float('nan')

    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for latents, labels in dataloader:
            latents = latents.to(device)
            labels = labels.to(device)
            logits = model(latents)
            loss = criterion(logits, labels)

            batch_size = latents.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    stacked_logits = torch.cat(all_logits, dim=0)
    stacked_labels = torch.cat(all_labels, dim=0)
    avg_loss = total_loss / total_samples if total_samples else float('nan')
    metrics = compute_metrics(stacked_logits, stacked_labels, avg_loss)
    probs = torch.sigmoid(stacked_logits)
    preds = (probs >= 0.5).long()
    metrics['labels'] = stacked_labels.numpy()
    metrics['preds'] = preds.numpy()
    metrics['probs'] = probs.numpy()
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> None:
    best_state = None
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_samples = 0

        for latents, labels in train_loader:
            latents = latents.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(latents)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = latents.size(0)
            running_loss += loss.item() * batch_size
            running_samples += batch_size

        train_loss = running_loss / running_samples if running_samples else float('nan')
        val_metrics = evaluate(model, val_loader, criterion, device)

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f'Epoch {epoch:03d} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val Loss: {val_metrics["loss"]:.4f} | '
            f'Val Acc: {val_metrics["accuracy"]:.4f} | '
            f'Val F1: {val_metrics["f1"]:.4f}'
        )

    if best_state is not None:
        model.load_state_dict(best_state)


def summarize_dataset(name: str, labels: torch.Tensor) -> None:
    unique, counts = torch.unique(labels.int(), return_counts=True)
    summary = {int(k.item()): int(v.item()) for k, v in zip(unique, counts)}
    print(f'{name} class distribution: {summary}')


def plot_confusion_matrices(
    metrics_by_split: Dict[str, Dict[str, Any]], output_dir: Path
) -> Path:
    splits = list(metrics_by_split.keys())
    fig, axes = plt.subplots(
        1,
        len(splits),
        figsize=(5 * len(splits), 4),
        squeeze=False,
    )
    axes = axes[0]

    for ax, split in zip(axes, splits):
        metrics = metrics_by_split[split]
        cm = np.array(
            [
                [metrics['tn'], metrics['fp']],
                [metrics['fn'], metrics['tp']],
            ]
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            ax=ax,
            xticklabels=['Pred Healthy (0)', 'Pred Cirrhotic (1)'],
            yticklabels=['True Healthy (0)', 'True Cirrhotic (1)'],
        )
        ax.set_title(f'{split} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    output_path = output_dir / 'confusion_matrices.png'
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Confusion matrices saved to {output_path}')
    return output_path


def compute_feature_importance(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    feature_dim = model.network[0].in_features
    importances = torch.zeros(feature_dim, device=device)
    total_samples = 0

    for latents, _ in dataloader:
        latents = latents.to(device)
        latents = latents.clone().detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)
        logits = model(latents)
        probs = torch.sigmoid(logits)
        probs.sum().backward()

        grad = latents.grad.detach().abs().sum(dim=0)
        importances += grad
        total_samples += latents.size(0)

    if total_samples > 0:
        importances /= total_samples

    return importances.cpu().numpy()


def plot_feature_importance(
    importances: np.ndarray,
    output_dir: Path,
    top_k: int = 20,
) -> Path:
    top_k = min(top_k, len(importances))
    top_indices = np.argsort(importances)[::-1][:top_k]
    top_values = importances[top_indices]

    fig, ax = plt.subplots(figsize=(max(10, top_k * 0.6), 5))
    ax.bar(range(top_k), top_values, color='tab:blue')
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([str(idx) for idx in top_indices], rotation=45, ha='right')
    ax.set_ylabel('Mean |∂σ(logit)/∂latent_i|')
    ax.set_xlabel('Latent index')
    ax.set_title(f'Top {top_k} latent dimensions by importance (test set)')
    plt.tight_layout()

    output_path = output_dir / 'latent_feature_importance.png'
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Feature importance plot saved to {output_path}')
    return output_path


def report_top_features(importances: np.ndarray, top_k: int = 10) -> None:
    top_indices = np.argsort(importances)[::-1][:top_k]
    print(f'\nTop {top_k} latent dimensions by test-set importance:')
    for rank, idx in enumerate(top_indices, start=1):
        print(f'  {rank:02d}. Latent {idx:3d}: importance={importances[idx]:.6f}')


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_latents, train_labels, train_filenames = load_training_latents(args.checkpoint)
    summarize_dataset('Full training', train_labels)

    train_dataset, val_dataset = stratified_split(
        train_latents, train_labels, args.val_ratio, args.seed
    )

    summarize_dataset('Train subset', train_dataset.tensors[1])
    summarize_dataset('Validation subset', val_dataset.tensors[1])

    train_loader = create_dataloader(train_dataset, args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, args.batch_size, shuffle=False)

    input_dim = train_latents.size(1)
    model = LatentClassifier(in_features=input_dim, hidden_dim=args.hidden_dim).to(device)

    train_label_tensor = train_dataset.tensors[1]
    class_counts = torch.bincount(train_label_tensor.long())
    if len(class_counts) < 2 or class_counts[1] == 0:
        raise RuntimeError('Positive class absent from training data; cannot train classifier.')

    pos_weight = class_counts[0] / class_counts[1]
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        args.epochs,
    )

    test_latents, test_labels, test_filenames = load_test_latents(args.test_latent_dir)
    summarize_dataset('Test set', test_labels)

    train_metrics = evaluate(model, train_loader, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_dataset = TensorDataset(test_latents, test_labels)
    test_loader = create_dataloader(test_dataset, args.batch_size, shuffle=False)
    test_metrics = evaluate(model, test_loader, criterion, device)

    metrics_by_split = {
        'Train': train_metrics,
        'Validation': val_metrics,
        'Test': test_metrics,
    }

    def format_metrics(split: str, metrics: Dict[str, float]) -> None:
        print(
            f'\n{split} metrics:\n'
            f'  Loss:      {metrics["loss"]:.4f}\n'
            f'  Accuracy:  {metrics["accuracy"]:.4f}\n'
            f'  Precision: {metrics["precision"]:.4f}\n'
            f'  Recall:    {metrics["recall"]:.4f}\n'
            f'  F1 score:  {metrics["f1"]:.4f}\n'
            f'  ROC AUC:   {metrics["roc_auc"]:.4f}\n'
            f'  Confusion matrix: '
            f'TP={metrics["tp"]}, TN={metrics["tn"]}, FP={metrics["fp"]}, FN={metrics["fn"]}'
        )

    format_metrics('Training', train_metrics)
    format_metrics('Validation', val_metrics)
    format_metrics('Test', test_metrics)

    plot_confusion_matrices(metrics_by_split, output_dir)

    test_importances = compute_feature_importance(model, test_loader, device)
    report_top_features(test_importances, top_k=10)
    plot_feature_importance(test_importances, output_dir, top_k=20)


if __name__ == '__main__':
    main()

