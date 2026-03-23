#!/usr/bin/env python3
"""Train a CNN for radar micro-Doppler motion classification."""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms


CLASS_NAMES = ("hand", "fan", "sheet", "background")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SIZE = 224
NORMALIZE_MEAN = (0.5,)
NORMALIZE_STD = (0.5,)


@dataclass(frozen=True)
class DatasetSummary:
    public_count: int
    synthetic_count: int
    train_total: int
    test_count: int


class RadarSpectrogramDataset(Dataset):
    """Image dataset that loads grayscale spectrogram PNG files via OpenCV."""

    def __init__(self, root_dir: str, transform: transforms.Compose | None = None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.samples: list[tuple[str, int]] = self._scan_samples(root_dir)

    def _scan_samples(self, root_dir: str) -> list[tuple[str, int]]:
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset root not found: {root_dir}")

        samples: list[tuple[str, int]] = []
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Expected class folder missing: {class_dir}")

            for current_root, _, filenames in os.walk(class_dir):
                for filename in sorted(filenames):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        samples.append((os.path.join(current_root, filename), CLASS_TO_INDEX[class_name]))

        if not samples:
            raise RuntimeError(f"No image files found under {root_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image_path, label = self.samples[index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(image).unsqueeze(0)

        return image_tensor, label, image_path


class SimpleRadarCNN(nn.Module):
    """Three-block CNN classifier for grayscale micro-Doppler spectrograms."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN for radar micro-Doppler motion classification.")
    parser.add_argument("--public-root", default="public_dataset", help="Path to the public training dataset.")
    parser.add_argument(
        "--synthetic-root",
        default="synthetic_dataset",
        help="Path to the synthetic augmentation dataset.",
    )
    parser.add_argument("--test-root", default="processed_dataset", help="Path to the real radar test dataset.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training and test batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--num-workers", type=int, default=0, help="PyTorch DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--model-path",
        default="radar_motion_model.pth",
        help="Output path for the trained model checkpoint.",
    )
    parser.add_argument(
        "--confusion-matrix-path",
        default="confusion_matrix.png",
        help="Output path for the confusion matrix plot.",
    )
    parser.add_argument(
        "--predictions-path",
        default="test_predictions.png",
        help="Output path for the test prediction figure.",
    )
    parser.add_argument(
        "--no-show-plots",
        action="store_true",
        help="Save plots to disk without opening interactive windows.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def create_datasets(args: argparse.Namespace) -> tuple[ConcatDataset, RadarSpectrogramDataset, DatasetSummary]:
    transform = build_transform()
    public_dataset = RadarSpectrogramDataset(args.public_root, transform=transform)
    synthetic_dataset = RadarSpectrogramDataset(args.synthetic_root, transform=transform)
    test_dataset = RadarSpectrogramDataset(args.test_root, transform=transform)

    train_dataset = ConcatDataset([public_dataset, synthetic_dataset])
    summary = DatasetSummary(
        public_count=len(public_dataset),
        synthetic_count=len(synthetic_dataset),
        train_total=len(train_dataset),
        test_count=len(test_dataset),
    )
    return train_dataset, test_dataset, summary


def create_dataloaders(
    train_dataset: ConcatDataset,
    test_dataset: RadarSpectrogramDataset,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def print_dataset_summary(summary: DatasetSummary) -> None:
    print("## Dataset Summary")
    print()
    print(f"Public dataset: {summary.public_count} samples")
    print(f"Synthetic dataset: {summary.synthetic_count} samples")
    print(f"Training total: {summary.train_total} samples")
    print(f"Test dataset: {summary.test_count} samples")
    print()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets, _ in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((predictions == targets).sum().item())
        total_samples += batch_size

    average_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return average_loss, accuracy


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.append(predictions)
            all_targets.append(targets.numpy())

    targets_array = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
    predictions_array = np.concatenate(all_predictions) if all_predictions else np.array([], dtype=np.int64)
    confusion = compute_confusion_matrix(targets_array, predictions_array, num_classes=len(CLASS_NAMES))
    accuracy = float(np.trace(confusion) / max(np.sum(confusion), 1))
    return accuracy, confusion, targets_array, predictions_array


def compute_confusion_matrix(targets: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def compute_precision_recall(confusion: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    true_positives = np.diag(confusion).astype(np.float64)
    predicted_totals = confusion.sum(axis=0).astype(np.float64)
    actual_totals = confusion.sum(axis=1).astype(np.float64)

    precision = np.divide(
        true_positives,
        predicted_totals,
        out=np.zeros_like(true_positives),
        where=predicted_totals > 0,
    )
    recall = np.divide(
        true_positives,
        actual_totals,
        out=np.zeros_like(true_positives),
        where=actual_totals > 0,
    )
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    return precision, recall, macro_precision, macro_recall


def save_model(model: nn.Module, model_path: str) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "normalize_mean": NORMALIZE_MEAN,
        "normalize_std": NORMALIZE_STD,
    }
    torch.save(checkpoint, model_path)
    print(f"Saved trained model: {model_path}")


def resolve_show_plots(no_show_plots: bool) -> bool:
    backend = plt.get_backend().lower()
    if no_show_plots or "agg" in backend:
        if "agg" in backend and not no_show_plots:
            print(f"Matplotlib backend '{backend}' is non-interactive, so plots will be saved without opening windows.")
        return False
    return True


def plot_confusion_matrix(
    confusion: np.ndarray,
    output_path: str,
    show_plots: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(confusion, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_xticks(np.arange(len(CLASS_NAMES)))
    axis.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(CLASS_NAMES)))
    axis.set_yticklabels(CLASS_NAMES)

    for row_index in range(confusion.shape[0]):
        for column_index in range(confusion.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(confusion[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved confusion matrix plot: {output_path}")
    if show_plots:
        plt.show()
    plt.close(figure)


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().clone()
    image = image * NORMALIZE_STD[0] + NORMALIZE_MEAN[0]
    image = image.clamp(0.0, 1.0)
    return image.squeeze(0).numpy()


def plot_test_predictions(
    model: nn.Module,
    dataset: RadarSpectrogramDataset,
    device: torch.device,
    output_path: str,
    show_plots: bool,
    sample_count: int = 10,
) -> None:
    model.eval()
    sample_count = min(sample_count, len(dataset))
    sample_indices = random.sample(range(len(dataset)), sample_count)

    figure, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = np.array(axes).reshape(-1)

    with torch.no_grad():
        for axis, sample_index in zip(axes, sample_indices):
            image_tensor, label, image_path = dataset[sample_index]
            logits = model(image_tensor.unsqueeze(0).to(device))
            probabilities = torch.softmax(logits, dim=1)
            prediction = int(torch.argmax(probabilities, dim=1).item())
            confidence = float(torch.max(probabilities).item())

            display_image = denormalize_image(image_tensor)
            axis.imshow(display_image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(
                f"True: {CLASS_NAMES[label]}\nPred: {CLASS_NAMES[prediction]}\nConf: {confidence:.2f}",
                fontsize=9,
            )
            axis.set_xlabel(os.path.basename(image_path), fontsize=8)
            axis.axis("off")

    for axis in axes[sample_count:]:
        axis.axis("off")

    figure.suptitle("Test Predictions", fontsize=16)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved test prediction plot: {output_path}")
    if show_plots:
        plt.show()
    plt.close(figure)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    show_plots = resolve_show_plots(args.no_show_plots)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        train_dataset, test_dataset, summary = create_datasets(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Dataset setup failed: {exc}")
        return 1

    print_dataset_summary(summary)
    train_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = SimpleRadarCNN(num_classes=len(CLASS_NAMES)).to(device)
    # CrossEntropyLoss expects raw logits and applies the softmax operation internally.
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    print("Starting training...")
    for epoch in range(args.epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs:02d} | "
            f"loss={train_loss:.4f} | train_acc={train_accuracy:.4f}"
        )

    print()
    print("Evaluating on real radar test dataset...")
    test_accuracy, confusion, _, _ = evaluate_model(model=model, loader=test_loader, device=device)
    precision, recall, macro_precision, macro_recall = compute_precision_recall(confusion)

    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print()
    for class_name, class_precision, class_recall in zip(CLASS_NAMES, precision, recall):
        print(
            f"{class_name.capitalize():>10} | precision={class_precision:.4f} | recall={class_recall:.4f}"
        )
    print()

    save_model(model=model, model_path=args.model_path)
    plot_confusion_matrix(
        confusion=confusion,
        output_path=args.confusion_matrix_path,
        show_plots=show_plots,
    )
    plot_test_predictions(
        model=model,
        dataset=test_dataset,
        device=device,
        output_path=args.predictions_path,
        show_plots=show_plots,
        sample_count=10,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
