#!/usr/bin/env python3
"""Train a hybrid physics-guided CNN for radar micro-Doppler classification."""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASS_NAMES = ("hand", "fan", "sheet", "background")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SIZE = 128
PHYSICS_COLUMNS = ("dominant_freq", "bandwidth", "entropy", "energy_ratio", "harmonic_count")


@dataclass(frozen=True)
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hybrid physics-guided CNN for radar spectrogram classification.")
    parser.add_argument("--train-root", default="synthetic_dataset", help="Training image root directory.")
    parser.add_argument("--test-root", default="processed_dataset", help="Test image root directory.")
    parser.add_argument("--feature-csv", default="physics_features.csv", help="CSV file containing precomputed physics features.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and testing.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--model-path",
        default="physics_guided_radar_model.pth",
        help="Output path for the trained model checkpoint.",
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
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )


def load_feature_lookup(feature_csv_path: str) -> dict[str, np.ndarray]:
    if not os.path.isfile(feature_csv_path):
        raise FileNotFoundError(f"Physics feature CSV not found: {feature_csv_path}")

    frame = pd.read_csv(feature_csv_path)
    required_columns = {"image", "label", *PHYSICS_COLUMNS}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in physics feature CSV: {sorted(missing_columns)}")

    feature_lookup: dict[str, np.ndarray] = {}
    for _, row in frame.iterrows():
        feature_lookup[str(row["image"])] = row[list(PHYSICS_COLUMNS)].to_numpy(dtype=np.float32)
    return feature_lookup


def compute_energy_per_frequency(image: np.ndarray) -> np.ndarray:
    return np.mean(image, axis=1).astype(np.float32)


def compute_bandwidth(energy_per_freq: np.ndarray) -> float:
    total_energy = float(np.sum(energy_per_freq))
    if total_energy <= 0.0:
        return 0.0

    cumulative_energy = np.cumsum(energy_per_freq) / total_energy
    f_low = int(np.searchsorted(cumulative_energy, 0.05, side="left"))
    f_high = int(np.searchsorted(cumulative_energy, 0.95, side="left"))
    return float(max(f_high - f_low, 0))


def compute_entropy(energy_per_freq: np.ndarray) -> float:
    total_energy = float(np.sum(energy_per_freq))
    if total_energy <= 0.0:
        return 0.0

    probabilities = energy_per_freq / total_energy
    return float(-np.sum(probabilities * np.log(probabilities + 1.0e-12)))


def compute_energy_ratio(energy_per_freq: np.ndarray) -> float:
    total_energy = float(np.sum(energy_per_freq))
    if total_energy <= 0.0:
        return 0.0

    length = energy_per_freq.shape[0]
    center_start = int(0.4 * length)
    center_end = int(0.6 * length)
    center_energy = float(np.sum(energy_per_freq[center_start:center_end]))
    return center_energy / total_energy


def compute_harmonic_peak_count(energy_per_freq: np.ndarray) -> float:
    if energy_per_freq.size < 3:
        return 0.0

    smoothed = energy_per_freq.copy()
    kernel = np.ones(5, dtype=np.float32) / 5.0
    smoothed = np.convolve(smoothed, kernel, mode="same")
    mean_value = float(np.mean(smoothed))
    std_value = float(np.std(smoothed))
    threshold = mean_value + max(0.02, 0.15 * std_value)

    left = smoothed[1:-1] > smoothed[:-2]
    right = smoothed[1:-1] > smoothed[2:]
    prominent = smoothed[1:-1] > threshold
    return float(np.count_nonzero(left & right & prominent))


def compute_physics_features_from_image(image: np.ndarray) -> np.ndarray:
    """Fallback feature extraction for images not present in the CSV."""
    energy_per_freq = compute_energy_per_frequency(image)
    dominant_freq = float(int(np.argmax(energy_per_freq))) if energy_per_freq.size else 0.0
    bandwidth = compute_bandwidth(energy_per_freq)
    entropy = compute_entropy(energy_per_freq)
    energy_ratio = compute_energy_ratio(energy_per_freq)
    harmonic_count = compute_harmonic_peak_count(energy_per_freq)
    return np.array([dominant_freq, bandwidth, entropy, energy_ratio, harmonic_count], dtype=np.float32)


class RadarDataset(Dataset):
    """Return image tensor, physics feature vector, and class label for each sample."""

    def __init__(
        self,
        root_dir: str,
        feature_lookup: dict[str, np.ndarray],
        feature_stats: FeatureStats | None = None,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.feature_lookup = feature_lookup
        self.feature_stats = feature_stats
        self.transform = transform
        self.samples = self._scan_samples(root_dir)
        self.raw_physics_features = self._build_feature_matrix()

    def _scan_samples(self, root_dir: str) -> list[tuple[str, int]]:
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset root not found: {root_dir}")

        samples: list[tuple[str, int]] = []
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Expected class directory missing: {class_dir}")

            for current_root, _, filenames in os.walk(class_dir):
                for filename in sorted(filenames):
                    if filename.lower().endswith(".png"):
                        samples.append((os.path.join(current_root, filename), CLASS_TO_INDEX[class_name]))

        if not samples:
            raise RuntimeError(f"No PNG files found under {root_dir}")
        return samples

    def _build_feature_matrix(self) -> np.ndarray:
        feature_rows: list[np.ndarray] = []
        for image_path, _ in self.samples:
            image_name = os.path.basename(image_path)
            if image_name in self.feature_lookup:
                feature_rows.append(self.feature_lookup[image_name].astype(np.float32))
                continue

            # The provided CSV currently covers synthetic training images. For
            # held-out real radar images we compute the same 5 physics features
            # at runtime so evaluation can still proceed.
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Failed to read image while computing physics features: {image_path}")
            normalized = image.astype(np.float32) / 255.0
            feature_rows.append(compute_physics_features_from_image(normalized))

        return np.stack(feature_rows).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)

        physics_features = self.raw_physics_features[index].copy()
        if self.feature_stats is not None:
            physics_features = (physics_features - self.feature_stats.mean) / self.feature_stats.std

        return image_tensor, torch.from_numpy(physics_features.astype(np.float32)), label


class PhysicsGuidedCNN(nn.Module):
    """CNN image encoder fused with handcrafted radar physics features."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.cnn_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 5, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, image_tensor: torch.Tensor, physics_features: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn(image_tensor)
        cnn_features = self.cnn_projection(cnn_features)
        fused_features = torch.cat([cnn_features, physics_features], dim=1)
        return self.classifier(fused_features)


def compute_feature_stats(feature_matrix: np.ndarray) -> FeatureStats:
    mean = feature_matrix.mean(axis=0).astype(np.float32)
    std = feature_matrix.std(axis=0).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    return FeatureStats(mean=mean, std=std)


def create_dataloaders(args: argparse.Namespace) -> tuple[RadarDataset, RadarDataset, DataLoader, DataLoader, FeatureStats]:
    transform = build_transform()
    feature_lookup = load_feature_lookup(args.feature_csv)
    train_dataset = RadarDataset(root_dir=args.train_root, feature_lookup=feature_lookup, transform=transform)
    feature_stats = compute_feature_stats(train_dataset.raw_physics_features)
    train_dataset.feature_stats = feature_stats

    test_dataset = RadarDataset(
        root_dir=args.test_root,
        feature_lookup=feature_lookup,
        feature_stats=feature_stats,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dataset, test_dataset, train_loader, test_loader, feature_stats


def train_one_epoch(
    model: PhysicsGuidedCNN,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for image_tensor, physics_features, labels in dataloader:
        image_tensor = image_tensor.to(device)
        physics_features = physics_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image_tensor, physics_features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_samples += batch_size

    average_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return average_loss, accuracy


def evaluate(
    model: PhysicsGuidedCNN,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for image_tensor, physics_features, labels in dataloader:
            image_tensor = image_tensor.to(device)
            physics_features = physics_features.to(device)
            logits = model(image_tensor, physics_features)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.append(predictions)
            all_labels.append(labels.numpy())

    labels_array = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    predictions_array = np.concatenate(all_predictions) if all_predictions else np.array([], dtype=np.int64)
    return labels_array, predictions_array


def save_model(model: PhysicsGuidedCNN, feature_stats: FeatureStats, output_path: str) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "feature_mean": feature_stats.mean,
        "feature_std": feature_stats.std,
        "image_size": IMAGE_SIZE,
    }
    torch.save(checkpoint, output_path)
    print(f"Saved trained model: {output_path}")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        train_dataset, test_dataset, train_loader, test_loader, feature_stats = create_dataloaders(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Dataset setup failed: {exc}")
        return 1

    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print()

    model = PhysicsGuidedCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        epoch_loss, epoch_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs:02d} | "
            f"loss={epoch_loss:.4f} | training accuracy={epoch_accuracy:.4f}"
        )

    labels_array, predictions_array = evaluate(model=model, dataloader=test_loader, device=device)
    test_accuracy = accuracy_score(labels_array, predictions_array)
    precision = precision_score(labels_array, predictions_array, average="macro", zero_division=0)
    recall = recall_score(labels_array, predictions_array, average="macro", zero_division=0)
    f1 = f1_score(labels_array, predictions_array, average="macro", zero_division=0)
    conf_matrix = confusion_matrix(labels_array, predictions_array, labels=list(range(len(CLASS_NAMES))))

    print()
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    save_model(model=model, feature_stats=feature_stats, output_path=args.model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
