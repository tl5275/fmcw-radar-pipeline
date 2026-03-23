#!/usr/bin/env python3
"""Train and compare three radar micro-Doppler classifiers."""

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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASS_NAMES = ("hand", "fan", "sheet", "background")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}
IMAGE_SIZE = 128
PHYSICS_COLUMNS = ("dominant_freq", "bandwidth", "entropy", "energy_ratio", "harmonic_count")
BACKGROUND_TARGET_COUNT = 200


@dataclass(frozen=True)
class ModelResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare radar micro-Doppler classifiers.")
    parser.add_argument("--train-root", default="synthetic_dataset", help="Training dataset root directory.")
    parser.add_argument("--test-root", default="processed_dataset", help="Test dataset root directory.")
    parser.add_argument("--feature-csv", default="physics_features.csv", help="CSV file with physics features.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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
    """Resize grayscale spectrograms and normalize them for CNN input."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )


def load_feature_lookup(feature_csv_path: str) -> dict[str, np.ndarray]:
    """Load precomputed physics features indexed by image filename."""
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

    smoothed = np.convolve(energy_per_freq, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    threshold = float(np.mean(smoothed) + max(0.02, 0.15 * np.std(smoothed)))
    left = smoothed[1:-1] > smoothed[:-2]
    right = smoothed[1:-1] > smoothed[2:]
    prominent = smoothed[1:-1] > threshold
    return float(np.count_nonzero(left & right & prominent))


def compute_physics_features_from_image(image: np.ndarray) -> np.ndarray:
    """Compute the 5 handcrafted radar features from a grayscale spectrogram."""
    energy_per_freq = compute_energy_per_frequency(image)
    dominant_freq = float(int(np.argmax(energy_per_freq))) if energy_per_freq.size else 0.0
    bandwidth = compute_bandwidth(energy_per_freq)
    entropy = compute_entropy(energy_per_freq)
    energy_ratio = compute_energy_ratio(energy_per_freq)
    harmonic_count = compute_harmonic_peak_count(energy_per_freq)
    return np.array([dominant_freq, bandwidth, entropy, energy_ratio, harmonic_count], dtype=np.float32)


def add_gaussian_noise(image: np.ndarray) -> np.ndarray:
    noise_std = random.uniform(0.01, 0.05)
    noise = np.random.normal(0.0, noise_std, image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)


def make_low_energy_spectrogram(shape: tuple[int, int]) -> np.ndarray:
    base = np.random.uniform(0.0, 0.12, size=shape).astype(np.float32)
    blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=5.0, sigmaY=7.0)
    return np.clip(blurred, 0.0, 1.0)


def add_doppler_center_noise(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    center = (height // 2) + random.randint(-10, 10)
    spread = random.uniform(8.0, 18.0)
    rows = np.arange(height, dtype=np.float32)
    gaussian_profile = np.exp(-0.5 * ((rows - center) / spread) ** 2).astype(np.float32)
    column_variation = np.random.uniform(0.01, 0.15, size=(1, width)).astype(np.float32)
    center_band = gaussian_profile[:, None] * column_variation
    center_band = cv2.GaussianBlur(center_band, (0, 0), sigmaX=3.0, sigmaY=2.0)
    return np.clip(image + center_band, 0.0, 1.0)


def synthesize_background_image(base_image: np.ndarray) -> np.ndarray:
    """Create an additional background spectrogram using low-energy radar-like noise."""
    augmented = base_image.copy()

    if random.random() < 0.9:
        augmented = add_gaussian_noise(augmented)
    if random.random() < 0.9:
        augmented = add_doppler_center_noise(augmented)
    if random.random() < 0.7:
        low_energy = make_low_energy_spectrogram(augmented.shape)
        blend_weight = random.uniform(0.2, 0.5)
        augmented = np.clip((1.0 - blend_weight) * augmented + blend_weight * low_energy, 0.0, 1.0)

    return np.clip(augmented, 0.0, 1.0)


def load_grayscale_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image.astype(np.uint8)


class RadarDataset(Dataset):
    """Shared dataset for all three experiments, with optional extra background synthesis."""

    def __init__(
        self,
        root_dir: str,
        feature_lookup: dict[str, np.ndarray],
        transform: transforms.Compose | None = None,
        augment_background_to: int | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.feature_lookup = feature_lookup
        self.transform = transform
        self.records = self._scan_records(root_dir)
        if augment_background_to is not None:
            self._augment_background_records(augment_background_to)
        self.raw_physics_features = self._build_feature_matrix()
        self.scaled_physics_features = self.raw_physics_features.copy()

    def _scan_records(self, root_dir: str) -> list[dict[str, object]]:
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset root not found: {root_dir}")

        records: list[dict[str, object]] = []
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Expected class directory missing: {class_dir}")

            for current_root, _, filenames in os.walk(class_dir):
                for filename in sorted(filenames):
                    if filename.lower().endswith(".png"):
                        records.append(
                            {
                                "image_name": filename,
                                "image_path": os.path.join(current_root, filename),
                                "image_array": None,
                                "label": CLASS_TO_INDEX[class_name],
                            }
                        )

        if not records:
            raise RuntimeError(f"No PNG files found under {root_dir}")
        return records

    def _augment_background_records(self, target_count: int) -> None:
        background_records = [record for record in self.records if int(record["label"]) == CLASS_TO_INDEX["background"]]
        current_count = len(background_records)
        if current_count >= target_count:
            return

        needed = target_count - current_count
        if current_count == 0:
            raise RuntimeError("Cannot synthesize additional background samples because no background images exist.")

        for index in range(needed):
            base_record = random.choice(background_records)
            base_image = self._load_record_image(base_record).astype(np.float32) / 255.0
            augmented = synthesize_background_image(base_image)
            augmented_uint8 = np.clip(augmented * 255.0, 0.0, 255.0).astype(np.uint8)
            self.records.append(
                {
                    "image_name": f"background_train_aug_{index + 1:04d}.png",
                    "image_path": None,
                    "image_array": augmented_uint8,
                    "label": CLASS_TO_INDEX["background"],
                }
            )

    def _load_record_image(self, record: dict[str, object]) -> np.ndarray:
        image_array = record["image_array"]
        if image_array is not None:
            return np.asarray(image_array, dtype=np.uint8)

        image_path = str(record["image_path"])
        return load_grayscale_image(image_path)

    def _build_feature_matrix(self) -> np.ndarray:
        feature_rows: list[np.ndarray] = []
        for record in self.records:
            image_name = str(record["image_name"])
            if image_name in self.feature_lookup:
                feature_rows.append(self.feature_lookup[image_name].astype(np.float32))
                continue

            normalized = self._load_record_image(record).astype(np.float32) / 255.0
            feature_rows.append(compute_physics_features_from_image(normalized))

        return np.stack(feature_rows).astype(np.float32)

    def apply_scaler(self, scaler: StandardScaler) -> None:
        self.scaled_physics_features = scaler.transform(self.raw_physics_features).astype(np.float32)

    def class_counts(self) -> dict[str, int]:
        counts = {class_name: 0 for class_name in CLASS_NAMES}
        for record in self.records:
            counts[INDEX_TO_CLASS[int(record["label"])]] += 1
        return counts

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        record = self.records[index]
        image = self._load_record_image(record)

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)

        physics_features = torch.from_numpy(self.scaled_physics_features[index])
        return image_tensor, physics_features, int(record["label"])


class CNNBackbone(nn.Module):
    """Common CNN feature extractor used by all models for a fair comparison."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
        )
        self.attention_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.attention_fc = nn.Linear(64, 64)
        self.attention_projection = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )

    def forward_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        feature_map = self.feature_map(image_tensor)
        return self.embedding_head(feature_map)

    def forward_attention_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        feature_map = self.feature_map(image_tensor)
        pooled = self.attention_pool(feature_map).flatten(1)
        attention = torch.softmax(self.attention_fc(pooled), dim=1)
        weighted_features = pooled * attention
        return self.attention_projection(weighted_features)


class CNNBaseline(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = CNNBackbone()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, image_tensor: torch.Tensor, physics_features: torch.Tensor | None = None) -> torch.Tensor:
        del physics_features
        features = self.backbone.forward_embedding(image_tensor)
        return self.classifier(features)


class CNNPhysics(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = CNNBackbone()
        self.classifier = nn.Sequential(
            nn.Linear(128 + 5, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, image_tensor: torch.Tensor, physics_features: torch.Tensor | None = None) -> torch.Tensor:
        if physics_features is None:
            raise ValueError("Physics features are required for CNNPhysics.")
        features = self.backbone.forward_embedding(image_tensor)
        fused = torch.cat([features, physics_features], dim=1)
        return self.classifier(fused)


class CNNPhysicsAttention(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = CNNBackbone()
        self.classifier = nn.Sequential(
            nn.Linear(128 + 5, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, image_tensor: torch.Tensor, physics_features: torch.Tensor | None = None) -> torch.Tensor:
        if physics_features is None:
            raise ValueError("Physics features are required for CNNPhysicsAttention.")
        features = self.backbone.forward_attention_embedding(image_tensor)
        fused = torch.cat([features, physics_features], dim=1)
        return self.classifier(fused)


def create_datasets(
    args: argparse.Namespace,
) -> tuple[RadarDataset, RadarDataset, StandardScaler]:
    transform = build_transform()
    feature_lookup = load_feature_lookup(args.feature_csv)

    train_dataset = RadarDataset(
        root_dir=args.train_root,
        feature_lookup=feature_lookup,
        transform=transform,
        augment_background_to=BACKGROUND_TARGET_COUNT,
    )
    test_dataset = RadarDataset(
        root_dir=args.test_root,
        feature_lookup=feature_lookup,
        transform=transform,
        augment_background_to=None,
    )

    scaler = StandardScaler()
    scaler.fit(train_dataset.raw_physics_features)
    train_dataset.apply_scaler(scaler)
    test_dataset.apply_scaler(scaler)
    return train_dataset, test_dataset, scaler


def create_dataloaders(
    train_dataset: RadarDataset,
    test_dataset: RadarDataset,
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


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
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

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> ModelResult:
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
    confusion = confusion_matrix(labels_array, predictions_array, labels=np.arange(len(CLASS_NAMES)))
    return ModelResult(
        name="",
        accuracy=accuracy_score(labels_array, predictions_array),
        precision=precision_score(labels_array, predictions_array, average="macro", zero_division=0),
        recall=recall_score(labels_array, predictions_array, average="macro", zero_division=0),
        f1=f1_score(labels_array, predictions_array, average="macro", zero_division=0),
        confusion=confusion,
    )


def save_model(
    model: nn.Module,
    output_path: str,
    scaler: StandardScaler,
    model_name: str,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "class_names": CLASS_NAMES,
        "physics_columns": PHYSICS_COLUMNS,
        "image_size": IMAGE_SIZE,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }
    torch.save(checkpoint, output_path)
    print(f"Saved model: {output_path}")


def print_dataset_summary(train_dataset: RadarDataset, test_dataset: RadarDataset) -> None:
    train_counts = train_dataset.class_counts()
    test_counts = test_dataset.class_counts()
    print("Training counts:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {train_counts[class_name]}")
    print("Testing counts:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {test_counts[class_name]}")
    print()


def run_experiment(
    model_name: str,
    model: nn.Module,
    model_path: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    scaler: StandardScaler,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> ModelResult:
    print(f"=== {model_name} ===")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss, epoch_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        print(
            f"Epoch {epoch + 1:02d}/{epochs:02d} | "
            f"loss={epoch_loss:.4f} | training accuracy={epoch_accuracy:.4f}"
        )

    result = evaluate_model(model=model, dataloader=test_loader, device=device)
    result = ModelResult(
        name=model_name,
        accuracy=result.accuracy,
        precision=result.precision,
        recall=result.recall,
        f1=result.f1,
        confusion=result.confusion,
    )

    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Precision: {result.precision:.4f}")
    print(f"Recall: {result.recall:.4f}")
    print(f"F1 score: {result.f1:.4f}")
    print("Confusion matrix:")
    print(result.confusion)
    print()

    save_model(model=model, output_path=model_path, scaler=scaler, model_name=model_name)
    print()
    return result


def print_comparison(results: list[ModelResult]) -> None:
    print("## Model                Accuracy")
    print()
    for result in results:
        print(f"{result.name:<24} {result.accuracy * 100:6.2f}%")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        train_dataset, test_dataset, scaler = create_datasets(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Dataset setup failed: {exc}")
        return 1

    print_dataset_summary(train_dataset, test_dataset)
    train_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    results = []
    results.append(
        run_experiment(
            model_name="CNN baseline",
            model=CNNBaseline(num_classes=len(CLASS_NAMES)),
            model_path="cnn_baseline_model.pth",
            train_loader=train_loader,
            test_loader=test_loader,
            scaler=scaler,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device,
        )
    )
    results.append(
        run_experiment(
            model_name="CNN + Physics",
            model=CNNPhysics(num_classes=len(CLASS_NAMES)),
            model_path="cnn_physics_model.pth",
            train_loader=train_loader,
            test_loader=test_loader,
            scaler=scaler,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device,
        )
    )
    results.append(
        run_experiment(
            model_name="CNN + Physics + Attn",
            model=CNNPhysicsAttention(num_classes=len(CLASS_NAMES)),
            model_path="cnn_physics_attention_model.pth",
            train_loader=train_loader,
            test_loader=test_loader,
            scaler=scaler,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device,
        )
    )

    print_comparison(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
