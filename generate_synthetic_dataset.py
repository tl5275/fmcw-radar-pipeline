#!/usr/bin/env python3
"""Generate a synthetic radar micro-Doppler dataset with physics-preserving augmentation."""

from __future__ import annotations

import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np


INPUT_ROOT = "processed_dataset"
OUTPUT_ROOT = "synthetic_dataset"
CLASS_NAMES = ("hand", "fan", "sheet", "background")
IMAGE_SIZE = 256
MIN_AUGMENTATIONS_PER_IMAGE = 10
MAX_AUGMENTATIONS_PER_IMAGE = 20
PREVIEW_COUNT = 10
PREVIEW_OUTPUT = "synthetic_augmentation_preview.png"
RANDOM_SEED = 42


def scan_dataset(root_dir: str) -> dict[str, list[str]]:
    """Collect PNG image paths for each expected class."""
    image_paths: dict[str, list[str]] = {class_name: [] for class_name in CLASS_NAMES}
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Input dataset not found: {root_dir}")

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Expected class folder missing: {class_dir}")

        for current_root, _, filenames in os.walk(class_dir):
            for filename in sorted(filenames):
                if filename.lower().endswith(".png"):
                    image_paths[class_name].append(os.path.join(current_root, filename))

    return image_paths


def prepare_output_root(root_dir: str) -> None:
    """Create synthetic dataset folders and clear previous PNG outputs."""
    os.makedirs(root_dir, exist_ok=True)
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(root_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for current_root, _, filenames in os.walk(class_dir):
            for filename in filenames:
                if filename.lower().endswith(".png"):
                    os.remove(os.path.join(current_root, filename))


def load_grayscale_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def roll_with_zero_fill(image: np.ndarray, shift: int, axis: int) -> np.ndarray:
    """Shift spectrogram content while blanking newly exposed regions."""
    shifted = np.roll(image, shift, axis=axis)
    if shift == 0:
        return shifted

    if axis == 0:
        if shift > 0:
            shifted[:shift, :] = 0.0
        else:
            shifted[shift:, :] = 0.0
    else:
        if shift > 0:
            shifted[:, :shift] = 0.0
        else:
            shifted[:, shift:] = 0.0
    return shifted


def apply_doppler_frequency_shift(image: np.ndarray) -> np.ndarray:
    shift = random.randint(-20, 20)
    return roll_with_zero_fill(image, shift=shift, axis=0)


def apply_time_shift(image: np.ndarray) -> np.ndarray:
    shift = random.randint(-25, 25)
    return roll_with_zero_fill(image, shift=shift, axis=1)


def apply_time_mask(image: np.ndarray) -> np.ndarray:
    width = random.randint(10, 40)
    start = random.randint(0, IMAGE_SIZE - width)
    masked = image.copy()
    masked[:, start : start + width] = 0.0
    return masked


def apply_frequency_mask(image: np.ndarray) -> np.ndarray:
    height = random.randint(10, 30)
    start = random.randint(0, IMAGE_SIZE - height)
    masked = image.copy()
    masked[start : start + height, :] = 0.0
    return masked


def apply_gaussian_radar_noise(image: np.ndarray) -> np.ndarray:
    noise = np.random.normal(0.0, 0.02, image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)


def apply_doppler_warp(image: np.ndarray) -> np.ndarray:
    """Warp the Doppler axis smoothly to mimic nonuniform target velocity."""
    height, width = image.shape
    amplitude = random.uniform(3.0, 12.0)
    cycles = random.uniform(0.5, 2.0)
    phase = random.uniform(0.0, 2.0 * np.pi)
    rows = np.arange(height, dtype=np.float32)
    warped = np.zeros_like(image)

    for column_index in range(width):
        shift = amplitude * np.sin((2.0 * np.pi * cycles * column_index / width) + phase)
        source_rows = np.clip(rows - shift, 0.0, height - 1.0)
        warped[:, column_index] = np.interp(source_rows, rows, image[:, column_index])

    return warped


def apply_contrast_variation(image: np.ndarray) -> np.ndarray:
    alpha = random.uniform(0.8, 1.2)
    return np.clip(alpha * image, 0.0, 1.0)


def augment_spectrogram(image: np.ndarray) -> np.ndarray:
    """Compose radar-safe augmentations without flips or rotations."""
    augmented = image.copy()

    augmented = apply_doppler_frequency_shift(augmented)
    augmented = apply_time_shift(augmented)

    if random.random() < 0.8:
        augmented = apply_doppler_warp(augmented)
    if random.random() < 0.7:
        augmented = apply_time_mask(augmented)
    if random.random() < 0.7:
        augmented = apply_frequency_mask(augmented)

    augmented = apply_gaussian_radar_noise(augmented)
    augmented = apply_contrast_variation(augmented)
    return np.clip(augmented, 0.0, 1.0)


def save_image(image: np.ndarray, output_path: str) -> None:
    encoded = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    if not cv2.imwrite(output_path, encoded):
        raise RuntimeError(f"Failed to write augmented image: {output_path}")


def visualize_augmented_samples(sample_paths: list[str]) -> None:
    if not sample_paths:
        print("No augmented images available for visualization.")
        return

    sample_paths = random.sample(sample_paths, min(PREVIEW_COUNT, len(sample_paths)))
    figure, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = np.array(axes).reshape(-1)

    for axis, image_path in zip(axes, sample_paths):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(os.path.basename(os.path.dirname(image_path)).capitalize(), fontsize=10)
        axis.axis("off")

    for axis in axes[len(sample_paths) :]:
        axis.axis("off")

    figure.suptitle("Random Synthetic Micro-Doppler Spectrograms", fontsize=16)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(PREVIEW_OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved augmentation preview: {PREVIEW_OUTPUT}")

    backend = plt.get_backend().lower()
    if "agg" in backend:
        print(f"Matplotlib backend '{backend}' is non-interactive, so the preview was saved instead of displayed.")
    else:
        plt.show()
    plt.close(figure)


def print_summary(
    original_count: int,
    synthetic_count: int,
    original_per_class: dict[str, int],
    synthetic_per_class: dict[str, int],
) -> None:
    print()
    print("## Dataset Augmentation Summary")
    print()
    print(f"Original samples: {original_count}")
    print(f"Synthetic samples: {synthetic_count}")
    print(f"Total samples: {original_count + synthetic_count}")
    print()
    for class_name in CLASS_NAMES:
        print(
            f"{class_name.capitalize():>10} | original={original_per_class[class_name]} | "
            f"synthetic={synthetic_per_class[class_name]}"
        )


def main() -> int:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    image_paths_by_class = scan_dataset(INPUT_ROOT)
    prepare_output_root(OUTPUT_ROOT)

    original_per_class = {class_name: len(paths) for class_name, paths in image_paths_by_class.items()}
    synthetic_per_class = {class_name: 0 for class_name in CLASS_NAMES}
    saved_augmented_paths: list[str] = []

    for class_name in CLASS_NAMES:
        class_paths = image_paths_by_class[class_name]
        for image_path in class_paths:
            original_image = load_grayscale_image(image_path)
            augmentations_for_image = random.randint(
                MIN_AUGMENTATIONS_PER_IMAGE,
                MAX_AUGMENTATIONS_PER_IMAGE,
            )

            for _ in range(augmentations_for_image):
                augmented_image = augment_spectrogram(original_image)
                synthetic_per_class[class_name] += 1
                output_filename = f"{class_name}_aug_{synthetic_per_class[class_name]:04d}.png"
                output_path = os.path.join(OUTPUT_ROOT, class_name, output_filename)
                save_image(augmented_image, output_path)
                saved_augmented_paths.append(output_path)

    original_count = sum(original_per_class.values())
    synthetic_count = sum(synthetic_per_class.values())
    print_summary(
        original_count=original_count,
        synthetic_count=synthetic_count,
        original_per_class=original_per_class,
        synthetic_per_class=synthetic_per_class,
    )
    visualize_augmented_samples(saved_augmented_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
