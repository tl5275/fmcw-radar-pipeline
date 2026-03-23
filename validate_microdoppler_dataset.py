#!/usr/bin/env python3
"""Validate a processed micro-Doppler image dataset using radar-inspired checks."""

from __future__ import annotations

import os
import random
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_ROOT = "processed_dataset"
CSV_OUTPUT = "doppler_statistics.csv"
SAMPLE_FIGURE_OUTPUT = "validation_samples.png"
CLASS_ORDER = ("hand", "fan", "sheet", "background")
CLASS_TITLES = {
    "hand": "Hand",
    "fan": "Fan",
    "sheet": "Sheet",
    "background": "Background",
}

ENERGY_THRESHOLD = 0.02
BACKGROUND_CENTER_RATIO_THRESHOLD = 0.6
HAND_VARIANCE_THRESHOLD = 0.004
SHEET_BANDWIDTH_THRESHOLD = 20
FAN_FFT_RATIO_THRESHOLD = 3.0
FAN_PEAK_SIGMA_THRESHOLD = 1.5
FREQUENCY_RANGE_HZ = 500.0
RANDOM_SAMPLES_PER_CLASS = 5


def scan_dataset(dataset_root: str) -> dict[str, list[str]]:
    """Recursively collect PNG paths under processed_dataset/<class>/."""
    image_paths: dict[str, list[str]] = defaultdict(list)
    for current_root, _, filenames in os.walk(dataset_root):
        for filename in filenames:
            if not filename.lower().endswith(".png"):
                continue

            full_path = os.path.join(current_root, filename)
            relative_path = os.path.relpath(full_path, dataset_root)
            parts = relative_path.split(os.sep)
            if not parts:
                continue

            class_name = parts[0].lower()
            if class_name in CLASS_ORDER:
                image_paths[class_name].append(full_path)

    for class_name in image_paths:
        image_paths[class_name].sort()
    return image_paths


def load_grayscale_image(image_path: str) -> np.ndarray | None:
    """Load a PNG as a normalized grayscale float image in [0, 1]."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return image.astype(np.float32) / 255.0


def compute_activity_threshold(row_profile: np.ndarray) -> float:
    """Use an adaptive threshold so wide but weak Doppler bands are preserved."""
    return float(max(np.mean(row_profile) + 0.5 * np.std(row_profile), 0.15))


def compute_dominant_doppler_frequency(row_profile: np.ndarray) -> float:
    """Map the strongest row to Hz assuming the image spans 0-500 Hz vertically."""
    dominant_row = int(np.argmax(row_profile))
    height = row_profile.shape[0]
    if height <= 1:
        return 0.0

    # build_microdoppler_dataset.py flips the spectrogram vertically before
    # saving, so the bottom row corresponds to 0 Hz and the top row to 500 Hz.
    normalized_position = (height - 1 - dominant_row) / (height - 1)
    return float(normalized_position * FREQUENCY_RANGE_HZ)


def compute_bandwidth_metrics(row_profile: np.ndarray, height: int) -> tuple[int, float]:
    activity_threshold = compute_activity_threshold(row_profile)
    bandwidth_pixels = int(np.count_nonzero(row_profile > activity_threshold))
    bandwidth_hz = float((bandwidth_pixels / max(height, 1)) * FREQUENCY_RANGE_HZ)
    return bandwidth_pixels, bandwidth_hz


def compute_spectral_entropy(image: np.ndarray) -> float:
    flat = image.astype(np.float64).ravel()
    total_power = float(np.sum(flat))
    if total_power <= 0.0:
        return 0.0

    probabilities = flat / total_power
    entropy = -np.sum(probabilities * np.log2(probabilities + 1.0e-12))
    return float(entropy / np.log2(probabilities.size))


def analyze_fan_periodicity(image: np.ndarray) -> tuple[bool, int, float]:
    """Check whether the fan spectrogram contains periodic harmonic structure."""
    centered = image - np.mean(image, axis=0, keepdims=True)
    fft_energy = np.abs(np.fft.rfft(centered, axis=0))
    if fft_energy.shape[0] <= 2:
        return False, 0, 0.0

    harmonic_profile = np.mean(fft_energy[1:], axis=1)
    peak_ratio = float(np.max(harmonic_profile) / max(np.mean(harmonic_profile), 1.0e-12))

    peak_count = 0
    if harmonic_profile.size >= 3:
        peak_threshold = float(np.mean(harmonic_profile) + FAN_PEAK_SIGMA_THRESHOLD * np.std(harmonic_profile))
        peak_mask = (
            (harmonic_profile[1:-1] > harmonic_profile[:-2])
            & (harmonic_profile[1:-1] > harmonic_profile[2:])
            & (harmonic_profile[1:-1] > peak_threshold)
        )
        peak_count = int(np.count_nonzero(peak_mask))

    has_periodic_structure = peak_count > 0 or peak_ratio >= FAN_FFT_RATIO_THRESHOLD
    return has_periodic_structure, peak_count, peak_ratio


def validate_image(class_name: str, image_path: str, image: np.ndarray) -> dict[str, object]:
    energy = float(np.mean(image))
    row_profile = np.mean(image, axis=1)
    dominant_frequency_hz = compute_dominant_doppler_frequency(row_profile)
    bandwidth_pixels, bandwidth_hz = compute_bandwidth_metrics(row_profile, image.shape[0])
    spectral_entropy = compute_spectral_entropy(image)

    record: dict[str, object] = {
        "image_path": image_path,
        "class_name": class_name,
        "energy": energy,
        "dominant_doppler_frequency_hz": dominant_frequency_hz,
        "doppler_bandwidth_pixels": bandwidth_pixels,
        "doppler_bandwidth_hz": bandwidth_hz,
        "spectral_entropy": spectral_entropy,
        "background_center_ratio": np.nan,
        "hand_time_variance": np.nan,
        "fan_peak_count": np.nan,
        "fan_peak_ratio": np.nan,
        "is_valid": True,
        "violations": "",
    }

    violations: list[str] = []
    if energy < ENERGY_THRESHOLD:
        violations.append("Low energy spectrogram")

    if class_name == "background":
        center_band = image[110:146, :]
        center_energy = float(np.mean(center_band))
        center_ratio = float(center_energy / max(energy, 1.0e-12))
        record["background_center_ratio"] = center_ratio
        if center_ratio < BACKGROUND_CENTER_RATIO_THRESHOLD:
            violations.append("Background Doppler not centered")

    elif class_name == "fan":
        has_periodic_structure, peak_count, peak_ratio = analyze_fan_periodicity(image)
        record["fan_peak_count"] = peak_count
        record["fan_peak_ratio"] = peak_ratio
        if not has_periodic_structure:
            violations.append("Fan periodic harmonics not detected")

    elif class_name == "hand":
        time_variance = np.var(image, axis=1)
        mean_time_variance = float(np.mean(time_variance))
        record["hand_time_variance"] = mean_time_variance
        if mean_time_variance < HAND_VARIANCE_THRESHOLD:
            violations.append("Hand motion appears static")

    elif class_name == "sheet":
        if bandwidth_pixels < SHEET_BANDWIDTH_THRESHOLD:
            violations.append("Sheet Doppler bandwidth too narrow")

    if violations:
        record["is_valid"] = False
        record["violations"] = "; ".join(violations)

    return record


def display_random_samples(image_paths_by_class: dict[str, list[str]]) -> None:
    figure, axes = plt.subplots(len(CLASS_ORDER), RANDOM_SAMPLES_PER_CLASS, figsize=(15, 12))
    if len(CLASS_ORDER) == 1:
        axes = np.expand_dims(axes, axis=0)

    figure.suptitle("Random Micro-Doppler Samples for Manual Inspection", fontsize=16)

    for row_index, class_name in enumerate(CLASS_ORDER):
        sampled_paths = image_paths_by_class.get(class_name, [])
        if sampled_paths:
            sampled_paths = random.sample(sampled_paths, min(len(sampled_paths), RANDOM_SAMPLES_PER_CLASS))

        for column_index in range(RANDOM_SAMPLES_PER_CLASS):
            axis = axes[row_index, column_index]
            axis.axis("off")

            if column_index < len(sampled_paths):
                image = cv2.imread(sampled_paths[column_index], cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
                    axis.set_title(CLASS_TITLES[class_name], fontsize=10)
            else:
                axis.text(0.5, 0.5, "No Image", ha="center", va="center", fontsize=9)

            if column_index == 0:
                axis.set_ylabel(CLASS_TITLES[class_name], rotation=90, fontsize=12, labelpad=12)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(SAMPLE_FIGURE_OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved manual inspection figure: {SAMPLE_FIGURE_OUTPUT}")

    backend = plt.get_backend().lower()
    if "agg" in backend:
        print(f"Matplotlib backend '{backend}' is non-interactive, so the figure was saved instead of displayed.")
    else:
        plt.show()

    plt.close(figure)


def print_report(total_images: int, invalid_images: int, violations_per_class: dict[str, int]) -> None:
    print()
    print("## Dataset Validation Report")
    print()
    print(f"Total images: {total_images}")
    print(f"Invalid images: {invalid_images}")
    print()
    for class_name in CLASS_ORDER:
        print(f"{CLASS_TITLES[class_name]} violations: {violations_per_class[class_name]}")


def main() -> int:
    random.seed(42)

    if not os.path.isdir(DATASET_ROOT):
        print(f"Dataset folder not found: {DATASET_ROOT}")
        return 1

    image_paths_by_class = scan_dataset(DATASET_ROOT)
    total_images = sum(len(paths) for paths in image_paths_by_class.values())
    if total_images == 0:
        print(f"No PNG files found under {DATASET_ROOT}")
        return 1

    records: list[dict[str, object]] = []
    violations_per_class = {class_name: 0 for class_name in CLASS_ORDER}
    invalid_images = 0

    for class_name in CLASS_ORDER:
        for image_path in image_paths_by_class.get(class_name, []):
            image = load_grayscale_image(image_path)
            if image is None:
                invalid_images += 1
                violations_per_class[class_name] += 1
                records.append(
                    {
                        "image_path": image_path,
                        "class_name": class_name,
                        "energy": np.nan,
                        "dominant_doppler_frequency_hz": np.nan,
                        "doppler_bandwidth_pixels": np.nan,
                        "doppler_bandwidth_hz": np.nan,
                        "spectral_entropy": np.nan,
                        "background_center_ratio": np.nan,
                        "hand_time_variance": np.nan,
                        "fan_peak_count": np.nan,
                        "fan_peak_ratio": np.nan,
                        "is_valid": False,
                        "violations": "Unreadable PNG image",
                    }
                )
                print(f"[WARN] {image_path}: Unreadable PNG image")
                continue

            record = validate_image(class_name, image_path, image)
            if not bool(record["is_valid"]):
                invalid_images += 1
                violations_per_class[class_name] += 1
                print(f"[WARN] {image_path}: {record['violations']}")
            records.append(record)

    statistics_frame = pd.DataFrame(records)
    statistics_frame.to_csv(CSV_OUTPUT, index=False)
    print(f"Saved Doppler statistics: {CSV_OUTPUT}")

    display_random_samples(image_paths_by_class)
    print_report(total_images, invalid_images, violations_per_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
