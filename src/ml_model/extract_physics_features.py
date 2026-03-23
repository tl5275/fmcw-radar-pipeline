#!/usr/bin/env python3
"""Extract physics-based radar features from synthetic micro-Doppler spectrograms."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


DATASET_ROOT = "synthetic_dataset"
CSV_OUTPUT = "physics_features.csv"
NPY_OUTPUT = "physics_features.npy"
CLASS_NAMES = ("hand", "fan", "sheet", "background")


def scan_dataset(dataset_root: str) -> dict[str, list[str]]:
    """Collect PNG paths under the expected synthetic dataset class folders."""
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    image_paths_by_class: dict[str, list[str]] = {class_name: [] for class_name in CLASS_NAMES}
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Expected class folder missing: {class_dir}")

        for current_root, _, filenames in os.walk(class_dir):
            for filename in sorted(filenames):
                if filename.lower().endswith(".png"):
                    image_paths_by_class[class_name].append(os.path.join(current_root, filename))

    return image_paths_by_class


def load_spectrogram(image_path: str) -> np.ndarray:
    """Load a spectrogram PNG as a normalized grayscale array in [0, 1]."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read spectrogram image: {image_path}")
    return image.astype(np.float32) / 255.0


def compute_energy_per_frequency(spectrogram: np.ndarray) -> np.ndarray:
    """Average energy over time to obtain the Doppler energy profile."""
    return np.mean(spectrogram, axis=1).astype(np.float64)


def compute_bandwidth(energy_per_freq: np.ndarray) -> tuple[int, int, int]:
    """Estimate the energy-containing Doppler bandwidth using 5-95% cumulative energy."""
    total_energy = float(np.sum(energy_per_freq))
    if total_energy <= 0.0:
        return 0, 0, 0

    cumulative_energy = np.cumsum(energy_per_freq) / total_energy
    f_low = int(np.searchsorted(cumulative_energy, 0.05, side="left"))
    f_high = int(np.searchsorted(cumulative_energy, 0.95, side="left"))
    bandwidth = max(f_high - f_low, 0)
    return f_low, f_high, bandwidth


def compute_spectral_entropy(energy_per_freq: np.ndarray) -> float:
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
    return float(center_energy / total_energy)


def compute_harmonic_peak_count(energy_per_freq: np.ndarray) -> int:
    """Count prominent peaks in the Doppler energy profile."""
    if energy_per_freq.size < 11:
        return 0

    smoothed = savgol_filter(energy_per_freq, 11, 2, mode="interp")
    prominence = max(0.02, 0.15 * float(np.std(smoothed)))
    height = float(np.mean(smoothed))
    peaks, _ = find_peaks(smoothed, height=height, prominence=prominence, distance=6)
    return int(len(peaks))


def extract_feature_record(image_path: str, label: str) -> dict[str, object]:
    spectrogram = load_spectrogram(image_path)
    energy_per_freq = compute_energy_per_frequency(spectrogram)

    dominant_freq = int(np.argmax(energy_per_freq)) if energy_per_freq.size else 0
    _, _, bandwidth = compute_bandwidth(energy_per_freq)
    entropy = compute_spectral_entropy(energy_per_freq)
    energy_ratio = compute_energy_ratio(energy_per_freq)
    harmonic_count = compute_harmonic_peak_count(energy_per_freq)

    return {
        "image": os.path.basename(image_path),
        "label": label,
        "dominant_freq": dominant_freq,
        "bandwidth": bandwidth,
        "entropy": entropy,
        "energy_ratio": energy_ratio,
        "harmonic_count": harmonic_count,
    }


def print_summary(total_samples: int, samples_per_class: dict[str, int]) -> None:
    print()
    print("## Feature Extraction Summary")
    print()
    print(f"Total samples: {total_samples}")
    print()
    for class_name in CLASS_NAMES:
        print(f"{class_name.capitalize()}: {samples_per_class[class_name]}")


def main() -> int:
    image_paths_by_class = scan_dataset(DATASET_ROOT)
    records: list[dict[str, object]] = []
    samples_per_class = {class_name: 0 for class_name in CLASS_NAMES}

    for class_name in CLASS_NAMES:
        for image_path in image_paths_by_class[class_name]:
            try:
                record = extract_feature_record(image_path, class_name)
            except RuntimeError as exc:
                print(f"[WARN] Skipping {image_path}: {exc}")
                continue

            records.append(record)
            samples_per_class[class_name] += 1

    if not records:
        print("No features were extracted.")
        return 1

    feature_frame = pd.DataFrame(
        records,
        columns=[
            "image",
            "label",
            "dominant_freq",
            "bandwidth",
            "entropy",
            "energy_ratio",
            "harmonic_count",
        ],
    )
    feature_frame.to_csv(CSV_OUTPUT, index=False)

    feature_matrix = feature_frame[
        ["dominant_freq", "bandwidth", "entropy", "energy_ratio", "harmonic_count"]
    ].to_numpy(dtype=np.float32)
    np.save(NPY_OUTPUT, feature_matrix)

    print(f"Saved feature table: {CSV_OUTPUT}")
    print(f"Saved feature matrix: {NPY_OUTPUT}")
    print_summary(total_samples=len(records), samples_per_class=samples_per_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
