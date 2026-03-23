#!/usr/bin/env python3
"""Lightweight demo entry point for the FMCW radar AI pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_model.extract_physics_features import (  # noqa: E402
    compute_bandwidth,
    compute_energy_per_frequency,
    compute_energy_ratio,
    compute_harmonic_peak_count,
    compute_spectral_entropy,
    load_spectrogram,
)


CLASS_NAMES = ("hand", "fan", "sheet", "background")
RAW_DATASETS = ("synthetic_dataset", "processed_dataset", "CRO Readings")
MODEL_ARTIFACT_PATTERNS = ("*.pth", "*.pkl", "*.npy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quick FMCW radar pipeline demo.")
    parser.add_argument(
        "--sample-image",
        type=Path,
        help="Optional PNG spectrogram path for feature extraction.",
    )
    parser.add_argument(
        "--scan-project",
        action="store_true",
        help="Print a quick summary of datasets, scripts, and available artifacts.",
    )
    return parser.parse_args()


def count_pngs(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*.png") if path.is_file())


def find_first_png(root: Path) -> Path | None:
    if not root.exists():
        return None
    return next((path for path in root.rglob("*.png") if path.is_file()), None)


def list_model_artifacts() -> list[Path]:
    artifacts: list[Path] = []
    for pattern in MODEL_ARTIFACT_PATTERNS:
        artifacts.extend(PROJECT_ROOT.glob(pattern))
    return sorted({artifact.resolve() for artifact in artifacts})


def print_project_summary() -> None:
    print("=== FMCW Radar AI Project Summary ===")
    print(f"Project root: {PROJECT_ROOT}")
    print()
    print("Datasets:")
    for dataset_name in RAW_DATASETS:
        dataset_path = PROJECT_ROOT / dataset_name
        status = "present" if dataset_path.exists() else "missing"
        suffix = f" ({count_pngs(dataset_path)} PNG files)" if dataset_path.is_dir() else ""
        print(f"  - {dataset_name}: {status}{suffix}")
    print()
    print("Core modules:")
    for module_path in (
        PROJECT_ROOT / "src" / "signal_processing",
        PROJECT_ROOT / "src" / "ml_model",
        PROJECT_ROOT / "src" / "tracking",
    ):
        print(f"  - {module_path.relative_to(PROJECT_ROOT)}")
    print()
    artifacts = list_model_artifacts()
    print(f"Model artifacts found: {len(artifacts)}")
    for artifact in artifacts[:5]:
        print(f"  - {artifact.name}")
    if len(artifacts) > 5:
        print(f"  - ... and {len(artifacts) - 5} more")


def resolve_demo_image(sample_image: Path | None) -> Path | None:
    if sample_image is not None:
        return sample_image if sample_image.is_absolute() else (PROJECT_ROOT / sample_image)

    for dataset_name in ("synthetic_dataset", "processed_dataset"):
        candidate = find_first_png(PROJECT_ROOT / dataset_name)
        if candidate is not None:
            return candidate
    return None


def describe_spectrogram(image_path: Path) -> None:
    spectrogram = load_spectrogram(str(image_path))
    energy_per_freq = compute_energy_per_frequency(spectrogram)

    dominant_freq = int(energy_per_freq.argmax()) if energy_per_freq.size else 0
    _, _, bandwidth = compute_bandwidth(energy_per_freq)
    entropy = compute_spectral_entropy(energy_per_freq)
    energy_ratio = compute_energy_ratio(energy_per_freq)
    harmonic_count = compute_harmonic_peak_count(energy_per_freq)

    print()
    print("=== Sample Feature Extraction ===")
    print(f"Image: {image_path}")
    print(f"Shape: {spectrogram.shape[0]} x {spectrogram.shape[1]}")
    print(f"Dominant frequency bin: {dominant_freq}")
    print(f"Estimated bandwidth: {bandwidth}")
    print(f"Spectral entropy: {entropy:.4f}")
    print(f"Center energy ratio: {energy_ratio:.4f}")
    print(f"Harmonic peak count: {harmonic_count}")


def main() -> int:
    args = parse_args()

    print("FMCW Radar AI demo pipeline")
    print("===========================")

    if args.scan_project or args.sample_image is None:
        print_project_summary()

    image_path = resolve_demo_image(args.sample_image)
    if image_path is None:
        print()
        print("No sample PNG spectrogram found. Provide --sample-image to run feature extraction.")
        return 0

    if not image_path.exists():
        print()
        print(f"Sample image not found: {image_path}")
        return 1

    describe_spectrogram(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
