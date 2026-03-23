#!/usr/bin/env python3
"""Convert radar oscilloscope recordings into a micro-Doppler image dataset."""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
import numpy as np
from scipy.signal import butter, sosfiltfilt, spectrogram

try:
    from tqdm import tqdm
except ImportError:
    class _TqdmFallback:
        """Small tqdm-compatible wrapper used when tqdm is unavailable."""

        def __call__(self, iterable, **_: object):
            return iterable

        @staticmethod
        def write(message: str) -> None:
            print(message)

    tqdm = _TqdmFallback()

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABEL_ORDER = ("hand", "fan", "sheet", "background")
LABEL_KEYWORDS = {
    "hand movement": "hand",
    "fan movement": "fan",
    "sheet movement": "sheet",
    "background": "background",
}
CHANNEL_FILE_PATTERN = re.compile(r"(?P<prefix>.*?)CH(?P<channel>[12])\.CSV$", re.IGNORECASE)
INTERVAL_KEYWORDS = (
    "sample interval",
    "sampling interval",
    "x increment",
    "xincr",
    "xincrement",
    "time step",
    "delta t",
    "dt",
)
UNIT_SCALE = {
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1.0e-3,
    "us": 1.0e-6,
    "µs": 1.0e-6,
    "μs": 1.0e-6,
    "ns": 1.0e-9,
}


class DataFormatError(RuntimeError):
    """Raised when a CSV file cannot be interpreted as a waveform."""


class DataQualityError(RuntimeError):
    """Raised when a recording fails validation checks."""


@dataclass(frozen=True)
class TrialRecord:
    label: str
    trial_dir: Path
    tx_path: Path
    rx_path: Path
    recording_key: str
    source_group: str

    @property
    def progress_name(self) -> str:
        return f"{self.source_group} / {self.trial_dir.name}"


@dataclass(frozen=True)
class ProcessingConfig:
    input_root: Path
    output_root: Path
    image_size: int
    highpass_hz: float
    band_low_hz: float
    band_high_hz: float
    min_energy: float
    peak_prominence_db: float
    background_peak_prominence_db: float


def parse_args() -> ProcessingConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Convert radar oscilloscope recordings inside 'CRO Readings' into a "
            "micro-Doppler spectrogram dataset suitable for machine learning."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("CRO Readings"),
        help="Root dataset folder containing movement folders and trial folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("processed_dataset"),
        help="Destination folder for the generated spectrogram image dataset.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square PNG size in pixels for each saved spectrogram image.",
    )
    parser.add_argument(
        "--highpass-hz",
        type=float,
        default=20.0,
        help="High-pass cutoff frequency used to remove low-frequency leakage.",
    )
    parser.add_argument(
        "--band-low-hz",
        type=float,
        default=20.0,
        help="Band-pass lower cutoff frequency for motion Doppler content.",
    )
    parser.add_argument(
        "--band-high-hz",
        type=float,
        default=300.0,
        help="Band-pass upper cutoff frequency for motion Doppler content.",
    )
    parser.add_argument(
        "--min-energy",
        type=float,
        default=1.0e-9,
        help="Minimum mean-square signal energy required after filtering.",
    )
    parser.add_argument(
        "--peak-prominence-db",
        type=float,
        default=6.0,
        help="Minimum peak prominence required for motion classes.",
    )
    parser.add_argument(
        "--background-peak-prominence-db",
        type=float,
        default=3.0,
        help="Minimum peak prominence required for background recordings.",
    )
    args = parser.parse_args()
    return ProcessingConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        image_size=args.image_size,
        highpass_hz=args.highpass_hz,
        band_low_hz=args.band_low_hz,
        band_high_hz=args.band_high_hz,
        min_energy=args.min_energy,
        peak_prominence_db=args.peak_prominence_db,
        background_peak_prominence_db=args.background_peak_prominence_db,
    )


def infer_label(relative_parts: tuple[str, ...]) -> Optional[str]:
    lowered_parts = [part.lower() for part in relative_parts]
    lowered_path = " / ".join(lowered_parts)
    for keyword, label in LABEL_KEYWORDS.items():
        if keyword in lowered_path:
            return label

    if len(relative_parts) == 1 and re.fullmatch(r"ALL\d+", relative_parts[0], re.IGNORECASE):
        return "background"

    return None


def build_channel_pairs(directory: Path, filenames: Iterable[str]) -> dict[str, tuple[Path, Path]]:
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)
    for filename in filenames:
        match = CHANNEL_FILE_PATTERN.match(filename)
        if match is None:
            continue

        channel = f"CH{match.group('channel')}"
        grouped[match.group("prefix")][channel] = directory / filename

    pairs: dict[str, tuple[Path, Path]] = {}
    for prefix, channels in grouped.items():
        if "CH1" in channels and "CH2" in channels:
            pairs[prefix or directory.name] = (channels["CH1"], channels["CH2"])
    return pairs


def discover_trials(input_root: Path) -> list[TrialRecord]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    trials: list[TrialRecord] = []
    for current_dir, _, filenames in os.walk(input_root):
        if not filenames:
            continue

        directory = Path(current_dir)
        pairs = build_channel_pairs(directory, filenames)
        if not pairs:
            continue

        relative_parts = directory.relative_to(input_root).parts
        label = infer_label(relative_parts)
        if label is None:
            tqdm.write(f"Skipping unlabeled trial folder: {directory}")
            continue

        source_group = "background" if label == "background" else relative_parts[0]
        for recording_key, (tx_path, rx_path) in sorted(pairs.items()):
            trials.append(
                TrialRecord(
                    label=label,
                    trial_dir=directory,
                    tx_path=tx_path,
                    rx_path=rx_path,
                    recording_key=recording_key,
                    source_group=source_group,
                )
            )

    trials.sort(key=lambda item: (LABEL_ORDER.index(item.label), str(item.trial_dir), item.recording_key))
    return trials


def read_text_lines(csv_path: Path) -> list[str]:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return csv_path.read_text(encoding=encoding).splitlines()
        except UnicodeError:
            continue
    return csv_path.read_bytes().decode("latin-1", errors="ignore").splitlines()


def detect_delimiter(lines: list[str]) -> str:
    candidates = [",", "\t", ";"]
    best_delimiter = ","
    best_score = -1
    for delimiter in candidates:
        score = sum(line.count(delimiter) for line in lines[:50])
        if score > best_score:
            best_score = score
            best_delimiter = delimiter
    return best_delimiter


def safe_float(token: str) -> Optional[float]:
    cleaned = token.strip().strip('"').replace("\x00", "")
    if not cleaned:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_interval_from_header(lines: list[str]) -> Optional[float]:
    for raw_line in lines[:120]:
        line = raw_line.lower().replace("μ", "µ")
        for keyword in INTERVAL_KEYWORDS:
            if keyword not in line:
                continue

            pattern = re.compile(
                rf"{re.escape(keyword)}[^0-9+\-]*"
                r"(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
                r"\s*(?P<unit>[a-zµ]*)"
            )
            match = pattern.search(line)
            if match is None:
                continue

            value = float(match.group("value"))
            unit = match.group("unit")
            scale = UNIT_SCALE.get(unit, 1.0)
            interval = value * scale
            if interval > 0.0:
                return interval
    return None


@dataclass(frozen=True)
class ParsedLine:
    index: int
    tokens: list[str]
    values: list[Optional[float]]
    field_count: int
    numeric_count: int


def parse_csv_lines(lines: list[str], delimiter: str) -> list[ParsedLine]:
    parsed_lines: list[ParsedLine] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        tokens = [token.strip() for token in stripped.split(delimiter)]
        values = [safe_float(token) for token in tokens]
        numeric_count = sum(value is not None for value in values)
        parsed_lines.append(
            ParsedLine(
                index=index,
                tokens=tokens,
                values=values,
                field_count=len(tokens),
                numeric_count=numeric_count,
            )
        )
    return parsed_lines


def select_numeric_block(parsed_lines: list[ParsedLine]) -> list[ParsedLine]:
    if not parsed_lines:
        raise DataFormatError("CSV file is empty or contains no readable rows.")

    best_start = 0
    best_end = 0
    best_score = -1.0
    current_start: Optional[int] = None
    current_fields = 0

    # Oscilloscope exports often contain a metadata preamble followed by one
    # long, stable numeric table. We keep the longest block with a consistent
    # column count instead of assuming a fixed number of header rows.
    for index, entry in enumerate(parsed_lines):
        qualifies = entry.numeric_count >= 1
        if qualifies and current_start is None:
            current_start = index
            current_fields = entry.field_count
        elif qualifies and entry.field_count == current_fields:
            pass
        else:
            if current_start is not None:
                run_length = index - current_start
                score = run_length + min(current_fields, 4) * 0.25
                if score > best_score:
                    best_start = current_start
                    best_end = index
                    best_score = score
            current_start = index if qualifies else None
            current_fields = entry.field_count if qualifies else 0

    if current_start is not None:
        run_length = len(parsed_lines) - current_start
        score = run_length + min(current_fields, 4) * 0.25
        if score > best_score:
            best_start = current_start
            best_end = len(parsed_lines)
            best_score = score

    block = parsed_lines[best_start:best_end]
    if len(block) < 8:
        raise DataFormatError("Could not find a stable numeric waveform block in the CSV file.")
    return block


def infer_header_tokens(parsed_lines: list[ParsedLine], first_block_line_index: int, field_count: int) -> Optional[list[str]]:
    for position in range(first_block_line_index - 1, -1, -1):
        candidate = parsed_lines[position]
        if candidate.field_count != field_count:
            continue
        if candidate.numeric_count == 0:
            return [token.lower() for token in candidate.tokens]
        break
    return None


def build_numeric_table(block: list[ParsedLine]) -> np.ndarray:
    field_count = block[0].field_count
    table = np.full((len(block), field_count), np.nan, dtype=np.float64)
    for row_index, entry in enumerate(block):
        if entry.field_count != field_count:
            continue
        for column_index, value in enumerate(entry.values):
            if value is not None and math.isfinite(value):
                table[row_index, column_index] = value
    return table


def choose_time_column(table: np.ndarray, headers: Optional[list[str]]) -> Optional[int]:
    candidates: list[tuple[float, int]] = []
    for column_index in range(table.shape[1]):
        column = table[:, column_index]
        finite = column[np.isfinite(column)]
        if finite.size < 8:
            continue

        diffs = np.diff(finite)
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size < 4:
            continue

        positive_ratio = float(np.mean(diffs > 0.0))
        median_step = float(np.median(diffs))
        variability = float(np.median(np.abs(diffs - median_step))) / max(abs(median_step), 1.0e-12)
        header_bonus = 0.0
        if headers and column_index < len(headers):
            header = headers[column_index]
            if "time" in header or "sec" in header or header in {"t", "x"}:
                header_bonus = 1.0

        if positive_ratio < 0.85 or median_step <= 0.0:
            continue

        candidates.append((positive_ratio - variability + header_bonus, column_index))

    if not candidates:
        return None
    return max(candidates)[1]


def choose_waveform_column(table: np.ndarray, headers: Optional[list[str]], excluded_column: Optional[int]) -> Optional[int]:
    candidates: list[tuple[float, int]] = []
    for column_index in range(table.shape[1]):
        if excluded_column is not None and column_index == excluded_column:
            continue

        column = table[:, column_index]
        finite = column[np.isfinite(column)]
        if finite.size < 8:
            continue

        variability = float(np.nanstd(finite))
        if variability <= 0.0:
            continue

        diffs = np.diff(finite)
        monotonic_bias = abs(float(np.nanmean(np.sign(diffs)))) if diffs.size else 0.0
        header_bonus = 0.0
        if headers and column_index < len(headers):
            header = headers[column_index]
            if any(keyword in header for keyword in ("voltage", "volt", "channel", "signal", "amplitude", "value", "ch")):
                header_bonus += 1.0
            if "time" in header:
                header_bonus -= 2.0

        score = math.log10(variability + 1.0e-12) - monotonic_bias + header_bonus
        candidates.append((score, column_index))

    if candidates:
        return max(candidates)[1]

    if table.shape[1] == 1:
        return 0
    return None


def extract_waveform_and_fs(csv_path: Path) -> tuple[np.ndarray, float]:
    lines = read_text_lines(csv_path)
    if not lines:
        raise DataFormatError(f"No readable text found in {csv_path}.")

    delimiter = detect_delimiter(lines)
    parsed_lines = parse_csv_lines(lines, delimiter)
    numeric_block = select_numeric_block(parsed_lines)
    header_tokens = infer_header_tokens(
        parsed_lines=parsed_lines,
        first_block_line_index=parsed_lines.index(numeric_block[0]),
        field_count=numeric_block[0].field_count,
    )
    table = build_numeric_table(numeric_block)
    dt_from_header = extract_interval_from_header(lines)

    time_column = choose_time_column(table, header_tokens)
    waveform_column = choose_waveform_column(table, header_tokens, excluded_column=time_column)
    if waveform_column is None:
        raise DataFormatError(f"Could not determine waveform column for {csv_path}.")

    waveform = table[:, waveform_column]
    valid_mask = np.isfinite(waveform)

    dt: Optional[float] = None
    if time_column is not None:
        time_values = table[:, time_column]
        valid_mask &= np.isfinite(time_values)
        time_values = time_values[valid_mask]
        positive_diffs = np.diff(time_values)
        positive_diffs = positive_diffs[np.isfinite(positive_diffs) & (positive_diffs > 0.0)]
        if positive_diffs.size:
            dt_from_time = float(np.median(positive_diffs))
            if dt_from_header is not None:
                # Some exports include a sample index column instead of true
                # time; when header metadata strongly disagrees, trust the
                # explicit scope interval instead.
                ratio = max(dt_from_time, dt_from_header) / min(dt_from_time, dt_from_header)
                dt = dt_from_header if ratio > 10.0 else dt_from_time
            else:
                dt = dt_from_time

    if dt is None:
        dt = dt_from_header

    waveform = waveform[valid_mask]
    waveform = waveform[np.isfinite(waveform)]

    if waveform.size < 128:
        raise DataFormatError(f"Waveform in {csv_path} is too short for STFT processing.")
    if dt is None or not math.isfinite(dt) or dt <= 0.0:
        # fallback sampling interval when CSV lacks time metadata
        dt = 1e-5
        print(f"[WARN] Using fallback sampling interval (dt=1e-5) for {csv_path}")

    fs = 1.0 / dt
    return waveform.astype(np.float64), float(fs)


def align_waveforms(tx_waveform: np.ndarray, rx_waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = min(tx_waveform.size, rx_waveform.size)
    if length < 128:
        raise DataQualityError("Aligned TX/RX waveforms are too short for STFT processing.")
    return tx_waveform[:length], rx_waveform[:length]


def apply_butterworth(signal_data: np.ndarray, fs: float, cutoff: float | tuple[float, float], btype: str) -> np.ndarray:
    nyquist = 0.5 * fs
    if btype == "highpass":
        normalized = float(cutoff) / nyquist
        if normalized >= 1.0:
            raise DataQualityError(f"Sampling rate {fs:.2f} Hz is too low for {cutoff:.2f} Hz high-pass filtering.")
        sos = butter(4, normalized, btype="highpass", output="sos")
    else:
        low_hz, high_hz = cutoff
        low = low_hz / nyquist
        high = min(high_hz / nyquist, 0.99)
        if low >= high:
            raise DataQualityError(
                f"Sampling rate {fs:.2f} Hz is too low for the requested {low_hz:.2f}-{high_hz:.2f} Hz band-pass filter."
            )
        sos = butter(4, [low, high], btype="bandpass", output="sos")

    try:
        return sosfiltfilt(sos, signal_data)
    except ValueError as exc:
        raise DataQualityError(f"Filtering failed because the signal is too short: {exc}") from exc


def compute_micro_doppler(beat_signal: np.ndarray, fs: float, config: ProcessingConfig) -> tuple[np.ndarray, np.ndarray]:
    prefiltered = apply_butterworth(beat_signal, fs, config.highpass_hz, btype="highpass")
    filtered = apply_butterworth(
        prefiltered,
        fs,
        (config.band_low_hz, config.band_high_hz),
        btype="bandpass",
    )
    return filtered, spectrogram_from_signal(filtered, fs)


def spectrogram_from_signal(filtered_signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    frequencies, _, power = spectrogram(
        filtered_signal,
        fs=fs,
        window="hann",
        nperseg=128,
        noverlap=96,
        nfft=512,
        detrend=False,
        scaling="density",
        mode="psd",
    )

    if power.size == 0:
        raise DataQualityError("STFT returned an empty spectrogram.")

    frequency_mask = (frequencies >= 0.0) & (frequencies <= min(500.0, fs * 0.5))
    frequencies = frequencies[frequency_mask]
    power = power[frequency_mask, :]
    if power.size == 0 or power.shape[0] < 2 or power.shape[1] < 2:
        raise DataQualityError("Not enough frequency or time bins remain after spectrogram cropping.")

    power_db = 10.0 * np.log10(power + 1.0e-12)
    return frequencies, power_db


def validate_recording(
    filtered_signal: np.ndarray,
    power_db: np.ndarray,
    frequencies: np.ndarray,
    label: str,
    config: ProcessingConfig,
) -> None:
    energy = float(np.mean(np.square(filtered_signal)))
    if not math.isfinite(energy) or energy < config.min_energy:
        raise DataQualityError(f"Signal energy {energy:.3e} is below the configured threshold.")

    peak_index = np.unravel_index(np.argmax(power_db), power_db.shape)
    peak_db = float(power_db[peak_index])
    noise_floor_db = float(np.median(power_db))
    prominence_db = peak_db - noise_floor_db
    required_prominence = (
        config.background_peak_prominence_db if label == "background" else config.peak_prominence_db
    )
    if prominence_db < required_prominence:
        raise DataQualityError(
            f"Doppler peak prominence {prominence_db:.2f} dB is below the required {required_prominence:.2f} dB."
        )

    dominant_frequency = float(frequencies[peak_index[0]])
    if dominant_frequency < 0.0 or dominant_frequency > 500.0:
        raise DataQualityError(f"Doppler peak {dominant_frequency:.2f} Hz lies outside the expected range.")


def enhance_spectrogram(power_db: np.ndarray) -> np.ndarray:
    finite_values = power_db[np.isfinite(power_db)]
    if finite_values.size == 0:
        raise DataQualityError("Spectrogram contains no finite power values.")

    lower_clip = float(np.percentile(finite_values, 1.0))
    upper_clip = float(np.percentile(finite_values, 99.5))
    # Clip tails before normalization so a few hot pixels do not wash out the
    # motion micro-Doppler structure.
    clipped = np.clip(power_db, lower_clip, upper_clip)
    normalized = clipped - float(np.max(clipped))
    normalized = np.clip(normalized, -80.0, 0.0)
    image = (normalized + 80.0) / 80.0
    return np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)


def resize_axis(data: np.ndarray, output_size: int, axis: int) -> np.ndarray:
    if data.shape[axis] == output_size:
        return data

    old_positions = np.linspace(0.0, 1.0, data.shape[axis])
    new_positions = np.linspace(0.0, 1.0, output_size)

    if axis == 0:
        resized_columns = [
            np.interp(new_positions, old_positions, data[:, column_index])
            for column_index in range(data.shape[1])
        ]
        return np.stack(resized_columns, axis=1)

    resized_rows = [
        np.interp(new_positions, old_positions, data[row_index, :])
        for row_index in range(data.shape[0])
    ]
    return np.stack(resized_rows, axis=0)


def resize_image(image: np.ndarray, output_size: int) -> np.ndarray:
    resized = resize_axis(image, output_size, axis=0)
    resized = resize_axis(resized, output_size, axis=1)
    return np.clip(resized, 0.0, 1.0)


def save_spectrogram_image(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, np.flipud(image), cmap="gray", vmin=0.0, vmax=1.0, format="png")


def process_trial(trial: TrialRecord, config: ProcessingConfig, sample_index: int) -> Path:
    tx_waveform, tx_fs = extract_waveform_and_fs(trial.tx_path)
    rx_waveform, rx_fs = extract_waveform_and_fs(trial.rx_path)

    if not np.isclose(tx_fs, rx_fs, rtol=0.01, atol=1.0e-6):
        raise DataQualityError(
            f"Sampling-rate mismatch between TX ({tx_fs:.2f} Hz) and RX ({rx_fs:.2f} Hz) recordings."
        )

    tx_waveform, rx_waveform = align_waveforms(tx_waveform, rx_waveform)
    beat_signal = rx_waveform - tx_waveform
    beat_signal = beat_signal - np.mean(beat_signal)
    filtered_signal, (frequencies, power_db) = compute_micro_doppler(beat_signal, tx_fs, config)
    validate_recording(filtered_signal, power_db, frequencies, trial.label, config)

    image = enhance_spectrogram(power_db)
    image = resize_image(image, config.image_size)

    filename = f"{trial.label}_{sample_index:04d}.png"
    output_path = config.output_root / trial.label / filename
    save_spectrogram_image(image, output_path)
    return output_path


def print_summary(saved_counts: dict[str, int], skipped_reasons: Counter[str]) -> None:
    print()
    print("Dataset summary:")
    print(f"Hand samples: {saved_counts['hand']}")
    print(f"Fan samples: {saved_counts['fan']}")
    print(f"Sheet samples: {saved_counts['sheet']}")
    print(f"Background samples: {saved_counts['background']}")
    if skipped_reasons:
        print()
        print(f"Skipped recordings: {sum(skipped_reasons.values())}")
        for reason, count in skipped_reasons.most_common():
            print(f"  {reason}: {count}")


def main() -> int:
    config = parse_args()
    trials = discover_trials(config.input_root)
    if not trials:
        raise SystemExit(f"No trial folders with CH1/CH2 CSV pairs were found under {config.input_root}.")

    for label in LABEL_ORDER:
        (config.output_root / label).mkdir(parents=True, exist_ok=True)

    saved_counts: dict[str, int] = {label: 0 for label in LABEL_ORDER}
    skipped_reasons: Counter[str] = Counter()

    for trial in tqdm(trials, desc="Processing trials", unit="trial"):
        tqdm.write(f"Processing {trial.progress_name}")
        try:
            next_index = saved_counts[trial.label] + 1
            output_path = process_trial(trial, config, next_index)
            saved_counts[trial.label] = next_index
            tqdm.write(f"Saved {output_path}")
        except (DataFormatError, DataQualityError, OSError) as exc:
            skipped_reasons[str(exc)] += 1
            tqdm.write(f"Skipping {trial.progress_name}: {exc}")

    print_summary(saved_counts, skipped_reasons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
