# FMCW Radar AI System

Python project for FMCW radar signal processing, feature extraction, tracking, and physics-guided deep learning experiments.

## Repository Layout

```text
.
|-- src/
|   |-- signal_processing/
|   |-- detection/
|   |-- tracking/
|   `-- ml_model/
|-- demo/
|-- results/
|-- docs/
`-- archive/
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/tl5275/fmcw-radar-pipeline.git
cd fmcw-radar-pipeline
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the demo pipeline

Project scan plus sample feature extraction:

```bash
python demo/demo_pipeline.py --scan-project
```

Run on a specific spectrogram image:

```bash
python demo/demo_pipeline.py --sample-image synthetic_dataset/background/background_aug_0001.png
```

### 5. Run core scripts

Generate synthetic spectrogram data:

```bash
python src/signal_processing/generate_synthetic_dataset.py
```

Build the micro-Doppler dataset:

```bash
python src/signal_processing/build_microdoppler_dataset.py
```

Extract physics features:

```bash
python src/ml_model/extract_physics_features.py
```

Train the improved physics-guided CNN:

```bash
python src/ml_model/train_physics_guided_cnn_improved.py
```

Train the tracking-oriented radar model:

```bash
python src/tracking/train_radar_kalman_model.py
```

## Notes

- Raw datasets, checkpoints, cache files, and generated artifacts are excluded from Git with `.gitignore`.
- Some training scripts expect local dataset folders such as `synthetic_dataset/` and `processed_dataset/` to exist in the repository root.
- Archived experimental scripts are stored under `archive/`.
