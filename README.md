# 🚀 AI-Driven FMCW Radar Target Detection & Tracking System

An end-to-end radar perception system combining classical signal processing with physics-guided deep learning, built using real X-band FMCW radar measurements.

---

## 🧠 Overview

This project implements a complete radar processing pipeline:

- 📡 Beat signal acquisition from real radar hardware
- 📊 Range–Doppler processing (FFT-based)
- 🎯 Target detection using CA-CFAR
- 📍 Target tracking using Kalman Filter
- 🧠 Motion classification using CNN + physics-based features

Unlike conventional approaches, this system integrates **physically interpretable Doppler features with deep learning**, improving robustness and interpretability.

---

## ⚙️ System Architecture

![Pipeline](results/images/pipeline.png)

> End-to-end pipeline from raw radar signals → detection → tracking → classification

---

## 📡 Radar Processing Pipeline

1. Beat signal extraction from oscilloscope data  
2. Range FFT → distance estimation  
3. Doppler FFT → velocity estimation  
4. Range–Doppler map generation  
5. CA-CFAR adaptive detection  
6. Kalman filtering for trajectory tracking  
7. Micro-Doppler spectrogram generation  
8. CNN + Physics feature fusion classification  

---

## 🧠 Key Contributions

- ✅ End-to-end FMCW radar system using real experimental data  
- ✅ Physics-guided deep learning (feature-level fusion)  
- ✅ Detection-aware evaluation metric (**Pe2E**)  
- ✅ Integration of signal processing + AI + tracking  

---

## 📊 Results

| Model | Accuracy |
|------|--------|
| CNN Baseline | 71.93% |
| CNN + Physics Features | **75.44%** |

✔ Physics-based features improve classification robustness  
✔ Kalman filtering improves temporal stability  

---

## 🎥 Demo

🚧 Demo video coming soon

---

## 🛠️ Tech Stack

- Python (NumPy, SciPy, Pandas)
- PyTorch (Deep Learning)
- Signal Processing (FFT, CFAR)
- Radar Systems (FMCW, Doppler Processing)

---

## 📂 Project Structure

src/
├── signal_processing/
├── detection/
├── tracking/
├── ml_model/

demo/
results/
docs/


---

## ▶️ How to Run

```bash
git clone https://github.com/tl5275/fmcw-radar-ai-system.git
cd fmcw-radar-ai-system

pip install -r requirements.txt

python demo/demo_pipeline.py
