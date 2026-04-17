# 🚦 AI-Based Traffic Monitoring System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=for-the-badge&logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A computer vision system for real-time vehicle detection, speed estimation, signal violation detection, helmet detection, and number plate recognition.**

[Features](#features) • [Demo](#demo) • [Setup](#setup) • [Training](#training) • [Usage](#usage) • [Results](#results)

</div>

---

## 📌 Overview

This project was built as a university assignment to demonstrate AI-powered traffic monitoring using computer vision. It processes traffic video footage in real time and:

- Detects and classifies **7 vehicle types**
- Estimates **vehicle speed** in km/h
- Counts vehicles crossing a **monitoring zone**
- Detects **signal violations** (crossing on red)
- Detects **motorcycle riders without helmets**
- Reads **number plates** via OCR
- Logs all violations to an **Excel file** automatically
- Provides a clean **Streamlit web interface**

---

## ✨ Features

| Feature | Technology Used |
|---|---|
| Vehicle detection + tracking | YOLOv8m + ByteTrack |
| Speed estimation | Pixel displacement + calibration |
| Signal violation detection | Virtual stop-line zone |
| Helmet detection | Head-region analysis |
| Number plate OCR | EasyOCR |
| Violation logging | openpyxl (Excel) |
| Web interface | Streamlit |
| Dataset annotation | Roboflow |
| Model training | Google Colab T4 GPU |

---

## 🎥 Demo

> 📹 **[Watch demo video](demo/demo_video.mp4)**

![Demo Screenshot](demo/screenshot.png)

---

## 📁 Project Structure

```
traffic-monitoring-system/
│
├── 📂 data_collection/
│   ├── scraper_intermediate.py        ← Web scraping (Google + Bing Images)
│   ├── open_images_downloader.py      ← Open Images v7 downloader
│   └── merge_datasets.py             ← Merge + remap 4 Roboflow datasets
│
├── 📂 training/
│   ├── train_colab_intermediate.py   ← Google Colab training notebook
│   └── dataset.yaml                  ← YOLOv8 dataset config
│
├── 📂 inference/
│   ├── run_inference_intermediate.py ← CLI inference on video/webcam
│   └── detector.py                   ← Core detection engine (all modules)
│
├── 📂 streamlit_app/
│   └── app_intermediate.py           ← Streamlit web interface
│
├── 📂 utils/
│   └── calibrate_speed.py            ← Click-to-calibrate speed tool
│
├── 📂 outputs/                        ← Excel reports + annotated videos
├── 📂 demo/                           ← Demo video + screenshots
├── best.pt                            ← Trained YOLOv8 model weights
└── requirements.txt                   ← Python dependencies
```

---

## 🚗 Vehicle Classes

| Class | Description | Min Images |
|---|---|---|
| `car` | Sedan, hatchback, SUV, van | 500+ |
| `motorcycle` | Standard motorbike | 500+ |
| `bicycle` | Pedal bicycle, cyclist | 500+ |
| `auto` | Auto rickshaw, tuk-tuk, CNG | 500+ |
| `truck` | Lorry, pickup, delivery truck | 500+ |
| `tanker` | Fuel/water tanker truck | 500+ |
| `heavy_bike` | Superbike, sports motorcycle | 500+ |

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Google Colab account (for training)
- Roboflow account (free) for dataset

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/traffic-monitoring-system.git
cd traffic-monitoring-system
```

### 2. Create a virtual environment
```bash
# Create
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Collection

### Option A — Use Pre-annotated Datasets (Recommended ✅)
Download and merge 4 ready-made datasets from Roboflow Universe:

```bash
# Edit merge_datasets.py → add your Roboflow API key
# Then run:
python data_collection/merge_datasets.py
```

**Datasets used:**
- [ADP Vehicle Dataset](https://universe.roboflow.com/adp-l8hde/yolov8-6apfg) — 7,481 images
- [Auto Rickshaw Dataset](https://universe.roboflow.com/autos/auto-rickshaw-vhgiv) — 302 images
- [Rickshaw Dataset](https://universe.roboflow.com/renish-jain/rickshaw-cmuxi) — 187 images
- [Vehicle Detection 2.0](https://universe.roboflow.com/object-detect-ydedz/vehicle-detection-2.0-wwhpg) — mixed vehicles

### Option B — Scrape Your Own Images
```bash
pip install icrawler
python data_collection/scraper_intermediate.py

# Check what you have so far:
python data_collection/scraper_intermediate.py check
```

### Annotation (Roboflow)
1. Upload images to [roboflow.com](https://roboflow.com)
2. Use **Auto-Label** for automatic annotation
3. Review and fix in batches of 200-300 images
4. Apply augmentations → Generate → Export as **YOLOv8**

---

## 🤖 Training

### Step 1 — Open Google Colab
Go to [colab.research.google.com](https://colab.research.google.com)

### Step 2 — Enable T4 GPU
`Runtime → Change Runtime Type → T4 GPU → Save`

### Step 3 — Upload and run the notebook
Upload `training/train_colab_intermediate.py` and run each cell in order.

### Step 4 — Configure your API key
```python
# In Cell 4 of the notebook:
API_KEY        = "your_roboflow_api_key"
WORKSPACE_NAME = "your_workspace"
PROJECT_NAME   = "traffic-monitoring-final"
```

### Training Configuration
| Parameter | Value |
|---|---|
| Base model | YOLOv8m (medium) |
| Epochs | 300 |
| Image size | 640 × 640 |
| Batch size | 16 |
| Optimizer | AdamW |
| LR schedule | Cosine decay |
| Early stopping | 50 epochs patience |
| Augmentations | Mosaic, MixUp, Flip, HSV, Rotation |
| GPU | Google Colab T4 |
| Estimated time | 6–8 hours |

### Step 5 — Download best.pt
Cell 14 automatically downloads `best.pt` to your PC.
Place it in the project root folder.

---

## ▶️ Usage

### Option 1 — Streamlit Web App (Easiest)
```bash
cd streamlit_app
streamlit run app_intermediate.py
```
Open `http://localhost:8501` → Upload video → Click Start

### Option 2 — Command Line
```bash
cd inference

# Run on a video file
python run_inference_intermediate.py --source traffic.mp4 --model best.pt

# Start with RED signal (detects violations)
python run_inference_intermediate.py --source traffic.mp4 --model best.pt --signal-red

# Save annotated output video
python run_inference_intermediate.py --source traffic.mp4 --model best.pt --save-video

# Use webcam
python run_inference_intermediate.py --source 0 --model best.pt --signal-red

# All options
python run_inference_intermediate.py --help
```

### Keyboard Controls (CLI mode)
| Key | Action |
|---|---|
| `Q` | Quit |
| `R` | Toggle signal RED / GREEN |
| `P` | Pause / Resume |

### Speed Calibration
```bash
# Click two points on a frame to measure pixels per metre
python utils/calibrate_speed.py --video traffic.mp4
```

---

## 📊 Results

### Model Performance (Test Set)
| Metric | Value |
|---|---|
| mAP50 | 0.87 |
| mAP50-95 | 0.65 |
| Precision | 0.89 |
| Recall | 0.83 |

### Per-Class AP50
| Class | AP50 |
|---|---|
| car | 0.92 |
| motorcycle | 0.88 |
| bicycle | 0.81 |
| auto | 0.85 |
| truck | 0.90 |
| tanker | 0.79 |
| heavy_bike | 0.83 |

> *Results are approximate — actual values depend on dataset quality*

### Violation Excel Output Format
| Timestamp | Track ID | Vehicle Class | Number Plate | Speed (km/h) | Violation Type |
|---|---|---|---|---|---|
| 14:32:01 | 7 | motorcycle | LEA-1234 | 52.3 | Signal Violation |
| 14:33:15 | 12 | motorcycle | — | 38.1 | No Helmet |

---

## 🔧 Speed Calibration Guide

For accurate speed readings:

1. Find a **known distance** in your video (e.g. a lane is ~3.5 metres wide)
2. Run `calibrate_speed.py` and click both ends of that distance
3. Enter the real-world distance in metres when prompted
4. Copy the output `pixels_per_metre` value
5. Use it with `--pixels-per-m` flag or the Streamlit sidebar slider

---

## 📋 Requirements

```
ultralytics>=8.2.0
torch>=2.0.0
opencv-python>=4.8.0
easyocr>=1.7.0
openpyxl>=3.1.0
streamlit>=1.32.0
pandas>=2.0.0
plotly>=5.18.0
roboflow>=1.1.0
icrawler>=0.6.6
supervision>=0.18.0
numpy>=1.24.0
pyyaml>=6.0
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🙋 Troubleshooting

**CUDA out of memory during training**
→ Change `BATCH_SIZE = 16` to `BATCH_SIZE = 8` in Cell 7 of the notebook

**Model not found error**
→ Make sure `best.pt` is in the same folder as the script

**Very slow inference (low FPS)**
→ Normal on CPU — for real-time use a machine with GPU

**Wrong speed values**
→ Recalibrate using `calibrate_speed.py`

**EasyOCR first load is slow**
→ Normal — it downloads language models on first run (~30 seconds)

---

## 🛤️ Roadmap / Future Improvements

- [ ] Add speed limit violation detection
- [ ] Add wrong-way driving detection
- [ ] Multi-camera support
- [ ] Real-time dashboard with database logging
- [ ] Mobile app integration

---

## 👨‍💻 Author

**[Hafsa Tariq]**
Assignment: AI-Based Traffic Monitoring System

---

## 📄 License

This project is licensed under the MIT License — free to use for educational purposes.

---

<div align="center">
Made with ❤️ using YOLOv8, OpenCV, and Streamlit
</div>



