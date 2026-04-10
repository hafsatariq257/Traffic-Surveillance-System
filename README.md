# 🚦 TrafficAI Tracker
 
> Real-time traffic monitoring system with AI-powered vehicle detection, speed estimation, and violation tracking.
 
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-ff4b4b?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff8c42?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## Live Website 

Visit the website live at : https://trafficai-tracker.streamlit.app/

---
 
## ✨ Features
 
- 🚗 **Vehicle Detection** — detects Cars, Motorcycles, Buses, Trucks, and Bicycles
- 📏 **Speed Estimation** — estimates speed of each tracked vehicle in km/h
- 🚨 **Violation Tracking** — flags vehicles that cross the stop line on a red signal or exceed speed limit
- 📊 **Live Stats** — real-time vehicle count and violation counter
- 📁 **Excel Export** — download full violations log as `.xlsx`
- 🎨 **Clean UI** — dark themed Streamlit app with Lottie animations
 
---
 
## 🛠️ Tech Stack
 
| Tool | Purpose |
|------|---------|
| [YOLOv8](https://github.com/ultralytics/ultralytics) | Object detection |
| [Streamlit](https://streamlit.io) | Web UI |
| [OpenCV](https://opencv.org) | Video processing |
| [streamlit-lottie](https://github.com/andfanilo/streamlit-lottie) | Animations |
| [openpyxl](https://openpyxl.readthedocs.io) | Excel logging |
 
---
 
## 🚀 Getting Started
 
### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/TrafficAI-Tracker.git
cd TrafficAI-Tracker
```
 
### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Mac/Linux
```
 
### 3. Install dependencies
```bash
pip install numpy==1.26.4
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
 
### 4. Run the app
```bash
streamlit run app.py
```
 
---
 
## 📂 Project Structure
 
```
TrafficAI-Tracker/
│
├── app.py                  # Main Streamlit app
├── inference.py            # YOLOv8 inference + vehicle tracking logic
├── requirements.txt        # Python dependencies
├── yolov8n.pt             # YOLOv8 nano model weights
│
├── utils/
│   └── excel_logger.py    # Violation logging to Excel
│
└── .streamlit/
    └── config.toml        # Streamlit config (upload size limit)
```
 
---
 
## ⚙️ Configuration
 
Adjust these in the sidebar at runtime:
 
| Setting | Default | Description |
|--------|---------|-------------|
| Stop line position | 55% | Vertical position of the stop line |
| Signal is RED | ON | Toggle red/green signal state |
| Speed limit | 60 km/h | Vehicles exceeding this are flagged |
 
---
 
## 📋 Requirements
 
See `requirements.txt`. Key packages:
 
```
streamlit
ultralytics
opencv-python
pandas
openpyxl
streamlit-lottie
requests
```
 
> ⚠️ **Note:** Install `numpy==1.26.4` and `torch==2.1.0` manually before running `pip install -r requirements.txt` to avoid version conflicts.
 
---
 
## 📄 License
 
[MIT](LICENSE)

 ---
 
 ## 🤝 Contact & Feedback
I'm always looking for ways to improve the "bloom" experience!
* **LinkedIn:** [https://www.linkedin.com/in/bushrasiraj/]
* **Portfolio:** [https://bushrasiraj-portfolio.lovable.app/]
* **Email:** [BushraSiraj586@gmail.com]
  
---

*Designed with ❤️ by **Bushra Siraj***

---

## 📸 Preview

<img width="930" height="425" alt="2026-04-10 01_00_36-" src="https://github.com/user-attachments/assets/9a94e7a1-f73d-471c-a085-dfad75609f4c" />
<img width="960" height="421" alt="2026-04-10 01_01_44-Settings" src="https://github.com/user-attachments/assets/e6dcf4c4-155d-49ca-99b8-732a9928b51a" />
