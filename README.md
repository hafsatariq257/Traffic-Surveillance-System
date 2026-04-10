🚦 TrafficAI TrackerA high-performance AI framework for real-time vehicle perception, kinematic speed estimation, and automated traffic law enforcement.🌐 Live DeploymentExperience the Customer-Intelligence-Engine in action:View Live App✨ Features🔍 Precision Detection — Identifies Cars, Motorcycles, Buses, Trucks, and Bicycles using YOLOv8.⚡ Kinematic Speed Estimation — Calculates real-time velocity in km/h for every tracked vehicle.🚨 Automated Enforcement — Automatically flags vehicles for Red Light jumps and Speed Limit violations.📊 Live Intelligence Stats — Real-time analytics showing vehicle throughput and total infractions.📁 Forensic Excel Export — Generate and download detailed violation logs as .xlsx.🎨 Professional UI — High-contrast Yellow & Black theme designed for maximum visibility.🛠️ Tech StackToolPurposeYOLOv8Neural Network for Object DetectionStreamlitDashboard & Web InterfaceOpenCVReal-time Video Stream ProcessingPandasData Structuring & AnalysisopenpyxlExcel Database Serialization🚀 Getting Started1. Clone the EngineBashgit clone https://github.com/hafsatariq257/TrafficAI-Tracker.git
cd TrafficAI-Tracker
2. Prepare the EnvironmentBashpython -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Mac/Linux
3. Install Core DependenciesBash# Critical: Install these versions first to ensure stability
pip install numpy==1.26.4
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
4. Ignite the AppBashstreamlit run app.py
📂 Project ArchitecturePlaintextTrafficAI-Tracker/
│
├── app.py                  # Main Dashboard (Yellow Theme)
├── inference.py            # AI Inference & Tracking Logic
├── requirements.txt        # System Dependencies
├── yolov8n.pt              # Neural Network Weights
│
├── utils/
│   └── excel_logger.py     # Forensic Logging Engine
│
└── .streamlit/
    └── config.toml         # Streamlit Performance Config
⚙️ Runtime CalibrationAdjust the detection parameters in the sidebar:SettingDefaultFunctional DescriptionStop line position55%Vertical boundary for signal enforcementSignal ToggleREDSwitch between signal states manuallySpeed Limit60 km/hThreshold for automated speeding tickets📄 LicenseDistributed under the MIT License.🤝 Connect & CollaborateGitHub: hafsatariq257LinkedIn:www.linkedin.com/in/hafsa-tariq-- Developed by Hafsa Tariq

PREVIEW:

<img width="1918" height="1078" alt="image" src="https://github.com/user-attachments/assets/33400964-25e6-44ab-a7af-4fa1d55fc571" />









screenshots here)
