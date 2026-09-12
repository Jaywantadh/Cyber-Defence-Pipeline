# Cyber Defence Pipeline

An autonomous, safety-gated, and auditable cyber threat defense pipeline built in pure Python.

This project processes raw network flow telemetry (CICIDS2017), normalizes traffic features, performs binary threat classification using machine learning, proposes defensive actions, quantifies operational risk, enforces an inverted human-in-the-loop safety gate, and records full execution trails into an immutable audit database with an operational Streamlit dashboard.

---

## Architecture Overview

```
Raw Flow Telemetry (CICIDS2017 CSV)
       │
       ▼
[Stage 1: Ingestion & Normalization] ──> modules/ingestion.py
       │ Clean NaNs/Infs & format 78 flow features into standardized event dict
       ▼
[Stage 2: Threat Detection ML Model] ──> modules/detection.py
       │ RandomForest binary classifier (77 features, excluding Destination Port shortcut)
       ▼
[Stage 3: Autonomous Network Agent]  ──> modules/network_agent.py
       │ Proposes candidate action (ISOLATE_FLOW vs FLAG_FOR_REVIEW vs NO_ACTION)
       ▼
[Stage 4: Risk Engine]               ──> modules/risk_engine.py
       │ Multiplies detection confidence by action severity weight -> [0.0 - 1.0]
       ▼
[Stage 5: Safety Gate Guardrail]     ──> modules/safety_gate.py
       │ Inverted safety logic: Risk >= 0.7 blocks execution -> NEEDS_HUMAN_APPROVAL
       ▼
[Stage 6: Immutable Audit Log]       ──> modules/audit_log.py
       │ Persists full 6-stage decision trail into logs/audit.db (SQLite)
       ▼
[Stage 7: Operational Dashboard]     ──> dashboard/app.py
         Real-time Streamlit monitoring, summary metrics, and decision inspection
```

---

## Features

- **Data Sanitization**: Handles CICIDS2017 header whitespace anomalies, infinite flow rates, and missing feature imputation.
- **Port-Independent ML Detection**: Trained with `Destination Port` excluded to ensure detection models learn authentic traffic dynamics (TCP window sizes, packet intervals, flow rates) rather than service-port shortcuts.
- **Data-Driven Action Thresholds**: Uses median flow packet rate (`HIGH_RATE_THRESHOLD = 5.88638 pkts/s`) from empirical attack quantile distributions to route automated responses.
- **Inverted Safety Guardrail**: Higher risk scores dictate **greater caution**—high-consequence actions on high-confidence threats are blocked pending explicit human authorization rather than executed blindly.
- **Immutable Audit Trail**: SQLite-backed decision persistence recording ground-truth comparison, model confidence, proposed action, risk scores, and safety gate reasoning.
- **Interactive UI**: Streamlit dashboard with full-database summary metrics, visually highlighted pending approvals, and expandable per-event reasoning.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Jaywantadh/Cyber-Defence-Pipeline.git
cd Cyber-Defence-Pipeline
```

### 2. Set Up Virtual Environment

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Acquisition

This pipeline is evaluated on the Canadian Institute for Cybersecurity **CICIDS2017** dataset.

1. Download the Tuesday working hours dataset (`Tuesday-WorkingHours.pcap_ISCX.csv`, containing benign traffic, FTP-Patator, and SSH-Patator brute-force attacks).
2. Place the file directly in the `data/raw/` directory:
   ```
   data/raw/Tuesday-WorkingHours.pcap_ISCX.csv
   ```

*(Due to GitHub file size limits, raw dataset CSV files are excluded via `.gitignore`.)*

---

## How to Run

### Step 1: Inspect the Raw Dataset
Verify the CSV header formatting, row counts, and attack distribution:
```bash
python scripts/inspect_dataset.py
```

### Step 2: Train & Evaluate the Detection Model
Trains the RandomForest baseline classifier, prints the classification report and top flow feature importances, and saves the serialized model to `models/threat_detector.pkl`:
```bash
python modules/detection.py
```

### Step 3: Run Component Pipeline Tests
Each module can be run independently to test its stage in the pipeline:
```bash
# Test ingestion & event normalization
python modules/ingestion.py

# Test action proposals (network agent)
python modules/network_agent.py

# Test risk score calculation
python modules/risk_engine.py

# Test safety gate evaluation
python modules/safety_gate.py

# Test audit database logging
python modules/audit_log.py
```

### Step 4: Run Full Pipeline at Scale (5,000 Event Batch)
Executes all 6 stages sequentially across a stratified random sample of 5,000 events, writes decision trails to `logs/audit.db`, and outputs a full cross-tabulation report:
```bash
python scripts/run_full_pipeline.py
```

### Step 5: Launch the Operational Dashboard
Launch the Streamlit web dashboard to monitor pipeline metrics and review audit trails:
```bash
streamlit run dashboard/app.py
```
Open your browser and navigate to:
```
http://localhost:8501
```

---

## Repository Structure

```text
Cyber-Defence-Pipeline/
├── config.py                 # Pipeline configuration constants
├── requirements.txt          # Python dependencies
├── sitecustomize.py          # Platform hook for Windows Python 3.12 compatibility
├── .gitignore                # Git exclusions (large CSVs, local DBs, venvs)
├── README.md                 # Project documentation
├── data/
│   └── raw/                  # Destination for raw CICIDS2017 CSV files
├── models/
│   └── threat_detector.pkl   # Serialized trained RandomForest model
├── modules/                  # Core architectural pipeline modules
│   ├── dataset_loader.py     # CSV loader and column sanitizer
│   ├── ingestion.py          # Cleaning (NaN/Inf) & event normalization
│   ├── detection.py          # ML feature matrix, training, evaluation, & inference
│   ├── network_agent.py      # Rule-based candidate action proposal
│   ├── risk_engine.py        # Quantitative risk scoring & categorization
│   ├── safety_gate.py        # Inverted safety gate policy enforcement
│   └── audit_log.py          # SQLite audit persistence & querying
├── dashboard/
│   └── app.py                # Streamlit operational dashboard
├── scripts/
│   ├── inspect_dataset.py    # Read-only dataset exploration script
│   └── run_full_pipeline.py  # Stratified batch pipeline runner (5k events)
├── tests/
│   └── sample_events.py      # Hand-crafted event fixtures
└── logs/
    └── audit.db              # SQLite decision database (generated upon run)
```

---

## License

This project is licensed under the MIT License.
