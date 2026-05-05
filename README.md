# 🔧 CNC Fault Detector

Real-time CNC machine fault detection using a PyTorch MLP trained on sensor data — predicting tool failure before it happens, deployed as a live FastAPI service on AWS EC2.

[![Live API](https://img.shields.io/badge/Live%20API-AWS%20EC2-orange)](http://43.205.255.123:8000/)
[![API Docs](https://img.shields.io/badge/API%20Docs-Swagger-green)](http://43.205.255.123:8000/docs)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML-red)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Containerised-blue)](https://docker.com)

---

## 📊 Results

| Metric | Value |
|---|---|
| Recall (Failure Class) | 88% |
| False Alarm Reduction | ~40% vs baseline |
| Class Imbalance | 96.7% healthy — handled via pos_weight=15.0 |
| Dataset | UCI AI4I 2020 — 10,000 rows |
| Training Stopped | Epoch 75 of 200 (early stopping) |

---

## 🧠 What It Does

CNC machines fail silently — a worn tool or thermal anomaly produces a defective part before any operator notices. This system monitors five sensor readings in real time and predicts whether a machining operation is heading toward failure, giving operators time to intervene before damage occurs.

This is a core problem in **smart manufacturing / Industry 4.0** — models like this are consumed by factory dashboards and SCADA systems via REST to enable predictive maintenance.

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch MLP with BatchNorm + Dropout |
| Experiment Tracking | MLflow with SQLite backend |
| API | FastAPI + Uvicorn |
| Containerisation | Docker |
| Registry | AWS ECR |
| Hosting | AWS EC2 (t2.micro) |
| Dependency Management | uv |

---

## 🏗️ Model Architecture

```
Input (5 features)
→ Linear(5 → 64) + BatchNorm + ReLU + Dropout(0.3)
→ Linear(64 → 32) + BatchNorm + ReLU + Dropout(0.3)
→ Linear(32 → 16) + BatchNorm + ReLU + Dropout(0.3)
→ Linear(16 → 1) → Sigmoid → probability
```

**Key training decisions:**
- `BCEWithLogitsLoss` with `pos_weight=15.0` — makes missing a failure 15x more costly than a false alarm
- `ReduceLROnPlateau` — halves LR when validation loss stalls
- Early stopping `patience=15` — stopped at epoch 75, preserving best checkpoint
- Stratified train/val/test split — maintains 3.3% failure ratio across all sets

---

## 🌐 API Endpoints

### `POST /predict`

**Request:**
```json
{
  "air_temp": 301.0,
  "process_temp": 313.0,
  "rpm": 1200,
  "torque": 65.0,
  "tool_wear": 250
}
```

**Response:**
```json
{
  "prediction": "FAILURE",
  "probability": 0.9852,
  "status_code": 1
}
```

### `GET /health`
```json
{
  "status": "healthy",
  "model": "CNCFaultDetector",
  "version": "1.0.0"
}
```

---

## 📁 Project Structure

```
cnc-fault-detector/
├── app.py              ← FastAPI inference server
├── Dockerfile
├── pyproject.toml
├── model.pt
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluation.py
│   └── predict.py
└── data/
    └── scaler.pkl
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/mathewprasanth/cnc-fault-detector.git
cd cnc-fault-detector
uv sync
uvicorn app:app --reload
```

## 🐳 Run with Docker

```bash
docker build -t cnc-fault-detector .
docker run -p 8000:8000 cnc-fault-detector
```

## Example curl

```bash
curl -X POST "http://43.205.255.123:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"air_temp": 301.0, "process_temp": 313.0, "rpm": 1200, "torque": 65.0, "tool_wear": 250}'
```

---

## ⚠️ Limitations

- Trained on simulated data (UCI AI4I) — real factory sensor distributions may differ
- Single failure mode binary classification — multi-failure type detection not yet supported

---

## 👤 Author

**Mathew Prasanth, P.E.**
AI/ML Engineer | U.S. Licensed Professional Engineer
[LinkedIn](https://www.linkedin.com/in/mathewprasanth/) · [Live API](http://43.205.255.123:8000/)

*AWS Certified ML Specialty · AWS Cloud Practitioner*
