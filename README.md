# End-to-End MLOps Pipeline for Telecom Churn Prediction

## Live Deployment

* Live API: [https://mlops-pipeline-5pxk.onrender.com/](https://mlops-pipeline-5pxk.onrender.com/)
* Swagger Docs: [https://mlops-pipeline-5pxk.onrender.com/docs](https://mlops-pipeline-5pxk.onrender.com/docs)
* GitHub Repository: [https://github.com/sujith-psv/mlops-pipeline](https://github.com/sujith-psv/mlops-pipeline)

---

# Project Overview

This project is a complete end-to-end MLOps pipeline for telecom customer churn prediction.

The system covers the full machine learning lifecycle:

* data validation
* preprocessing
* model training
* experiment tracking
* drift monitoring
* workflow orchestration
* API serving
* containerization
* cloud deployment

The project uses production-style MLOps tooling to simulate a real-world ML engineering workflow.

---

# Features

## Machine Learning Pipeline

* Telecom churn prediction using Scikit-learn
* Automated preprocessing pipeline
* Feature encoding and scaling
* Model serialization with Joblib

## Data Validation

* Great Expectations integration
* Schema validation
* Null value checks
* Data quality monitoring

## Experiment Tracking

* MLflow integration
* Accuracy and F1-score tracking
* Model artifact logging
* Experiment comparison

## Monitoring

* Evidently AI drift detection
* Data drift reports
* HTML monitoring dashboard generation

## Workflow Orchestration

* Prefect integration
* Automated ML workflow execution
* Pipeline task management

## API Deployment

* FastAPI inference API
* Interactive Swagger documentation
* JSON-based prediction endpoint

## Containerization

* Dockerized application
* Reproducible environment
* Cloud-ready deployment

## Cloud Hosting

* Public deployment using Render
* Live production-style inference API

---

# Tech Stack

| Category               | Technologies        |
| ---------------------- | ------------------- |
| Programming Language   | Python              |
| Machine Learning       | Scikit-learn        |
| Data Processing        | Pandas, NumPy       |
| API Framework          | FastAPI             |
| Experiment Tracking    | MLflow              |
| Validation             | Great Expectations  |
| Monitoring             | Evidently AI        |
| Workflow Orchestration | Prefect             |
| Containerization       | Docker              |
| Cloud Deployment       | Render              |
| Visualization          | Matplotlib, Seaborn |

---

# Project Architecture

```text
Raw Data
    ↓
Great Expectations Validation
    ↓
Data Preprocessing
    ↓
Model Training
    ↓
MLflow Experiment Tracking
    ↓
Model Serialization
    ↓
Evidently Drift Monitoring
    ↓
Prefect Workflow Orchestration
    ↓
FastAPI Inference API
    ↓
Docker Containerization
    ↓
Cloud Deployment on Render
```

---

# Folder Structure

```text
mlops-pipeline/
│
├── data/
│   └── raw/
│       └── telco.csv
│
├── models/
│   └── churn_pipeline.pkl
│
├── reports/
│   └── drift_report.html
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── training/
│   │   └── train.py
│   │
│   ├── validation/
│   │   └── validate_data.py
│   │
│   ├── monitoring/
│   │   └── drift_detection.py
│   │
│   └── orchestration/
│       └── prefect_flow.py
│
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Model Performance

| Metric   | Score |
| -------- | ----- |
| Accuracy | ~78%  |
| F1 Score | ~53%  |

---

# API Usage

## Base URL

```text
https://mlops-pipeline-5pxk.onrender.com
```

## Swagger Documentation

```text
https://mlops-pipeline-5pxk.onrender.com/docs
```

---

# Prediction Endpoint

## POST `/predict`

### Sample Request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.85,
  "TotalCharges": 1080.5
}
```

### Sample Response

```json
{
  "prediction": "Yes"
}
```

---

# Running Locally

## Clone Repository

```bash
git clone https://github.com/sujith-psv/mlops-pipeline.git
cd mlops-pipeline
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Training Pipeline

```bash
python src/training/train.py
```

---

## Run Data Validation

```bash
python src/validation/validate_data.py
```

---

## Run Drift Detection

```bash
python src/monitoring/drift_detection.py
```

---

## Run FastAPI Server

```bash
uvicorn src.api.main:app --reload
```

---

# Docker Usage

## Build Docker Image

```bash
docker build -t churn-mlops .
```

---

## Run Docker Container

```bash
docker run -p 8000:8000 churn-mlops
```

---

## Open API

```text
http://localhost:8000/docs
```

---

# Monitoring

The project includes:

* drift detection reports
* data quality validation
* experiment tracking
* orchestration monitoring

Generated monitoring reports are stored in:

```text
reports/
```

---

# Future Improvements

Potential future upgrades:

* CI/CD with GitHub Actions
* Kubernetes deployment
* Streamlit frontend
* Authentication system
* Automated retraining
* Database integration
* Model registry integration
* Real-time monitoring dashboards

---

# Screenshots To Add

Recommended screenshots for better presentation:

* Swagger UI
* MLflow UI
* Evidently Drift Report
* Prefect Workflow Dashboard
* Docker Container Running
* Render Deployment Dashboard

---

# Key Learnings

This project demonstrates:

* production-style ML deployment
* model serving architecture
* workflow orchestration
* cloud deployment workflows
* MLOps monitoring practices
* Docker-based infrastructure
* API engineering
* experiment tracking systems

---

# Author

Sujith

GitHub: [https://github.com/sujith-psv](https://github.com/sujith-psv)

---

# License

This project is intended for educational and portfolio purposes.
