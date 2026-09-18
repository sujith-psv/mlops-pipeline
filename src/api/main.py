from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import pandas as pd
import joblib

app = FastAPI(
    title="Telco Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# Load trained model
pipeline = joblib.load("models/churn_pipeline.pkl")

# Optimized during model training
CHURN_THRESHOLD = 0.63


@app.get("/")
def home():
    return {
        "message": "Telco Churn Prediction API is working!"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.post("/predict")
def predict(data: CustomerData):

    try:
        # Convert request to dictionary
        input_data = data.model_dump()

        # Convert dictionary to DataFrame
        df = pd.DataFrame([input_data])

        # Get churn probability
        probability = pipeline.predict_proba(df)[0][1]

        # Apply optimized threshold
        prediction = int(probability >= CHURN_THRESHOLD)

        return {
            "prediction": prediction,
            "probability": float(probability),
            "threshold": CHURN_THRESHOLD
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )