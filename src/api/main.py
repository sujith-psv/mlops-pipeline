from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import joblib

app = FastAPI()
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

@app.get("/")
def home():
    return {"message": "FastAPI is working!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: CustomerData):

    try:
        # Convert request to dictionary
        input_data = data.model_dump()

        # Convert dictionary to DataFrame
        df = pd.DataFrame([input_data])

        # Make prediction
        prediction = pipeline.predict(df)[0]

        # Get prediction probability
        probability = pipeline.predict_proba(df)[0][1]

        return {
            "prediction": int(prediction),
            "probability": float(probability)
        }

    except Exception as e:
        return {
            "error": str(e)
        }
#uvicorn src.api.main:app --reload