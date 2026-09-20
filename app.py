from pathlib import Path
from typing import Literal, Optional, Union
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.feature_engineering import TelcoFeatureEngineer  # needed by joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "churn_model.pkl"

if not MODEL_PATH.exists():
    raise RuntimeError("Model not found. Run: python train.py")

model = joblib.load(MODEL_PATH)
app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

class CustomerInput(BaseModel):
    customerID: Optional[str] = None
    gender: Literal["Male","Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes","No"]
    Dependents: Literal["Yes","No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes","No"]
    MultipleLines: Literal["Yes","No","No phone service"]
    InternetService: Literal["DSL","Fiber optic","No"]
    OnlineSecurity: Literal["Yes","No","No internet service"]
    OnlineBackup: Literal["Yes","No","No internet service"]
    DeviceProtection: Literal["Yes","No","No internet service"]
    TechSupport: Literal["Yes","No","No internet service"]
    StreamingTV: Literal["Yes","No","No internet service"]
    StreamingMovies: Literal["Yes","No","No internet service"]
    Contract: Literal["Month-to-month","One year","Two year"]
    PaperlessBilling: Literal["Yes","No"]
    PaymentMethod: Literal[
        "Electronic check","Mailed check",
        "Bank transfer (automatic)","Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Optional[Union[float,str]] = None

@app.get("/")
def health():
    return {"status":"ok","message":"Customer Churn Prediction API"}

@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        row = pd.DataFrame([customer.model_dump()])
        label = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0,1])
        return {
            "prediction": "Yes" if label == 1 else "No",
            "churn_probability": round(probability, 4)
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
