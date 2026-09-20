import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]

class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "customerID" in X.columns:
            X = X.drop(columns=["customerID"])

        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

        X["NumServices"] = sum(
            (X[col] == "Yes").astype(int) for col in SERVICE_COLUMNS
        )

        X["TenureGroup"] = pd.cut(
            X["tenure"],
            bins=[-1, 12, 24, 48, 60, np.inf],
            labels=["0-12", "13-24", "25-48", "49-60", "61+"]
        )

        X["SeniorCitizen"] = X["SeniorCitizen"].astype(str)
        return X
