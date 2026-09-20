from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from src.feature_engineering import TelcoFeatureEngineer

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "TelcoCustomerChurn.csv")
X = df.drop(columns=["Churn"])
y = (df["Churn"] == "Yes").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "NumServices"]
categorical_features = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "TenureGroup"
]

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

pipeline = Pipeline([
    ("features", TelcoFeatureEngineer()),
    ("preprocess", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42))
])

grid = GridSearchCV(
    pipeline,
    {
        "model__max_depth":[4,5,6],
        "model__min_samples_leaf":[10,20],
        "model__criterion":["gini","entropy"],
        "model__class_weight":[None,"balanced"]
    },
    scoring="f1", cv=4, n_jobs=-1
)

grid.fit(X_train, y_train)
model = grid.best_estimator_
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:,1]

results = {
    "accuracy": accuracy_score(y_test,pred),
    "precision": precision_score(y_test,pred),
    "recall": recall_score(y_test,pred),
    "f1": f1_score(y_test,pred),
    "roc_auc": roc_auc_score(y_test,prob),
    "confusion_matrix": confusion_matrix(y_test,pred).tolist(),
    "best_params": grid.best_params_
}
joblib.dump(model, ROOT / "model" / "churn_model.pkl")
(ROOT / "model" / "metrics.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
