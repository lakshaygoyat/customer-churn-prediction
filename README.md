# Customer Churn Prediction

End-to-end IBM Telco Customer Churn solution using scikit-learn and FastAPI.

## Public Git Repository
`https://github.com/lakshaygoyat/customer-churn-prediction`

Replace the placeholder before submission.

## Workflow
Business Problem → Data → Preparation → EDA → Feature Engineering → Model → Evaluation → Interpretation → Saved Model → API

## Data Summary
- 7,043 customers
- 21 original columns
- 1,869 churners (26.54%)
- 11 blank `TotalCharges` values
- 0 duplicate rows
- 0 duplicate customer IDs

## Final Model
Required model: Decision Tree Classifier.

Best tuning parameters:
```json
{
  "model__class_weight": "balanced",
  "model__criterion": "gini",
  "model__max_depth": 4,
  "model__min_samples_leaf": 20
}
```

Test metrics:
- Accuracy: 0.7487
- Precision: 0.5183
- Recall: 0.7576
- F1: 0.6155
- ROC-AUC: 0.8274
- Confusion matrix: [[1157, 395], [136, 425]]

## Project Structure
```text
customer_churn_submission/
├── data/
├── notebook/churn_analysis.ipynb
├── model/churn_model.pkl
├── model/metrics.json
├── model/feature_importance.csv
├── src/feature_engineering.py
├── app.py
├── train.py
├── requirements.txt
├── sample_request.json
├── sample_response.json
└── README.md
```

## Local Development

### Prerequisites
Install Python 3.10+, VS Code, VS Code Python extension, and Git.

### Create environment
Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows CMD:
```cmd
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code: `Ctrl+Shift+P` → `Python: Select Interpreter` → select `.venv`.

### Run notebook
```bash
jupyter notebook
```
Open `notebook/churn_analysis.ipynb` and use **Run All**.

### Rebuild model
```bash
python train.py
```

### Run API in development mode
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Swagger:
`http://127.0.0.1:8000/docs`

### Test prediction
Use Swagger `/predict` → **Try it out**, paste the contents of `sample_request.json`.

Or Windows curl:
```cmd
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" --data-binary "@sample_request.json"
```

Example output:
```json
{
  "prediction": "Yes",
  "churn_probability": 0.8335
}
```

## Leakage Prevention
Preprocessing is fitted only using training folds because imputation, one-hot encoding, feature engineering and the classifier are wrapped in a scikit-learn pipeline. The same saved pipeline is reused by FastAPI. `OneHotEncoder(handle_unknown="ignore")` makes inference safer for unseen category combinations.

## Feature Engineering
1. `NumServices`: number of active services. It represents customer-product engagement.
2. `TenureGroup`: lifecycle bucket created from tenure. It captures non-linear churn behaviour across customer stages.

## Business Metric Choice
For proactive telecom retention, recall is generally prioritized because a false negative means a real churner is missed and receives no intervention. Precision is still monitored to control unnecessary retention contacts.

## Bonus Work
- GridSearchCV hyperparameter tuning
- class imbalance handling using `class_weight`
- ROC-AUC
- stratified train/test split
- reproducible `train.py`
- API input validation
- feature importance export
- model/preprocessing saved as a single reusable pipeline
