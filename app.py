import os
import json
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


def _project_root():
    return os.path.dirname(os.path.abspath(__file__))


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_feature_names(project_root):
    data_path = os.path.join(project_root, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if os.path.exists(data_path):
        cols = list(pd.read_csv(data_path, nrows=1).columns)
        return [c for c in cols if c not in ["customerID", "Churn"]]

    feature_names_path = os.path.join(project_root, "outputs", "preprocessed", "feature_names.json")
    if os.path.exists(feature_names_path):
        with open(feature_names_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


def _transform_input(payload, feature_names, scaler, label_encoders):
    if isinstance(payload, dict) and "features" in payload and isinstance(payload["features"], dict):
        features = payload["features"]
    else:
        features = payload

    if not isinstance(features, dict):
        raise ValueError("Payload must be a JSON object (dict) or contain a 'features' object.")

    # Build dataframe with required columns
    if feature_names is None:
        feature_names = sorted(list(features.keys()))

    row = {k: features.get(k, None) for k in feature_names}
    df = pd.DataFrame([row])

    # Handle TotalCharges conversion similar to preprocessing
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Map binary Yes/No columns
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(df[col])

    # Encode categorical columns using fitted LabelEncoders
    if isinstance(label_encoders, dict):
        for col, le in label_encoders.items():
            if col in df.columns:
                # Unknown categories -> -1
                classes = set(getattr(le, "classes_", []))
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda x: x if x in classes else None)
                df[col] = df[col].fillna("__UNKNOWN__")

                # If unknown token not in encoder, encode unknown as -1
                if "__UNKNOWN__" not in classes:
                    df[col] = df[col].apply(lambda x: -1 if x == "__UNKNOWN__" else int(le.transform([x])[0]))
                else:
                    df[col] = le.transform(df[col])

    # Ensure numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    X = df.values
    if scaler is not None:
        X = scaler.transform(X)

    return X, df


def _form_schema():
    return {
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 1],
        "Partner": ["No", "Yes"],
        "Dependents": ["No", "Yes"],
        "tenure": None,
        "PhoneService": ["No", "Yes"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["No", "Yes", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "Yes", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["No", "Yes"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        "MonthlyCharges": None,
        "TotalCharges": None,
    }


def _default_form_values():
    return {
        "gender": "Male",
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
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 845.5,
    }


PROJECT_ROOT = _project_root()
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "best_model.pkl"))
SCALER_PATH = os.environ.get("SCALER_PATH", os.path.join(PROJECT_ROOT, "outputs", "preprocessed", "scaler.pkl"))
ENCODERS_PATH = os.environ.get("ENCODERS_PATH", os.path.join(PROJECT_ROOT, "outputs", "preprocessed", "label_encoders.pkl"))


app = Flask(__name__)


model = _load_pickle(MODEL_PATH)
scaler = _load_pickle(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
label_encoders = _load_pickle(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else {}
feature_names = _get_feature_names(PROJECT_ROOT)


@app.get("/")
def home():
    return render_template(
        "index.html",
        schema=_form_schema(),
        values=_default_form_values(),
        result=None,
        error=None,
    )


@app.post("/predict-form")
def predict_form():
    schema = _form_schema()
    values = {}
    for key in schema.keys():
        values[key] = request.form.get(key, "")

    # Cast numeric
    for num_key in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if num_key in values:
            try:
                if num_key in ["SeniorCitizen", "tenure"]:
                    values[num_key] = int(values[num_key])
                else:
                    values[num_key] = float(values[num_key])
            except Exception:
                return render_template(
                    "index.html",
                    schema=schema,
                    values=values,
                    result=None,
                    error=f"Invalid value for {num_key}",
                )

    try:
        X, df = _transform_input(values, feature_names, scaler, label_encoders)
        pred = int(model.predict(X)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])

        result = {
            "prediction": pred,
            "prediction_label": "Churn" if pred == 1 else "No Churn",
            "probability_churn": proba,
        }
        return render_template(
            "index.html",
            schema=schema,
            values=values,
            result=result,
            error=None,
        )
    except Exception as e:
        return render_template(
            "index.html",
            schema=schema,
            values=values,
            result=None,
            error=str(e),
        )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Expected JSON body"}), 400

    try:
        X, df = _transform_input(payload, feature_names, scaler, label_encoders)
        pred = int(model.predict(X)[0])

        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])

        return jsonify({
            "prediction": pred,
            "prediction_label": "Churn" if pred == 1 else "No Churn",
            "probability_churn": proba,
            "used_features": list(df.columns),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
