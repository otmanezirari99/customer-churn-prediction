"""
Script principal d'entraînement des modèles avec MLflow
Auteur: Otmane ZIRARI
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import json
import argparse
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import warnings
import os
warnings.filterwarnings('ignore')

from preprocessing import preprocess_data


def _json_safe_metrics(metrics):
    """Convert metrics dict to JSON-serializable values."""
    safe = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            safe[k] = v.item()
        else:
            safe[k] = float(v) if isinstance(v, (int, float)) else v
    return safe


def save_run_artifacts(run_dir, model, metrics, y_true, y_pred, y_pred_proba=None):
    """Persist model + evaluation artifacts for a single run."""
    os.makedirs(run_dir, exist_ok=True)

    # Metrics
    safe_metrics = _json_safe_metrics(metrics)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(safe_metrics, f, indent=2)
    pd.DataFrame([safe_metrics]).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        os.path.join(run_dir, "confusion_matrix.csv")
    )

    # Classification report
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(run_dir, "classification_report.csv")
    )

    # Save model
    import pickle
    with open(os.path.join(run_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """Calcule les métriques d'évaluation"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def train_logistic_regression(X_train, X_test, y_train, y_test, use_smote=False):
    """Entraîne un modèle de régression logistique"""
    print("\n=== Entraînement Logistic Regression ===")
    
    # Appliquer SMOTE si demandé
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Modèle avec pondération des classes
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X_train_resampled, y_train_resampled)
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métriques
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    
    return model, metrics, y_pred, y_pred_proba


def train_decision_tree(X_train, X_test, y_train, y_test, use_smote=False):
    """Entraîne un arbre de décision"""
    print("\n=== Entraînement Decision Tree ===")
    
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    model = DecisionTreeClassifier(
        class_weight='balanced',
        max_depth=10,
        min_samples_split=20,
        random_state=42
    )
    
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    
    return model, metrics, y_pred, y_pred_proba


def train_random_forest(X_train, X_test, y_train, y_test, use_smote=False):
    """Entraîne un Random Forest"""
    print("\n=== Entraînement Random Forest ===")
    
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    
    return model, metrics, y_pred, y_pred_proba


def train_xgboost(X_train, X_test, y_train, y_test, use_smote=False):
    """Entraîne un modèle XGBoost"""
    print("\n=== Entraînement XGBoost ===")
    
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    
    return model, metrics, y_pred, y_pred_proba


def train_lightgbm(X_train, X_test, y_train, y_test, use_smote=False):
    """Entraîne un modèle LightGBM"""
    print("\n=== Entraînement LightGBM ===")
    
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    
    return model, metrics, y_pred, y_pred_proba


def train_svm(X_train, X_test, y_train, y_test, use_smote=False):
    """Entraîne un modèle SVM"""
    print("\n=== Entraînement SVM ===")
    
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # SVM peut être lent sur de gros datasets, on utilise un échantillon si nécessaire
    if len(X_train_resampled) > 5000:
        sample_idx = np.random.choice(len(X_train_resampled), 5000, replace=False)
        X_train_resampled = X_train_resampled[sample_idx]
        y_train_resampled = y_train_resampled.iloc[sample_idx] if isinstance(y_train_resampled, pd.Series) else y_train_resampled[sample_idx]
    
    model = SVC(
        kernel='rbf',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    
    return model, metrics, y_pred, y_pred_proba


def main():
    """Fonction principale pour entraîner tous les modèles"""
    parser = argparse.ArgumentParser(description="Train churn models (step-by-step).")
    parser.add_argument("--only-preprocess", action="store_true", help="Run preprocessing only")
    parser.add_argument("--save-preprocessed", action="store_true", help="Save preprocessed splits to disk")
    parser.add_argument("--skip-mlflow", action="store_true", help="Disable MLflow logging")
    parser.add_argument("--outputs-dir", default=None, help="Base directory to save per-run artifacts")
    args = parser.parse_args()

    # Déterminer le chemin relatif au projet
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Dossier outputs (artifacts sur disque)
    if args.outputs_dir is None:
        outputs_dir = os.path.join(project_root, "outputs")
    else:
        outputs_dir = args.outputs_dir
    os.makedirs(outputs_dir, exist_ok=True)

    # Configuration MLflow
    if not args.skip_mlflow:
        mlruns_path = os.path.join(project_root, "mlruns")
        mlflow.set_tracking_uri(f"file:{mlruns_path}")
        mlflow.set_experiment("Telco_Customer_Churn_Prediction")
    
    # Charger et prétraiter les données
    file_path = os.path.join(project_root, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("Chargement et prétraitement des données...")
    X_train, X_test, y_train, y_test, scaler, label_encoders = preprocess_data(file_path)

    if args.save_preprocessed:
        import pickle
        pre_dir = os.path.join(outputs_dir, "preprocessed")
        os.makedirs(pre_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(pre_dir, "splits.npz"),
            X_train=X_train,
            X_test=X_test,
            y_train=np.asarray(y_train),
            y_test=np.asarray(y_test),
        )
        with open(os.path.join(pre_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(pre_dir, "label_encoders.pkl"), "wb") as f:
            pickle.dump(label_encoders, f)
        print(f"Preprocessed data saved to {pre_dir}")

    if args.only_preprocess:
        print("Preprocessing only requested. Exiting.")
        return
    
    # Dictionnaire des modèles à entraîner
    models = {
        'Logistic_Regression': train_logistic_regression,
        'Decision_Tree': train_decision_tree,
        'Random_Forest': train_random_forest,
        'XGBoost': train_xgboost,
        'LightGBM': train_lightgbm,
        'SVM': train_svm
    }
    
    # Options pour gérer le déséquilibre
    imbalance_options = [False, True]  # False = class_weight, True = SMOTE
    
    best_model = None
    best_score = 0
    best_model_name = None

    run_summaries = []
    
    # Entraîner chaque modèle avec différentes stratégies
    for model_name, train_func in models.items():
        for use_smote in imbalance_options:
            strategy = "SMOTE" if use_smote else "Class_Weight"
            run_name = f"{model_name}_{strategy}"

            # Entraîner le modèle
            model, metrics, y_pred, y_pred_proba = train_func(X_train, X_test, y_train, y_test, use_smote)

            # Sauvegarder artifacts sur disque (un dossier par run)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(outputs_dir, "runs", run_name, ts)
            save_run_artifacts(run_dir, model, metrics, y_test, y_pred, y_pred_proba)

            # MLflow (optionnel)
            if not args.skip_mlflow:
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_params({
                        'model': model_name,
                        'strategy': strategy,
                        'use_smote': use_smote
                    })
                    mlflow.log_metrics(metrics)
                    if model_name == 'XGBoost':
                        mlflow.xgboost.log_model(model, "model")
                    elif model_name == 'LightGBM':
                        mlflow.lightgbm.log_model(model, "model")
                    else:
                        mlflow.sklearn.log_model(model, "model")

            # Afficher les résultats
            print(f"\n{run_name} - Résultats:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  Saved to: {run_dir}")

            run_summaries.append({
                "run_name": run_name,
                "saved_to": run_dir,
                **_json_safe_metrics(metrics)
            })

            # Garder le meilleur modèle (basé sur F1-score)
            if metrics['f1_score'] > best_score:
                best_score = metrics['f1_score']
                best_model = model
                best_model_name = run_name
    
    print(f"\n\n=== Meilleur modèle: {best_model_name} ===")
    print(f"F1-Score: {best_score:.4f}")

    if len(run_summaries) > 0:
        summary_path = os.path.join(outputs_dir, "runs", "runs_summary.csv")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        pd.DataFrame(run_summaries).to_csv(summary_path, index=False)
        print(f"\nRésumé des runs sauvegardé dans {summary_path}")
    
    # Sauvegarder le meilleur modèle
    import pickle
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nModèle sauvegardé dans {model_path}")


if __name__ == "__main__":
    main()

