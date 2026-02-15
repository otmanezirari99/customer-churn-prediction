"""
Script d'évaluation et visualisation des résultats
Auteur: Otmane ZIRARI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
import mlflow
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from preprocessing import preprocess_data

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Affiche la matrice de confusion"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'Matrice de Confusion - {model_name}')
    plt.ylabel('Vraie valeur')
    plt.xlabel('Prédiction')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, model_name, save_path=None):
    """Affiche la courbe ROC"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title(f'Courbe ROC - {model_name}')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curve(y_true, y_pred_proba, model_name, save_path=None):
    """Affiche la courbe Precision-Recall"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Courbe Precision-Recall - {model_name}')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, feature_names, model_name, save_path=None, top_n=15):
    """Affiche l'importance des features (pour les modèles qui le supportent)"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print(f"Le modèle {model_name} ne supporte pas l'importance des features")
            return
        
        # Créer un DataFrame pour faciliter le tri
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_imp_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Features les plus importantes - {model_name}')
        plt.xlabel('Importance')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Erreur lors de l'affichage de l'importance des features: {e}")


def evaluate_model_detailed(model, X_test, y_test, model_name, feature_names=None, reports_dir=None):
    """Évaluation détaillée d'un modèle"""
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Créer le dossier de sauvegarde
    if reports_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Matrice de confusion
    plot_confusion_matrix(y_test, y_pred, model_name, 
                         os.path.join(reports_dir, f"confusion_matrix_{model_name}.png"))
    
    # Courbe ROC
    plot_roc_curve(y_test, y_pred_proba, model_name,
                   os.path.join(reports_dir, f"roc_curve_{model_name}.png"))
    
    # Courbe Precision-Recall
    plot_precision_recall_curve(y_test, y_pred_proba, model_name,
                                os.path.join(reports_dir, f"pr_curve_{model_name}.png"))
    
    # Importance des features
    if feature_names is not None:
        plot_feature_importance(model, feature_names, model_name,
                               os.path.join(reports_dir, f"feature_importance_{model_name}.png"))
    
    # Rapport de classification
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(reports_dir, f"classification_report_{model_name}.csv"))
    
    print(f"\n=== Rapport de classification - {model_name} ===")
    print(classification_report(y_test, y_pred))
    
    return {
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'classification_report': report
    }


def compare_models_from_mlflow(reports_dir=None):
    """Compare tous les modèles entraînés via MLflow"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    mlruns_path = os.path.join(project_root, "mlruns")
    mlflow.set_tracking_uri(f"file:{mlruns_path}")
    
    # Charger les runs de l'expérience
    experiment = mlflow.get_experiment_by_name("Telco_Customer_Churn_Prediction")
    
    if experiment is None:
        print("Aucune expérience trouvée. Veuillez d'abord entraîner les modèles.")
        return
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Trier par F1-score
    runs_sorted = runs.sort_values('metrics.f1_score', ascending=False)
    
    print("\n=== Comparaison des modèles ===")
    print(runs_sorted[['run_name', 'metrics.accuracy', 'metrics.precision', 
                       'metrics.recall', 'metrics.f1_score', 'metrics.roc_auc']].to_string())
    
    # Sauvegarder la comparaison
    if reports_dir is None:
        reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    comparison_df = runs_sorted[['run_name', 'metrics.accuracy', 'metrics.precision', 
                                 'metrics.recall', 'metrics.f1_score', 'metrics.roc_auc']]
    comparison_df.to_csv(os.path.join(reports_dir, "model_comparison.csv"), index=False)
    
    return comparison_df


def main():
    """Fonction principale d'évaluation"""
    # Déterminer les chemins
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Charger le meilleur modèle sauvegardé
    model_path = os.path.join(project_root, "models", "best_model.pkl")
    
    if not os.path.exists(model_path):
        print("Aucun modèle trouvé. Veuillez d'abord entraîner les modèles avec train.py")
        return
    
    # Charger les données
    file_path = os.path.join(project_root, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test, scaler, label_encoders = preprocess_data(file_path)
    
    # Charger le modèle
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Obtenir les noms des features
    feature_names = [col for col in pd.read_csv(file_path).columns 
                     if col not in ['customerID', 'Churn']]
    
    # Évaluation détaillée
    reports_dir = os.path.join(project_root, "reports")
    print("Évaluation du meilleur modèle...")
    evaluate_model_detailed(model, X_test, y_test, "best_model", feature_names, reports_dir)
    
    # Comparaison de tous les modèles
    print("\nComparaison de tous les modèles...")
    compare_models_from_mlflow(reports_dir)
    
    print("\n=== Évaluation terminée ===")
    print(f"Les rapports ont été sauvegardés dans le dossier {reports_dir}/")


if __name__ == "__main__":
    main()

