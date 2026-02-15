"""
Script de prétraitement des données pour le projet de prédiction du churn
Auteur: Otmane ZIRARI
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.filterwarnings('ignore')


def load_data(file_path):
    """Charge le dataset depuis le fichier CSV"""
    df = pd.read_csv(file_path)
    return df


def handle_missing_values(df):
    """Gère les valeurs manquantes dans le dataset"""
    # TotalCharges peut avoir des valeurs vides (strings) pour les nouveaux clients
    # Convertir en NaN puis remplacer par 0 ou la médiane
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Remplacer les NaN par 0 (clients nouveaux sans historique)
    df['TotalCharges'].fillna(0, inplace=True)
    
    return df


def encode_categorical_variables(df):
    """Encode les variables catégorielles"""
    df_encoded = df.copy()
    
    # Variables binaires Yes/No
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
    
    # Variables avec plusieurs catégories - utiliser Label Encoding
    categorical_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'Contract', 'PaymentMethod']
    
    label_encoders = {}
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
    
    return df_encoded, label_encoders


def prepare_features_target(df):
    """Sépare les features et la variable cible"""
    # Exclure customerID et Churn des features
    feature_cols = [col for col in df.columns if col not in ['customerID', 'Churn']]
    X = df[feature_cols]
    y = df['Churn']
    
    return X, y


def normalize_features(X_train, X_test):
    """Normalise les features numériques"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(file_path, test_size=0.2, random_state=42, normalize=True):
    """
    Pipeline complet de prétraitement
    
    Parameters:
    -----------
    file_path : str
        Chemin vers le fichier CSV
    test_size : float
        Proportion du dataset pour le test set
    random_state : int
        Seed pour la reproductibilité
    normalize : bool
        Si True, normalise les features
        
    Returns:
    --------
    X_train, X_test, y_train, y_test, scaler, label_encoders
    """
    # Charger les données
    df = load_data(file_path)
    
    # Gérer les valeurs manquantes
    df = handle_missing_values(df)
    
    # Encoder les variables catégorielles
    df_encoded, label_encoders = encode_categorical_variables(df)
    
    # Séparer features et target
    X, y = prepare_features_target(df_encoded)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = None
    if normalize:
        # Normaliser les features
        X_train, X_test, scaler = normalize_features(X_train, X_test)
    
    return X_train, X_test, y_train, y_test, scaler, label_encoders


if __name__ == "__main__":
    # Test du preprocessing
    # Déterminer le chemin relatif au projet
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    file_path = os.path.join(project_root, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    X_train, X_test, y_train, y_test, scaler, label_encoders = preprocess_data(file_path)
    
    print("Preprocessing terminé avec succès!")
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape X_test: {X_test.shape}")
    print(f"Shape y_train: {y_train.shape}")
    print(f"Shape y_test: {y_test.shape}")
    print(f"\nDistribution de Churn dans le train set:")
    print(y_train.value_counts())
    print(f"\nDistribution de Churn dans le test set:")
    print(y_test.value_counts())

