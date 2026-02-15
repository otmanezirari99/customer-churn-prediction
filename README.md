# Projet Machine Learning - Prédiction du Churn Client

**Auteur:** Otmane ZIRARI  
**Date:** 18 janvier 2026  
**Institution:** La Faculté des Sciences Semlalia de Marrakech  
**Module:** Machine Learning Techniques

## 📋 Description du Projet

Ce projet vise à développer un système de prédiction du churn client dans le secteur des télécommunications. L'objectif est d'identifier les clients susceptibles de quitter l'entreprise et de proposer des recommandations business pour réduire le taux de churn.

## 🎯 Objectifs

- Développer un modèle prédictif performant pour identifier les clients à risque
- Utiliser MLflow pour le tracking et la gestion des expérimentations
- Analyser les facteurs influençant le churn
- Proposer des recommandations business basées sur les résultats

## 📊 Dataset

- **Nom:** Telco Customer Churn Dataset
- **Source:** [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Taille:** 7,043 clients
- **Variables:** 21 features
- **Type:** Classification binaire (Churn: Yes/No)
- **Déséquilibre:** ~27% de churn (classe minoritaire)

### Variables Principales

#### Variables Démographiques
- `customerID`: Identifiant unique
- `gender`: Genre (Male/Female)
- `SeniorCitizen`: Client senior (0/1)
- `Partner`: A un partenaire (Yes/No)
- `Dependents`: A des personnes à charge (Yes/No)

#### Variables de Services
- `PhoneService`: Service téléphonique
- `MultipleLines`: Lignes multiples
- `InternetService`: Type de connexion (DSL/Fiber optic/No)
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`
- `StreamingTV`, `StreamingMovies`

#### Variables Contractuelles
- `tenure`: Ancienneté (en mois)
- `Contract`: Type de contrat (Month-to-month/One year/Two year)
- `PaperlessBilling`: Facturation électronique
- `PaymentMethod`: Méthode de paiement
- `MonthlyCharges`: Frais mensuels
- `TotalCharges`: Frais totaux

#### Variable Cible
- `Churn`: Client a quitté l'entreprise (Yes/No)

## 🏗️ Structure du Projet

```
archivedata/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── templates/
│   └── index.html             # Interface web (Flask) pour l'inférence
├── src/
│   ├── preprocessing.py      # Script de prétraitement des données
│   ├── train.py              # Script d'entraînement avec MLflow
│   └── evaluate.py           # Script d'évaluation et visualisation
├── outputs/
│   ├── preprocessed/          # Splits + scaler + encoders sauvegardés
│   └── runs/                  # Artifacts + métriques sauvegardés par run
├── app.py                     # API + interface web Flask (inférence)
├── models/
│   └── best_model.pkl        # Meilleur modèle sauvegardé
├── notebooks/
│   └── (notebooks Jupyter pour l'exploration)
├── reports/
│   └── (rapports et visualisations)
├── mlruns/
│   └── (expérimentations MLflow)
├── requirements.txt
└── README.md
```

## 🚀 Installation

1. **Cloner le repository** (si applicable)

2. **Installer les dépendances:**
```bash
pip install -r requirements.txt
```

## 🧠 Explication des fichiers (rôle de chaque composant)

### `src/preprocessing.py`

- Charge le dataset.
- Nettoie et transforme `TotalCharges` en numérique (valeurs vides -> `NaN` -> 0).
- Encode les variables catégorielles.
- Sépare features (`X`) et cible (`y = Churn`).
- Fait le split train/test.
- Optionnel: standardisation via `StandardScaler`.

Sortie (retour de fonction):
- `X_train, X_test, y_train, y_test, scaler, label_encoders`

### `src/train.py`

Script principal d'entraînement multi-modèles.

- Entraîne 6 modèles:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - LightGBM
  - SVM

- Teste 2 stratégies de gestion du déséquilibre:
  - `Class_Weight`
  - `SMOTE`

- Calcule et log (optionnel) dans MLflow:
  - accuracy, precision, recall, f1_score, roc_auc

- **Sauvegarde des artifacts par run sur disque** (même si MLflow est désactivé):
  - `metrics.json`, `metrics.csv`
  - `confusion_matrix.csv`
  - `classification_report.csv`
  - `model.pkl`

- Sauvegarde le meilleur modèle global dans `models/best_model.pkl`.

### `src/evaluate.py`

- Recharge `models/best_model.pkl`.
- Génère des visualisations/rapports dans `reports/`:
  - Matrice de confusion (png)
  - Courbe ROC (png)
  - Courbe Precision-Recall (png)
  - Importance des features (si disponible)
  - `classification_report_*.csv`
- Peut aussi comparer les runs via MLflow (`model_comparison.csv`).

### `app.py` + `templates/index.html` (Flask)

Interface web pour utiliser le modèle sur des nouveaux clients.

- Page `/`:
  - Formulaire (dropdowns + champs numériques)
  - Affiche la prédiction et la probabilité.

- Endpoint API `/predict`:
  - `POST` JSON
  - Retour JSON (pratique pour intégration).

Le serveur charge:
- `models/best_model.pkl`
- `outputs/preprocessed/scaler.pkl`
- `outputs/preprocessed/label_encoders.pkl`


## 💻 Utilisation

### 1. Prétraitement des données

Le script `preprocessing.py` gère:
- Le chargement des données
- La gestion des valeurs manquantes
- L'encodage des variables catégorielles
- La normalisation des features
- La séparation train/test

```bash
python src/preprocessing.py
```

Prétraitement étape-par-étape via `train.py`:

```bash
python src/train.py --only-preprocess
```

Pour sauvegarder les splits + objets de preprocessing (utile pour l'inférence web):

```bash
python src/train.py --only-preprocess --save-preprocessed
```

### 2. Entraînement des modèles

Le script `train.py` entraîne plusieurs modèles avec MLflow:

**Modèles implémentés:**
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- SVM

**Stratégies de gestion du déséquilibre:**
- Pondération des classes (`class_weight='balanced'`)
- SMOTE (Synthetic Minority Over-sampling Technique)

```bash
python src/train.py
```

Pour entraîner **sans MLflow** (mais en gardant la sauvegarde des artifacts sur disque):

```bash
python src/train.py --skip-mlflow
```

Les résultats sont automatiquement trackés dans MLflow. Pour visualiser:
```bash
mlflow ui
```
Puis ouvrir http://localhost:5000 dans votre navigateur.

### 3. Évaluation des modèles

Le script `evaluate.py` génère:
- Matrices de confusion
- Courbes ROC
- Courbes Precision-Recall
- Importance des features
- Rapports de classification détaillés

```bash
python src/evaluate.py
```

## 📦 Sorties (où trouver les résultats)

### Artifacts par run (sur disque)

Après l'entraînement, vous obtenez un dossier par run:

`outputs/runs/<RUN_NAME>/<TIMESTAMP>/`

Exemple:

`outputs/runs/LightGBM_Class_Weight/20260214_003758/`

Contient:
- `metrics.json` et `metrics.csv`
- `confusion_matrix.csv`
- `classification_report.csv`
- `model.pkl`

Un résumé global est aussi généré:
- `outputs/runs/runs_summary.csv`

### Meilleur modèle

- `models/best_model.pkl`

### Rapports/plots

- `reports/`

## 📈 Résultats (exécution du 14/02/2026)

Les résultats ci-dessous proviennent du fichier:

- `outputs/runs/runs_summary.csv`

**Meilleur modèle (selon F1-score):** `LightGBM_Class_Weight`

Métriques:
- Accuracy: **0.7566**
- Precision: **0.5283**
- Recall: **0.7727**
- F1-score: **0.6276**
- ROC-AUC: **0.8363**

> Note: ces chiffres peuvent changer si vous relancez l'entraînement (split / randomness / versions).

## 📈 Algorithmes Utilisés

### 1. Logistic Regression
- Modèle linéaire simple et interprétable
- Baseline pour comparer les autres modèles
- Utile pour comprendre l'impact des variables

### 2. Decision Tree
- Visualisation facile des règles de décision
- Interprétable
- Gère variables catégorielles et numériques

### 3. Random Forest
- Ensemble d'arbres de décision
- Améliore la robustesse et réduit l'overfitting
- Évalue l'importance relative des variables

### 4. XGBoost / LightGBM
- Méthodes de boosting
- Performance élevée sur datasets déséquilibrés
- Rapides et efficaces

### 5. Support Vector Machine (SVM)
- Modèle à noyau efficace
- Détecte des frontières complexes entre classes

## 📊 Métriques d'Évaluation

Les modèles sont évalués sur:
- **Accuracy:** Précision globale
- **Precision:** Précision sur la classe churn
- **Recall:** Taux de détection des churners
- **F1-Score:** Moyenne harmonique de precision et recall
- **ROC-AUC:** Aire sous la courbe ROC

## 🔍 Résultats

Les résultats sont disponibles dans:
- **MLflow UI:** Interface web pour comparer les expérimentations
- **reports/:** Dossier contenant les visualisations et rapports CSV

En plus, le projet sauvegarde maintenant les résultats sur disque:
- **outputs/runs/**: métriques + artifacts par run

## 📝 Recommandations Business

Basées sur l'analyse des features importantes et les résultats des modèles:

1. **Focus sur les clients à contrat mensuel:** Plus susceptibles de churner
2. **Surveiller les frais mensuels élevés:** Corrélés avec le churn
3. **Améliorer le support technique:** Clients sans TechSupport churnent plus
4. **Encourager les contrats annuels:** Réduisent significativement le churn
5. **Cibler les nouveaux clients:** Faible tenure = risque élevé

## 🛠️ Technologies Utilisées

- **Python 3.8+**
- **Scikit-learn:** Machine Learning
- **XGBoost & LightGBM:** Boosting algorithms
- **MLflow:** Tracking des expérimentations
- **Pandas & NumPy:** Manipulation de données
- **Matplotlib & Seaborn:** Visualisation
- **Imbalanced-learn:** Gestion du déséquilibre
- **Flask:** Interface web pour l'inférence

## 📚 Références

- Dataset: [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- MLflow Documentation: https://mlflow.org/docs/latest/index.html

## 🌐 Déploiement / Inférence Web (Flask)

### 1) Lancer en local

Assurez-vous d'avoir déjà:
- `models/best_model.pkl`
- `outputs/preprocessed/scaler.pkl`
- `outputs/preprocessed/label_encoders.pkl`

Puis:

```bash
python app.py
```

Ouvrir:
- http://127.0.0.1:5000/

### 2) Lancer en production (Waitress)

```bash
python -m waitress --host=0.0.0.0 --port=5000 app:app
```

### 3) API JSON (intégration)

Endpoint:
- `POST /predict`

Body JSON:

```json
{
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
  "TotalCharges": 845.5
}
```

## 👤 Auteur

**Otmane ZIRARI**  
Faculté des Sciences Semlalia de Marrakech

## 📄 Licence

Ce projet est réalisé dans le cadre académique.

