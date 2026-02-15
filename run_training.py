"""
Script de démarrage rapide pour l'entraînement
Auteur: Otmane ZIRARI
"""

import sys
import os

# Ajouter le dossier src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train import main

if __name__ == "__main__":
    print("=" * 60)
    print("Entraînement des modèles de prédiction du Churn")
    print("=" * 60)
    main()

