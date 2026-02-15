"""
Script de démarrage rapide pour l'évaluation
Auteur: Otmane ZIRARI
"""

import sys
import os

# Ajouter le dossier src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from evaluate import main

if __name__ == "__main__":
    print("=" * 60)
    print("Évaluation des modèles de prédiction du Churn")
    print("=" * 60)
    main()

