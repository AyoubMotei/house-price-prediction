#  Prédiction des Prix de l'Immobilier (House Price Prediction)

## Contexte et Objectif du Projet
Ce projet a pour objectif ambitieux de prédire avec une haute précision les prix de vente de biens immobiliers résidentiels (Ames, Iowa) en s'appuyant sur pas moins de 79 variables explicatives.
 Ce projet implémente un flux complet (pipeline) de **Data Science** avec des techniques d'ingénierie des caractéristiques et d'apprentissage automatique de pointe (Stacking, Blending, XGBoost, etc.). Le but final est de s'approcher de la 1ère place d'une compétition Kaggle.

---

##  Structure et Approche (Methodology)

Le développement a été effectué de façon rigoureuse, étape par étape, tel que détaillé dans le notebook principal `Final_House_Prediction.ipynb` :

### 1. Analyse Exploratoire des Données (EDA)
- **Transformation Logarithmique** : La variable cible (`SalePrice`) est asymétrique. Une application du logarithme `log(1+x)` est appliquée pour rendre la distribution normale, réduire l'effet des très grands prix, et fiabiliser les calculs du RMSLE (Root Mean Squared Logarithmic Error).
- **Analyse Multivariée** : Recherche des relations importantes via la matrice de corrélation (par ex. le score de qualité globale `OverallQual` ou encore les surfaces habitables).
- **Catégorielles Discrètes** : Représentations graphiques de catégories de prix pour affiner l'intuition.

### 2. Ingénierie des Caractéristiques (Feature Engineering)
Créations structurées de nouvelles variables combinant l'information des existantes pour donner plus d'indices à l'algorithme :
- `TotalSF` : Surface totale (sommant le sous-sol, 1er et 2ème étage).
- `HouseAge` & `RemodAge` : Âge de la maison à sa vente, et âge de ses rénovations.
- variables d'interaction : `QualTotalSF`, `GarageRatio`, etc.
- **Target Encoding** : Substitution maligne de catégories textuelles (comme `Neighborhood`, `SaleType`) par des moyennes statistiques de prix, renforçant la puissance des modèles par arbres.

### 3. Gestion des Valeurs Aberrantes (Outliers)
- Filtrage des données "extrêmes" identifiées graphiquement (notamment les maisons de très grande superficie inhabitables ou à des prix disproportionnés) afin d'éviter qu'elles ne déséquilibrent l'apprentissage linéaire des modèles.

### 4. Pipelines de Prétraitement
Utilisation dynamique de `scikit-learn` pour transformer et préparer les données d'entrée (`ColumnTransformer`) :
- **Variables numériques** : Imputation par la médiane (SimpleImputer) suivie de Standardisation (StandardScaler).
- **Variables catégorielles** : Imputation pour boucher les trous (valeur="Missing"), puis passage au filtre binaire en mémoire via de l'encodage `One-Hot`.

### 5. Modélisation Prédictive & Validation Croisée (10-fold CV)
Entraînement extensif associant plusieurs algorithmes aux compétences complémentaires :
* **Régularisation Linéaire** : *Lasso* (pour la sélection automatique de variables), *Ridge* (pour stabiliser), et *ElasticNet* (pour l'hybridation).
* **Gradient Boosting Intégral** : Modèles d'arbres propulsés extrêmement véloces : *XGBoost*, *LightGBM*, et *CatBoost*.
* **Vote d'Équipe (Stacking & Blending)** : Optimisation globale fusionnant les capacités prédictives des différents algorithmes (par ex: une combinaison où un algorithme final réapprend les biais des précédents).

---

##  Technologies et Dépendances

Afin de pouvoir rejouer ou relancer les analyses, vous avez besoin des paquets suivants :

- `Python` 3.13
- `pandas` & `numpy` (Manipulation des données)
- `scikit-learn` (Pipelines, modèles linéaires, imputation)
- `xgboost`, `lightgbm`, `catboost` (Algorithmes d'arbres poussés)
- `matplotlib` & `seaborn` (Visualisation des données / graphiques)

**Installation des modèles avancés :**
```bash
pip install catboost xgboost lightgbm scikit-learn pandas numpy matplotlib seaborn
```

---

## Organisation des Fichiers

```text
📁 house-price-prediction/
│-- 📄 Final_House_Prediction.ipynb  # Notebook principal d'exploration et modélisation.
│-- 📄 README.md                     # Documentation du projet (ce fichier).
│-- 📄 submission_final.csv          # Fichier csv généré à soumettre sur plateforme.
│-- 📁 data/                         # Dossier contenant test.csv, train.csv.
```

---

##  Résultats Obtenus
L'intégration des techniques de traitement logarithmique, la création d'interactions (feature engineering), et l'ajustement scrupuleux des hyperparamètres d'un *Stacking Regressor* ont permis à ce pipeline de réduire drastiquement l'erreur logarithmique. L'ensemble produit une distribution prédictive extrêmement stable.

