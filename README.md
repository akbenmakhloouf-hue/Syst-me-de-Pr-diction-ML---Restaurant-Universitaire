# 🎯 Système de Prédiction ML - Restaurant Universitaire

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Description du Projet

Ce projet utilise le **Machine Learning** pour prédire la fréquentation quotidienne du restaurant universitaire lors des trois services principaux : petit-déjeuner, déjeuner et dîner. L'objectif est de garantir que chaque étudiant dispose d'un repas complet et équilibré en évitant les ruptures de stock et le gaspillage alimentaire.

### 🎯 Problématique

Les étudiants arrivent souvent au restaurant universitaire pour constater que :
- Le repas est épuisé ou la quantité disponible est insuffisante
- Cela affecte négativement leur santé physique et mentale
- La gestion des stocks est inefficace (surproduction ou sous-production)

### 💡 Solution

Notre système prédit avec précision le nombre d'étudiants attendus pour chaque service, permettant :
- Une planification optimale des stocks alimentaires
- La réduction du gaspillage
- La garantie d'un repas complet pour chaque étudiant
- Une meilleure allocation des ressources humaines

## ✨ Fonctionnalités

- 🤖 **Prédiction ML** : Algorithme Random Forest entraîné sur 400+ jours de données historiques
- 📊 **Interface Web Interactive** : Application Flask moderne et intuitive
- 📈 **Visualisations** : Graphiques de performance et d'analyse des données
- 💾 **Modèles Persistants** : Sauvegarde et chargement des modèles entraînés
- 🎨 **Design Responsive** : Compatible mobile, tablette et desktop
- ⚡ **Prédictions en Temps Réel** : Résultats instantanés via API REST
- 📅 **Gestion des Jours Spéciaux** : Prise en compte des weekends et jours fériés

## 🏗️ Architecture du Projet

```
SYSTEME-DE-PREDICTION-ML---RESTAURANT-UNIVERSITAIRE/
│
├── train_model.py              # Script d'entraînement des modèles ML
├── app_web.py                  # Application web Flask
├── Data base (csv).csv         # Dataset historique
│
├── model_Petit_Dejeuner.pkl    # Modèle ML pour petit-déjeuner
├── model_Dejeuner.pkl          # Modèle ML pour déjeuner
├── model_Diner.pkl             # Modèle ML pour dîner
├── features_list.txt           # Liste des features utilisées
├── metriques_modeles.csv       # Métriques de performance
│
├── performance_modeles.png     # Graphiques de performance
├── importance_features.png     # Importance des variables
├── evolution_temporelle.png    # Évolution dans le temps
│
├── requirements.txt            # Dépendances Python
└── README.md                   # Documentation (ce fichier)
```

## 🔧 Technologies Utilisées

- **Python 3.8+** : Langage principal
- **scikit-learn** : Algorithmes de Machine Learning
- **Flask** : Framework web
- **Pandas & NumPy** : Manipulation et analyse de données
- **Matplotlib & Seaborn** : Visualisation de données
- **Joblib** : Sérialisation des modèles


## 🚀 Utilisation

### 1. Entraîner les Modèles

Avant la première utilisation, entraînez les modèles ML :

```bash
python train_model.py
```

**Ce script va :**
- ✅ Charger et nettoyer les données historiques
- ✅ Créer les features d'entraînement
- ✅ Entraîner 3 modèles Random Forest (un par repas)
- ✅ Évaluer les performances (MAE, R², RMSE)
- ✅ Générer des visualisations
- ✅ Sauvegarder les modèles entraînés

**Sortie attendue :**
```
✅ Données chargées : 400+ lignes
✅ Modèles entraînés avec succès
✅ Fichiers générés :
   - model_Petit_Dejeuner.pkl
   - model_Dejeuner.pkl
   - model_Diner.pkl
   - performance_modeles.png
   - importance_features.png
```

### 2. Lancer l'Application Web

```bash
python app_web.py
```

**Accéder à l'interface :**
- 🌐 [http://localhost:5000/systeme-prediction-restaurant](http://localhost:5000/systeme-prediction-restaurant)

### 3. Faire une Prédiction

1. Sélectionnez la date souhaitée (jour, mois, année)
2. Choisissez le jour de la semaine
3. Cochez "Weekend" ou "Jour férié" si applicable
4. Cliquez sur **"🔮 Prédire la Fréquentation"**
5. Consultez les résultats et recommandations

## 📊 Performance des Modèles

Les modèles ont été évalués sur des données de test avec les résultats suivants :

| Repas | MAE (étudiants) | R² Score | RMSE (étudiants) |
|-------|-----------------|----------|------------------|
| Petit Déjeuner | ±15-20 | ~0.85 | ~25 |
| Déjeuner | ±20-25 | ~0.90 | ~30 |
| Dîner | ±18-22 | ~0.87 | ~28 |

**Interprétation :**
- **MAE** : Erreur moyenne absolue (plus c'est bas, mieux c'est)
- **R²** : Précision du modèle (0.85 = 85% de précision)
- **RMSE** : Erreur quadratique moyenne

## 🎨 Captures d'Écran

### Interface Principale
```
┌──────────────────────────────────────┐
│  🎯 Système de Prédiction ML        │
│  Restaurant Universitaire            │
├──────────────────────────────────────┤
│  📅 Faire une Prédiction             │
│  ┌────────────────────────────────┐  │
│  │ Jour: Lundi                    │  │
│  │ Date: 10 Février 2025          │  │
│  │ ☐ Weekend  ☐ Jour férié        │  │
│  └────────────────────────────────┘  │
│  [🔮 Prédire la Fréquentation]       │
└──────────────────────────────────────┘
```

### Résultats de Prédiction
```
┌──────────────────────────────────────┐
│  📊 TOTAL PRÉVU: 650 étudiants      │
├──────────────────────────────────────┤
│  ☕ Petit Déj: 180                   │
│  🍽️ Déjeuner: 320                    │
│  🌙 Dîner: 150                       │
├──────────────────────────────────────┤
│  💡 Recommandations:                 │
│  • Préparer 715 repas (marge 10%)   │
│  • Stock minimum: 585 repas          │
└──────────────────────────────────────┘
```

## 🔍 Variables Utilisées

Le modèle utilise les features suivantes pour ses prédictions :

| Feature | Description |
|---------|-------------|
| `Jour_Semaine` | Jour de la semaine (1=Lundi, 7=Dimanche) |
| `Mois` | Mois de l'année (1-12) |
| `Annee` | Année |
| `Jour_Ferie` | Indicateur de jour férié (0/1) |
| `Weekend` | Indicateur de weekend (0/1) |
| `Jour_Annee` | Jour de l'année (1-365) |
| `Trimestre` | Trimestre (1-4) |
| `Semaine_Annee` | Numéro de semaine dans l'année |

## 📈 Exemples d'Utilisation

### Prédiction via l'Interface Web

1. Cas d'usage : Prévoir l'affluence pour un lundi normal
   - Résultat : ~650 étudiants (180 + 320 + 150)

2. Cas d'usage : Prévoir l'affluence pour un samedi
   - Résultat : ~200 étudiants (diminution de 70%)

3. Cas d'usage : Prévoir l'affluence pour un jour férié
   - Résultat : ~100 étudiants (forte diminution)

### Prédiction via API REST

```python
import requests

url = "http://localhost:5000/api/predire"
data = {
    "jour_semaine": 1,  # Lundi
    "jour": 10,
    "mois": 2,          # Février
    "annee": 2025,
    "weekend": 0,
    "jour_ferie": 0
}

response = requests.post(url, json=data)
predictions = response.json()

print(f"Petit Déjeuner: {predictions['Petit_Dejeuner']}")
print(f"Déjeuner: {predictions['Dejeuner']}")
print(f"Dîner: {predictions['Diner']}")
print(f"Total: {predictions['Total']}")
```

## 🛠️ Configuration Avancée

### Modifier les Hyperparamètres du Modèle

Dans `train_model.py`, ajustez les paramètres du Random Forest :

```python
model = RandomForestRegressor(
    n_estimators=200,      # Nombre d'arbres
    max_depth=20,          # Profondeur maximale
    min_samples_split=3,   # Échantillons min pour split
    min_samples_leaf=2,    # Échantillons min par feuille
    random_state=42
)
```

### Personnaliser l'Interface Web

Modifiez le CSS dans `app_web.py` pour changer les couleurs, polices, etc.

## 📝 Améliorations Futures

- [ ] Ajouter la prédiction pour plusieurs jours à l'avance
- [ ] Intégrer des données météorologiques
- [ ] Implémenter d'autres algorithmes ML (XGBoost, LSTM)
- [ ] Créer un tableau de bord administrateur
- [ ] Ajouter des notifications par email
- [ ] Développer une API mobile
- [ ] Intégrer un système de feedback en temps réel


```

## 📧 Contact

**Auteur** : Akbenmakhloouf-hue

**Projet** : [SYSTEME-DE-PREDICTION-ML---RESTAURANT-UNIVERSITAIRE](https://github.com/akbenmakhloouf-hue/SYSTEME-DE-PREDICTION-ML---RESTAURANT-UNIVERSITAIRE)


## 🙏 Remerciements

- L'équipe du restaurant universitaire pour les données
- La communauté scikit-learn pour les excellents outils ML
- Tous les contributeurs et testeurs du projet
