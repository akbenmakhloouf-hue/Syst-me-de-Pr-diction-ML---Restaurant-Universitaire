"""
SYSTÈME COMPLET DE PRÉDICTION - RESTAURANT UNIVERSITAIRE
=========================================================
Utilise TOUTES les données du PDF pour un modèle fiable
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

print("=" * 80)
print(" SYSTÈME DE PRÉDICTION ML - RESTAURANT UNIVERSITAIRE")
print("=" * 80)

# ============================================================================
# ÉTAPE 1 : CHARGEMENT DES DONNÉES COMPLÈTES
# ============================================================================

print("\n📂 ÉTAPE 1 : Chargement des données...")

try:
    df = pd.read_csv('Data base (csv).csv')
    print(f"✅ Données chargées : {len(df)} lignes")

    # RENOMMER LES COLONNES
    print("\n🔧 Renommage des colonnes...")
    df.columns = df.columns.str.strip()  # Enlever espaces

    df = df.rename(columns={
        'Jours de la semane': 'Jour_Semaine',
        'Année': 'Annee',
        'jour de Ferié': 'Jour_Ferie',
        'les étudiants arrivent au Petit Déjeuner': 'Petit_Dejeuner',
        'les étudiants arrivent au Déjeuner': 'Dejeuner',
        'les étudiants arrivent au dinner': 'Diner'
    })

    print("✅ Colonnes renommées avec succès !")
    print(f"Colonnes actuelles : {df.columns.tolist()}")

except FileNotFoundError:
    print("❌ ERREUR : Fichier 'Data_base.csv' non trouvé !")
    exit()

# Vérifier les colonnes
colonnes_requises = ['Jour_Semaine', 'Mois', 'Annee', 'Jour_Ferie', 'Weekend',
                     'Petit_Dejeuner', 'Dejeuner', 'Diner']

if not all(col in df.columns for col in colonnes_requises):
    print("❌ Colonnes manquantes !")
    print(f"Colonnes trouvées : {df.columns.tolist()}")
    print(f"Colonnes requises : {colonnes_requises}")
    exit()

# ============================================================================
# ÉTAPE 2 : PRÉPARATION ET ANALYSE DES DONNÉES
# ============================================================================

print("\n📊 ÉTAPE 2 : Analyse des données...")

# Créer des features supplémentaires
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['Jour_Annee'] = df['Date'].dt.dayofyear
    df['Trimestre'] = df['Date'].dt.quarter
    df['Semaine_Annee'] = df['Date'].dt.isocalendar().week
else:
    df['Jour_Annee'] = (df['Mois'] - 1) * 30 + 15
    df['Trimestre'] = ((df['Mois'] - 1) // 3) + 1
    df['Semaine_Annee'] = df['Mois'] * 4

# Statistiques descriptives
print("\n📈 Statistiques globales :")
print("-" * 80)
stats = df[['Petit_Dejeuner', 'Dejeuner', 'Diner', 'Total']].describe()
print(stats)

# Analyser par type de jour
print("\n📊 Moyennes par type de jour :")
print("-" * 80)
print(f"Semaine  : {df[df['Weekend'] == 0]['Total'].mean():.0f} étudiants/jour")
print(f"Weekend  : {df[df['Weekend'] == 1]['Total'].mean():.0f} étudiants/jour")
if df['Jour_Ferie'].sum() > 0:
    print(f"Férié    : {df[df['Jour_Ferie'] == 1]['Total'].mean():.0f} étudiants/jour")

# ============================================================================
# ÉTAPE 3 : PRÉPARATION DES FEATURES
# ============================================================================

print("\n🔧 ÉTAPE 3 : Préparation des features...")

features = ['Jour_Semaine', 'Mois', 'Annee', 'Jour_Ferie', 'Weekend',
            'Jour_Annee', 'Trimestre', 'Semaine_Annee']

# Filtrer les lignes valides
df_clean = df[df['Total'] > 0].copy()
print(f"✅ Données nettoyées : {len(df_clean)} jours valides")

# ============================================================================
# ÉTAPE 4 : ENTRAÎNEMENT DES MODÈLES
# ============================================================================

print("\n🤖 ÉTAPE 4 : Entraînement des modèles Random Forest...")
print("-" * 80)

models = {}
metrics = {}
predictions_test = {}

for target in ['Petit_Dejeuner', 'Dejeuner', 'Diner']:
    print(f"\n🔹 Entraînement : {target}")

    X = df_clean[features]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        bootstrap=True
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    cv_scores = cross_val_score(model, X, y, cv=5,
                                scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()

    print(f"   MAE Train      : {mae_train:.2f} étudiants")
    print(f"   MAE Test       : {mae_test:.2f} étudiants")
    print(f"   RMSE Test      : {rmse_test:.2f} étudiants")
    print(f"   R² Train       : {r2_train:.3f}")
    print(f"   R² Test        : {r2_test:.3f}")
    print(f"   CV MAE (5-fold): {cv_mae:.2f} étudiants")

    models[target] = model
    metrics[target] = {
        'mae_train': mae_train,
        'mae_test': mae_test,
        'rmse_test': rmse_test,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'cv_mae': cv_mae,
        'y_test': y_test,
        'y_pred_test': y_pred_test
    }
    predictions_test[target] = (y_test, y_pred_test)

# ============================================================================
# ÉTAPE 5 : VISUALISATIONS
# ============================================================================

print("\n📊 ÉTAPE 5 : Génération des graphiques...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Performance des Modèles de Prédiction', fontsize=16, fontweight='bold')

for idx, target in enumerate(['Petit_Dejeuner', 'Dejeuner', 'Diner']):
    y_test, y_pred = predictions_test[target]

    ax1 = axes[0, idx]
    ax1.scatter(y_test, y_pred, alpha=0.6, s=50)
    ax1.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2, label='Prédiction parfaite')
    ax1.set_xlabel('Valeurs Réelles', fontsize=10)
    ax1.set_ylabel('Prédictions', fontsize=10)
    ax1.set_title(f'{target}\nMAE: {metrics[target]["mae_test"]:.1f} | R²: {metrics[target]["r2_test"]:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1, idx]
    errors = y_pred - y_test.values
    ax2.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Erreur de prédiction', fontsize=10)
    ax2.set_ylabel('Fréquence', fontsize=10)
    ax2.set_title(f'Distribution des erreurs\nMoyenne: {errors.mean():.1f} | Std: {errors.std():.1f}')
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('performance_modeles.png', dpi=300, bbox_inches='tight')
print("✅ Graphique sauvegardé : performance_modeles.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Importance des Variables (Features)', fontsize=16, fontweight='bold')

for idx, target in enumerate(['Petit_Dejeuner', 'Dejeuner', 'Diner']):
    model = models[target]
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    axes[idx].barh(importances['Feature'], importances['Importance'])
    axes[idx].set_xlabel('Importance', fontsize=10)
    axes[idx].set_title(target, fontsize=12)
    axes[idx].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('importance_features.png', dpi=300, bbox_inches='tight')
print("✅ Graphique sauvegardé : importance_features.png")

if 'Date' in df_clean.columns:
    fig, ax = plt.subplots(figsize=(18, 6))
    df_sorted = df_clean.sort_values('Date')

    ax.plot(df_sorted['Date'], df_sorted['Petit_Dejeuner'],
            label='Petit Déjeuner', marker='o', markersize=2, alpha=0.7)
    ax.plot(df_sorted['Date'], df_sorted['Dejeuner'],
            label='Déjeuner', marker='s', markersize=2, alpha=0.7)
    ax.plot(df_sorted['Date'], df_sorted['Diner'],
            label='Dîner', marker='^', markersize=2, alpha=0.7)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Nombre d\'étudiants', fontsize=12)
    ax.set_title('Évolution de la Fréquentation dans le Temps',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('evolution_temporelle.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé : evolution_temporelle.png")

# ============================================================================
# ÉTAPE 6 : SAUVEGARDE DES MODÈLES
# ============================================================================

print("\n💾 ÉTAPE 6 : Sauvegarde des modèles...")

for target, model in models.items():
    filename = f'model_{target}.pkl'
    joblib.dump(model, filename)
    print(f"✅ Modèle sauvegardé : {filename}")

metrics_df = pd.DataFrame({
    'Repas': ['Petit_Dejeuner', 'Dejeuner', 'Diner'],
    'MAE_Test': [metrics[t]['mae_test'] for t in ['Petit_Dejeuner', 'Dejeuner', 'Diner']],
    'R2_Test': [metrics[t]['r2_test'] for t in ['Petit_Dejeuner', 'Dejeuner', 'Diner']],
    'CV_MAE': [metrics[t]['cv_mae'] for t in ['Petit_Dejeuner', 'Dejeuner', 'Diner']]
})
metrics_df.to_csv('metriques_modeles.csv', index=False)
print("✅ Métriques sauvegardées : metriques_modeles.csv")

with open('features_list.txt', 'w') as f:
    f.write(','.join(features))
print("✅ Liste des features sauvegardée : features_list.txt")

# ============================================================================
# ÉTAPE 7 : FONCTION DE PRÉDICTION
# ============================================================================

print("\n🎯 ÉTAPE 7 : Test de la fonction de prédiction...")


def predire(jour_semaine, jour, mois, annee, weekend=0, jour_ferie=0):
    jour_annee = (mois - 1) * 30 + jour
    trimestre = (mois - 1) // 3 + 1
    semaine_annee = mois * 4

    X_new = pd.DataFrame([[
        jour_semaine, mois, annee, jour_ferie, weekend,
        jour_annee, trimestre, semaine_annee
    ]], columns=features)

    predictions = {}
    for target, model in models.items():
        pred = max(0, int(model.predict(X_new)[0]))
        predictions[target] = pred

    predictions['Total'] = sum(predictions.values())

    return predictions


print("\n📝 Test : Lundi 10 Février 2025")
test_pred = predire(jour_semaine=1, jour=10, mois=2, annee=2025, weekend=0, jour_ferie=0)
print(f"   Petit Déjeuner : {test_pred['Petit_Dejeuner']} étudiants")
print(f"   Déjeuner       : {test_pred['Dejeuner']} étudiants")
print(f"   Dîner          : {test_pred['Diner']} étudiants")
print(f"   TOTAL          : {test_pred['Total']} étudiants")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

print("\n" + "=" * 80)
print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
print("=" * 80)

print("\n📊 RÉSUMÉ DES PERFORMANCES :")
print("-" * 80)
for target in ['Petit_Dejeuner', 'Dejeuner', 'Diner']:
    print(f"\n{target} :")
    print(f"  • Erreur moyenne (MAE)  : ±{metrics[target]['mae_test']:.1f} étudiants")
    print(f"  • Précision (R²)        : {metrics[target]['r2_test'] * 100:.1f}%")
    print(f"  • Validation croisée    : ±{metrics[target]['cv_mae']:.1f} étudiants")

print("\n📁 FICHIERS GÉNÉRÉS :")
print("-" * 80)
print("  ✅ model_Petit_Dejeuner.pkl")
print("  ✅ model_Dejeuner.pkl")
print("  ✅ model_Diner.pkl")
print("  ✅ metriques_modeles.csv")
print("  ✅ features_list.txt")
print("  ✅ performance_modeles.png")
print("  ✅ importance_features.png")
print("  ✅ evolution_temporelle.png")

print("\n🚀 PROCHAINE ÉTAPE :")
print("-" * 80)
print("  Lancez l'application web avec : python app_web.py")

print("\n" + "=" * 80)