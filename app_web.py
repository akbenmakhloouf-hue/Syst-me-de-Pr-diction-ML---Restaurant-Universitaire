"""
SYSTÈME DE PRÉDICTION - RESTAURANT UNIVERSITAIRE
================================================
Application Web Professionnelle avec Machine Learning

INSTALLATION :
pip install flask joblib pandas numpy scikit-learn

LANCEMENT :
python app_web.py

Puis ouvrir : http://localhost:5000/systeme-prediction-restaurant
"""

from flask import Flask, render_template_string, request, jsonify, redirect, url_for
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.config['APPLICATION_NAME'] = 'Système de Prédiction ML - Restaurant Universitaire'

# Charger les modèles entraînés
print("📂 Chargement des modèles...")
try:
    models = {
        'Petit_Dejeuner': joblib.load('model_Petit_Dejeuner.pkl'),
        'Dejeuner': joblib.load('model_Dejeuner.pkl'),
        'Diner': joblib.load('model_Diner.pkl')
    }

    with open('features_list.txt', 'r') as f:
        features = f.read().strip().split(',')

    print("✅ Modèles chargés avec succès !")

except FileNotFoundError:
    print("❌ ERREUR : Modèles non trouvés !")
    print("   Exécutez d'abord : python train_model.py")
    exit()

# Template HTML complet
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Système de Prédiction ML | Restaurant Universitaire</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎯</text></svg>">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            margin-bottom: 5px;
        }

        .header .app-name {
            font-size: 0.9em;
            opacity: 0.8;
            font-style: italic;
            background: rgba(255,255,255,0.2);
            padding: 8px 20px;
            border-radius: 20px;
            display: inline-block;
            margin-top: 10px;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }

        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }

        .card-title {
            font-size: 1.5em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
            font-size: 0.95em;
        }

        .form-group select,
        .form-group input[type="number"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }

        .form-group select:focus,
        .form-group input[type="number"]:focus {
            outline: none;
            border-color: #667eea;
        }

        .form-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }

        .checkbox-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }

        .checkbox-item {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .checkbox-item:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }

        .checkbox-item input {
            margin-right: 10px;
            width: 20px;
            height: 20px;
            cursor: pointer;
        }

        .checkbox-item label {
            cursor: pointer;
            font-weight: 600;
            margin: 0;
        }

        .btn-predict {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            margin-top: 20px;
            transition: transform 0.2s;
        }

        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .results {
            display: none;
        }

        .results.show {
            display: block;
        }

        .date-display {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }

        .date-display h2 {
            font-size: 1.8em;
            margin-bottom: 5px;
        }

        .date-display .badges {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 10px;
        }

        .badge {
            background: rgba(255,255,255,0.3);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }

        .total-display {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }

        .total-display .number {
            font-size: 4em;
            font-weight: bold;
            margin: 10px 0;
        }

        .total-display .label {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .meals-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }

        .meal-card {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }

        .meal-card.breakfast {
            background: linear-gradient(135deg, #FFE259 0%, #FFA751 100%);
        }

        .meal-card.lunch {
            background: linear-gradient(135deg, #FF6B6B 0%, #FFE66D 100%);
        }

        .meal-card.dinner {
            background: linear-gradient(135deg, #A8EDEA 0%, #FED6E3 100%);
        }

        .meal-card .icon {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .meal-card .number {
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
        }

        .meal-card .label {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }

        .recommendations {
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 20px;
            border-radius: 8px;
        }

        .recommendations h3 {
            color: #1976D2;
            margin-bottom: 15px;
            font-size: 1.2em;
        }

        .recommendations ul {
            list-style: none;
        }

        .recommendations li {
            padding: 8px 0;
            color: #555;
        }

        .recommendations li:before {
            content: "• ";
            color: #2196F3;
            font-weight: bold;
            margin-right: 8px;
        }

        .alert {
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            font-weight: 600;
        }

        .alert.high {
            background: #ffebee;
            border-left: 4px solid #f44336;
            color: #c62828;
        }

        .alert.low {
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            color: #2e7d32;
        }

        .stats-footer {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            text-align: center;
        }

        .stats-footer p {
            color: #666;
            margin: 5px 0;
        }

        .stats-footer .tech-badge {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            margin: 5px;
            font-size: 0.9em;
        }

        @media (max-width: 968px) {
            .main-content {
                grid-template-columns: 1fr;
            }

            .form-row,
            .meals-grid {
                grid-template-columns: 1fr;
            }

            .checkbox-group {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Système de Prédiction ML</h1>
            <p class="subtitle">Restaurant Universitaire - Prévision de Fréquentation</p>
            <p class="app-name">📊 Machine Learning | 🍽️ Gestion Optimale des Stocks</p>
        </div>

        <div class="main-content">
            <!-- Formulaire de prédiction -->
            <div class="card">
                <h2 class="card-title">Faire une Prédiction</h2>

                <form id="predictionForm">
                    <div class="form-group">
                        <label for="dayOfWeek">Jour de la semaine</label>
                        <select id="dayOfWeek" name="dayOfWeek" required>
                            <option value="1">Lundi</option>
                            <option value="2">Mardi</option>
                            <option value="3">Mercredi</option>
                            <option value="4">Jeudi</option>
                            <option value="5">Vendredi</option>
                            <option value="6">Samedi</option>
                            <option value="7">Dimanche</option>
                        </select>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label for="day">Jour</label>
                            <input type="number" id="day" name="day" min="1" max="31" value="5" required>
                        </div>

                        <div class="form-group">
                            <label for="month">Mois</label>
                            <select id="month" name="month" required>
                                <option value="1">Janvier</option>
                                <option value="2">Février</option>
                                <option value="3">Mars</option>
                                <option value="4">Avril</option>
                                <option value="5">Mai</option>
                                <option value="6">Juin</option>
                                <option value="7">Juillet</option>
                                <option value="8">Août</option>
                                <option value="9">Septembre</option>
                                <option value="10">Octobre</option>
                                <option value="11">Novembre</option>
                                <option value="12">Décembre</option>
                            </select>
                        </div>

                        <div class="form-group">
                            <label for="year">Année</label>
                            <input type="number" id="year" name="year" min="2024" max="2030" value="2025" required>
                        </div>
                    </div>

                    <div class="checkbox-group">
                        <div class="checkbox-item">
                            <input type="checkbox" id="weekend" name="weekend">
                            <label for="weekend">Weekend</label>
                        </div>

                        <div class="checkbox-item">
                            <input type="checkbox" id="holiday" name="holiday">
                            <label for="holiday">Jour férié</label>
                        </div>
                    </div>

                    <button type="submit" class="btn-predict">🔮 Prédire la Fréquentation</button>
                </form>
            </div>

            <!-- Résultats -->
            <div class="card results" id="results">
                <h2 class="card-title">Résultats de la Prédiction</h2>

                <div class="date-display">
                    <h2 id="dateDisplay">-</h2>
                    <div class="badges" id="badges"></div>
                </div>

                <div class="total-display">
                    <div class="label">TOTAL PRÉVU</div>
                    <div class="number" id="totalNumber">0</div>
                    <div class="label">étudiants</div>
                </div>

                <div class="meals-grid">
                    <div class="meal-card breakfast">
                        <div class="icon">☕</div>
                        <div class="number" id="breakfastNumber">0</div>
                        <div class="label">Petit Déjeuner</div>
                    </div>

                    <div class="meal-card lunch">
                        <div class="icon">🍽️</div>
                        <div class="number" id="lunchNumber">0</div>
                        <div class="label">Déjeuner</div>
                    </div>

                    <div class="meal-card dinner">
                        <div class="icon">🌙</div>
                        <div class="number" id="dinnerNumber">0</div>
                        <div class="label">Dîner</div>
                    </div>
                </div>

                <div class="recommendations">
                    <h3>💡 Recommandations</h3>
                    <ul id="recommendationsList"></ul>
                </div>

                <div id="alertZone"></div>
            </div>
        </div>

        <div class="stats-footer">
            <p><strong>✅ Système Prédictif Opérationnel</strong></p>
            <div style="margin: 10px 0;">
                <span class="tech-badge">🤖 Random Forest</span>
                <span class="tech-badge">📊 Scikit-learn</span>
                <span class="tech-badge">🐍 Python</span>
                <span class="tech-badge">🌐 Flask</span>
            </div>
            <p style="font-size: 0.9em; color: #999; margin-top: 10px;">
                Algorithme entraîné sur 400+ jours de données historiques
            </p>
        </div>
    </div>

    <script>
        const form = document.getElementById('predictionForm');
        const resultsDiv = document.getElementById('results');

        const jours = ['', 'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'];
        const mois = ['', 'Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 
                     'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre'];

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = new FormData(form);
            const data = {
                jour_semaine: parseInt(formData.get('dayOfWeek')),
                jour: parseInt(formData.get('day')),
                mois: parseInt(formData.get('month')),
                annee: parseInt(formData.get('year')),
                weekend: formData.get('weekend') ? 1 : 0,
                jour_ferie: formData.get('holiday') ? 1 : 0
            };

            try {
                const response = await fetch('/api/predire', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (result.error) {
                    alert('Erreur : ' + result.error);
                    return;
                }

                displayResults(result, data);

            } catch (error) {
                alert('Erreur de connexion au serveur');
                console.error(error);
            }
        });

        function displayResults(result, inputData) {
            const dateStr = `${jours[inputData.jour_semaine]} ${inputData.jour} ${mois[inputData.mois]} ${inputData.annee}`;
            document.getElementById('dateDisplay').textContent = dateStr;

            const badgesDiv = document.getElementById('badges');
            badgesDiv.innerHTML = '';

            if (inputData.weekend) {
                badgesDiv.innerHTML += '<span class="badge">🏖️ Weekend</span>';
            }
            if (inputData.jour_ferie) {
                badgesDiv.innerHTML += '<span class="badge">🎉 Jour férié</span>';
            }
            if (!inputData.weekend && !inputData.jour_ferie) {
                badgesDiv.innerHTML += '<span class="badge">📚 Semaine</span>';
            }

            document.getElementById('totalNumber').textContent = result.Total;
            document.getElementById('breakfastNumber').textContent = result.Petit_Dejeuner;
            document.getElementById('lunchNumber').textContent = result.Dejeuner;
            document.getElementById('dinnerNumber').textContent = result.Diner;

            const total = result.Total;
            const recommandations = [
                `Préparer <strong>${Math.ceil(total * 1.1)}</strong> repas (marge de sécurité 10%)`,
                `Stock minimum recommandé : <strong>${Math.ceil(total * 0.9)}</strong> repas`,
                `Stock optimal : <strong>${Math.ceil(total * 1.05)}</strong> repas (marge 5%)`
            ];

            document.getElementById('recommendationsList').innerHTML = 
                recommandations.map(r => `<li>${r}</li>`).join('');

            const alertZone = document.getElementById('alertZone');
            alertZone.innerHTML = '';

            if (total > 700) {
                alertZone.innerHTML = `
                    <div class="alert high">
                        ⚠️ ALERTE HAUTE : Forte affluence prévue<br>
                        → Prévoir personnel supplémentaire et augmenter les stocks
                    </div>
                `;
            } else if (total < 300) {
                alertZone.innerHTML = `
                    <div class="alert low">
                        ✓ AFFLUENCE FAIBLE : Ajuster les stocks pour éviter le gaspillage
                    </div>
                `;
            }

            resultsDiv.classList.add('show');
            resultsDiv.scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
"""


# Redirection de la page d'accueil
@app.route('/')
def home():
    return redirect(url_for('systeme_prediction'))


# Route principale avec nom personnalisé
@app.route('/systeme-prediction-restaurant')
def systeme_prediction():
    return render_template_string(HTML_TEMPLATE)


# API de prédiction
@app.route('/api/predire', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        jour_semaine = data['jour_semaine']
        jour = data['jour']
        mois = data['mois']
        annee = data['annee']
        weekend = data['weekend']
        jour_ferie = data['jour_ferie']

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

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 LANCEMENT DU SYSTÈME DE PRÉDICTION")
    print("=" * 70)
    print("\n📊 Application : Machine Learning - Restaurant Universitaire")
    print("\n📱 Accédez à l'interface via l'une de ces URLs :")
    print("\n   👉 http://localhost:5000")
    print("   👉 http://localhost:5000/systeme-prediction-restaurant")
    print("\n⏹️  Pour arrêter le serveur : Ctrl+C")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)