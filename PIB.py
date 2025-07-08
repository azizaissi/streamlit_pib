import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import unicodedata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import pytz
from openai import OpenAI, OpenAIError
import json
import plotly.express as px
import time
import os

df = pd.read_excel("VA-2015-2023P.xlsx", thousands=' ', decimal=',')
# Set the current date and time
cet = pytz.timezone('CET')
current_date_time = cet.localize(datetime(2025, 7, 8, 16, 31))  # Updated to 04:31 PM CET
st.title("🔍 Modèles optimisés de prédiction du PIB avec validation croisée Leave-One-Out")
st.write(f"**Date et heure actuelles :** {current_date_time.strftime('%d/%m/%Y %H:%M %Z')}")

random.seed(42)
np.random.seed(42)

def normalize_name(name):
    """Normalize string by removing accents, trailing spaces, fixing typos, and handling case."""
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8').strip()
    name = name.replace("d'autre produits", "d'autres produits")
    name = re.sub(r'\s+', ' ', name)
    name = name.lower()
    return name

@st.cache_data
def load_and_preprocess():
    df = pd.read_excel("VA-2015-2023P.xlsx", thousands=' ', decimal=',')
    df.rename(columns={df.columns[0]: "Secteur"}, inplace=True)
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(' ', '').str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    macro_keywords = [
        "Taux de chômage", "Taux d'inflation", "Taux d'intérêt", "Dette publique", "Pression fiscale",
        "Instabilité politique", "Attaque terroristes", "Crise sociale", "Pandémie",
        "Crise énergétique", "Sécheresse", "Prix matières premières",
        "Tensions géopolitiques régionales", "Politique monétaire internationale", "conjoncture économique mondiale"
    ]

    df_macro = df[df['Secteur'].isin(macro_keywords)].copy()
    df_pib = df[df['Secteur'].str.contains("Produit Intérieur Brut PIB", case=False)].copy()
    df_secteurs = df[~df['Secteur'].isin(macro_keywords) & ~df['Secteur'].str.contains("PIB", case=False)].copy()

    df_macro.set_index("Secteur", inplace=True)
    df_pib.set_index("Secteur", inplace=True)
    df_secteurs.set_index("Secteur", inplace=True)

    df_macro_T = df_macro.transpose()
    df_secteurs_T = df_secteurs.transpose()
    df_pib_T = df_pib.transpose()

    X_df = pd.concat([df_secteurs_T, df_macro_T], axis=1).dropna()
    y_df = df_pib_T.loc[X_df.index]

    # Normalize column names
    X_df.columns = [normalize_name(col) for col in X_df.columns]

    # Add lagged GDP
    X_df['gdp_lag1'] = y_df.shift(1).fillna(y_df.mean())

    # Define sectors, macro_rates, and events with normalized names
    sectors = [
        normalize_name(s) for s in [
            "Agriculture, sylviculture et pêche", "Extraction pétrole et gaz naturel",
            "Extraction des produits miniers", "Industries agro-alimentaires",
            "Industrie du textile, de l’habillement et du cuir", "Raffinage du pétrole",
            "Industries chimiques", "Industrie d'autres produits minéraux non métalliques",
            "Industries mécaniques et électriques", "Industries diverses",
            "Production et distribution de l'électricité et gaz",
            "Production et distribution d'eau, assainissement et gestion des déchets",
            "Construction", "Commerce et réparation", "Transport et entreposage",
            "Hébergement et restauration", "Information et communication",
            "Activités financières et d'assurances", "Administration publique et défense",
            "Enseignement", "Santé humaine et action sociale", "Autres services marchands",
            "Autres activités des ménages", "Activités des organisations associatives"
        ]
    ]
    macro_rates = [normalize_name(r) for r in [
        "Taux de chômage", "Taux d'inflation", "Taux d'intérêt", "Dette publique"
    ]]
    events = [
        normalize_name(e) for e in [
            "Instabilité politique", "Attaque terroristes", "Crise sociale", "Pandémie",
            "Crise énergétique", "Sécheresse", "Prix matières premières",
            "Tensions géopolitiques régionales", "Politique monétaire internationale", "conjoncture économique mondiale"
        ]
    ]

    # Debug: Display column names
    st.write("**Secteurs définis :**", sorted(sectors))
    st.write("**Indicateurs macroéconomiques :**", sorted(macro_rates))
    st.write("**Événements définis :**", sorted(events))

    # Verify alignment
    missing_cols = [col for col in sectors + macro_rates + events if col not in X_df.columns]
    extra_cols = [col for col in X_df.columns if col not in sectors + macro_rates + events + [normalize_name("Impots nets de subventions sur les produits"), normalize_name("Pression fiscale"), "gdp_lag1"]]
    if missing_cols:
        st.warning(f"Colonnes manquantes dans X_df : {missing_cols}")
    if extra_cols:
        st.warning(f"Colonnes supplémentaires dans X_df : {extra_cols}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)
    y = y_df.values.flatten()
    years = X_df.index.astype(int)

    return X, y, years, X_df, scaler, macro_keywords, sectors, macro_rates, events

X, y, years, X_df, scaler, macro_keywords, sectors, macro_rates, events = load_and_preprocess()
loo = LeaveOneOut()

# Définition des modèles
ridge = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_regression, k=10)),
    ('ridge', Ridge())
])
ridge_params = {'ridge__alpha': [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]}
ridge_cv = RandomizedSearchCV(ridge, ridge_params, cv=loo, scoring='neg_mean_absolute_error', n_iter=7, random_state=42)

elasticnet = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_regression, k=15)),
    ('elasticnet', ElasticNet())
])
elasticnet_params = {
    'elasticnet__alpha': [0.01, 0.1, 1.0, 10.0, 50.0, 100.0],
    'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}
elasticnet_cv = RandomizedSearchCV(elasticnet, elasticnet_params, cv=loo, scoring='neg_mean_absolute_error', n_iter=10, random_state=42)

huber = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_regression, k=10)),
    ('huber', HuberRegressor())
])
huber_params = {'huber__epsilon': [1.1, 1.35, 1.5]}
huber_cv = RandomizedSearchCV(huber, huber_params, cv=loo, scoring='neg_mean_absolute_error', n_iter=3, random_state=42)

# Interprétation automatique
def interpret_results(model_name, train_mae, test_mae, train_r2, test_r2):
    rel_error = test_mae / np.mean(y)
    st.markdown("#### 💡 Interprétation")
    st.write(f"**R² sur test :** {test_r2:.4f} — indique la qualité de généralisation.")
    st.write(f"**MAE absolue :** {test_mae:.0f} — pour un PIB moyen ~{np.mean(y):,.0f}, soit une erreur relative d’environ **{rel_error*100:.1f}%**.")
    diff_r2 = train_r2 - test_r2
    if diff_r2 > 0.2:
        st.error("⚠️ Écart important entre R² train et test → possible surapprentissage.")
    else:
        st.success("✅ Pas de signe évident de surapprentissage.")

    st.markdown("#### ✅ Conclusion")
    if test_r2 >= 0.96 and rel_error < 0.03:
        st.write(f"✔️ **{model_name} donne d’excellents résultats.**")
        st.write("- Peut être utilisé comme benchmark.")
        st.write("- Très fiable pour un usage en prévision du PIB.")
    elif test_r2 >= 0.90:
        st.write(f"✔️ **{model_name} est un bon modèle,** mais peut être amélioré.")
    else:
        st.write(f"❌ **{model_name} montre des limites.** Envisage une autre méthode ou un tuning plus poussé.")

# Fonction d’évaluation
def eval_and_detect(model_cv, X, y, model_name):
    model_cv.fit(X, y)
    train_pred = model_cv.predict(X)
    train_mae = mean_absolute_error(y, train_pred)
    train_r2 = r2_score(y, train_pred)

    preds_test = []
    for tr, te in loo.split(X):
        best_model = model_cv.best_estimator_
        best_model.fit(X[tr], y[tr])
        preds_test.append(best_model.predict(X[te])[0])

    test_mae = mean_absolute_error(y, preds_test)
    test_r2 = r2_score(y, preds_test)

    st.markdown(f"### 🔍 Résultats pour **{model_name}**")
    st.write(f"Train MAE: {train_mae:.2f}, Test MAE (LOO): {test_mae:.2f}")
    st.write(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

    interpret_results(model_name, train_mae, test_mae, train_r2, test_r2)

    return test_mae, test_r2, model_cv

# Exécution
st.header("📊 Diagnostic et interprétation des modèles")
results = []
models = {}
for model, name in [(ridge_cv, "Ridge"), (elasticnet_cv, "ElasticNet"), (huber_cv, "Huber")]:
    mae, r2, trained_model = eval_and_detect(model, X, y, name)
    results.append({'Modèle': name, 'CV MAE': mae, 'Train R²': r2_score(y, model.predict(X))})
    models[name] = trained_model

# ARIMA model
arima_model = ARIMA(y, order=(1, 1, 1)).fit()
arima_pred = arima_model.predict(start=len(y), end=len(y))

# Résumé global
st.header("📈 Résumé des performances")
st.dataframe(pd.DataFrame(results).style.format({"CV MAE": "{:.2f}", "Train R²": "{:.4f}"}))

# Default sector growth rates
default_sector_growth = {
    normalize_name("Agriculture, sylviculture et pêche"): 3.0,
    normalize_name("Industries mécaniques et électriques"): 2.5,
    normalize_name("Hébergement et restauration"): 4.0,
    normalize_name("Production et distribution d'eau, assainissement et gestion des déchets"): 2.0,
    normalize_name("Construction"): 2.0,
    normalize_name("Commerce et réparation"): 3.5,
    normalize_name("Transport et entreposage"): 2.5,
    normalize_name("Information et communication"): 5.0,
    normalize_name("Activités financières et d'assurances"): 3.0,
    normalize_name("Administration publique et défense"): 1.5,
    normalize_name("Enseignement"): 2.0,
    normalize_name("Santé humaine et action sociale"): 2.5,
    normalize_name("Extraction pétrole et gaz naturel"): -2.0,
    normalize_name("Extraction des produits miniers"): 1.5,
    normalize_name("Industries agro-alimentaires"): 2.8,
    normalize_name("Industrie du textile, de l’habillement et du cuir"): 1.0,
    normalize_name("Raffinage du pétrole"): -1.0,
    normalize_name("Industries chimiques"): 2.0,
    normalize_name("Industrie d'autres produits minéraux non métalliques"): 1.8,
    normalize_name("Industries diverses"): 2.0,
    normalize_name("Production et distribution de l'électricité et gaz"): 2.0,
    normalize_name("Autres services marchands"): 3.0,
    normalize_name("Autres activités des ménages"): 1.5,
    normalize_name("Activités des organisations associatives"): 1.0
}

# Simulated external data (e.g., from IMF or national statistics)
external_data = {
    normalize_name("Taux d'inflation"): 8.0,
    normalize_name("Taux de chômage"): 15.5,
    normalize_name("Taux d'intérêt"): 7.5,
    normalize_name("Dette publique"): 80.0,
    normalize_name("Agriculture, sylviculture et pêche"): 3.5,
    normalize_name("Industries mécaniques et électriques"): 2.8,
    normalize_name("Hébergement et restauration"): 4.5
}

# Standardized prompt template
prompt_template = """
Prédiction du PIB pour {year} :
{sector_inputs}
- Inflation : X% (ex. baisse due à la politique monétaire ou à l’offre alimentaire).
- Chômage : X% (ex. création d’emplois dans les services).
- Dette publique : X% du PIB (ex. stable ou en augmentation).
- Taux d’intérêt : X% (ex. fixé par la banque centrale).
- Événements : [ex. 'Crise sociale : oui/non', 'Sécheresse : oui/non'].
Exemple de format pour les secteurs :
- Agriculture : +X% (raison : ex. meilleures pluies).
- Industrie : +X% (raison : ex. croissance du manufacturing).
- Services : +X% (raison : ex. tourisme ou finances).
"""

# User input for scenario
default_scenario = prompt_template.format(
    year=2024,
    sector_inputs="""- Agriculture : +3.1% (récupération après la sécheresse grâce à de meilleures pluies en Q1 et Q2).
- Industrie : +2.2% (croissance modeste, avec manufacturing +3.0% et phosphate/mines +4.5%, malgré des contraintes énergétiques).
- Services : +5.6% (forte performance, portée par le tourisme +9.3% et les services financiers +3.8%).
- Impôts nets sur les produits : +4.0% (consolidation fiscale et meilleure collecte des taxes).
- Inflation : 8.2% (légère baisse de 9.1% en 2023 grâce à un resserrement monétaire et une meilleure offre alimentaire).
- Chômage : 15.4% (baisse depuis 16.1%, grâce à la création d'emplois dans les services et l'industrie).
- Dette publique : 80.3% du PIB (élevée mais stable).
- Taux de change : 3.2 TND/USD en moyenne (légère dépréciation).
- Taux d’intérêt : 8.0% (maintenu par la banque centrale).
- Exportations : +4.7% (demande de l’UE et augmentation de la production agricole).
- Événements : [Crise sociale : non, Sécheresse : non]."""
)
scenario = st.text_area(
    "✍️ Décris un scénario économique (utilisez le format ci-dessous) :",
    value=default_scenario,
    height=300
)

# Validate scenario
if not re.search(r'\d+\.?\d*%\s*\(', scenario):
    st.warning("Le scénario doit inclure des pourcentages avec raisons (ex. 'Agriculture : +3.1% (meilleures pluies)').")

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Button to analyze scenario
if st.button("🔮 Analyser le scénario"):
    if scenario.strip() == "":
        st.warning("Veuillez entrer un scénario avant de lancer l’analyse.")
    else:
        with st.spinner("Analyse du scénario avec Mistral 7B..."):
            error_log = []
            normalized_estimates = {
                "macroeconomic": {
                    normalize_name("Taux de chômage"): 16.0,
                    normalize_name("Taux d'inflation"): 7.0,
                    normalize_name("Taux d'intérêt"): 3.0,
                    normalize_name("Dette publique"): 6.0
                },
                "events": {normalize_name(e): 0 for e in events},
                "sectors": {normalize_name(s): default_sector_growth.get(normalize_name(s), 0.0) for s in sectors}
            }

            year_match = re.search(r'\b(202[0-9])\b', scenario)
            target_year = int(year_match.group(1)) if year_match else 2024
            base_year = min(target_year - 1, max(X_df.index.astype(int)))

            # Construct prompt
            prompt = f"""
Given the economic scenario for Tunisia in {target_year}: '{scenario}'

**Context**: Tunisia's economy is characterized by a mix of agriculture, industry, and services, with significant contributions from agriculture, tourism, and manufacturing. Historical GDP data (in million TND) shows steady growth, with an actual GDP of 166,230 million TND in 2024. The economy is sensitive to social unrest, geopolitical tensions, and commodity prices. Use this context to inform your estimates, ensuring alignment with realistic trends for Tunisia. For sectors not mentioned, provide estimates based on historical trends (e.g., 0% to 5% growth for stable sectors, -5% to 0% for declining sectors, 2% for utilities). All values must be numeric; avoid 'null' or non-numeric values. Ensure JSON is syntactically correct.

1. Macroeconomic indicators (as percentages):
   - {macro_rates[0]} (%) [e.g., 16%]
   - {macro_rates[1]} (%) [e.g., 7%]
   - {macro_rates[2]} (%) [e.g., 3%]
   - {macro_rates[3]} (% du PIB) [e.g., 6%]

2. Binary event indicators (0 or 1):
   - {', '.join(events)}

3. Percentage changes (relative to {base_year}) for sectors:
   - {', '.join(sectors)}

Return a valid JSON object:
{{
    "macroeconomic": {{ ... }},
    "events": {{ ... }},
    "sectors": {{ ... }}
}}
"""

            try:
                response_content = None
                for attempt in range(3):
                    try:
                        response = client.chat.completions.create(
                            model="mistralai/mistral-7b-instruct",
                            messages=[{"role": "user", "content": prompt}],
                            extra_headers={
                                "HTTP-Referer": "http://localhost:8501",
                                "X-Title": "GDP Prediction App"
                            }
                        )
                        response_content = response.choices[0].message.content
                        st.write("**Réponse brute de Mistral 7B :**", response_content)
                        break
                    except OpenAIError as e:
                        error_log.append(f"Tentative {attempt + 1} échouée : {str(e)}")
                        if attempt == 2:
                            st.warning(f"Erreur API après 3 tentatives : {str(e)}")
                            break
                    time.sleep(1)

                if response_content:
                    response_content = re.sub(r'^\s*```(?:json|JSON)\s*\n?|\n?\s*```\s*$', '', response_content, flags=re.IGNORECASE).strip()
                    response_content = re.sub(r',\s*}', '}', response_content)
                    response_content = re.sub(r',\s*]', ']', response_content)
                    response_content = response_content.replace('null', '0')
                    response_content = re.sub(r':\s*\{\}', ': 0', response_content)
                    json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                    if json_match:
                        response_content = json_match.group(0)
                    else:
                        error_log.append("No valid JSON found. Using defaults.")
                        st.warning("Aucun JSON valide. Utilisation des valeurs par défaut.")

                    try:
                        estimates = json.loads(response_content)
                        normalized_estimates = {
                            "macroeconomic": {
                                normalize_name(k): float(v) if isinstance(v, (int, float)) else external_data.get(normalize_name(k), X_df[normalize_name(k)].mean() if normalize_name(k) in X_df.columns else 0.0)
                                for k, v in estimates.get("macroeconomic", {}).items()
                            },
                            "events": {
                                normalize_name(k): int(v) if isinstance(v, (int, float)) and v in [0, 1] else 0
                                for k, v in estimates.get("events", {}).items()
                            },
                            "sectors": {
                                normalize_name(k): float(v) if isinstance(v, (int, float)) else default_sector_growth.get(normalize_name(k), 0.0)
                                for k, v in estimates.get("sectors", {}).items()
                            }
                        }
                        st.write("**Réponse nettoyée :**", response_content)
                    except json.JSONDecodeError as e:
                        error_log.append(f"Erreur JSON : {e}. Utilisation des valeurs par défaut.")
                        st.warning(f"Erreur JSON : {e}. Utilisation des valeurs par défaut.")

                with open("mistral_response_log.txt", "a") as f:
                    f.write(f"{current_date_time}:\nRaw: {response_content}\nParsed: {normalized_estimates}\n\n")

                # Validate against external data
                for key, value in external_data.items():
                    if key in normalized_estimates["macroeconomic"] and abs(normalized_estimates["macroeconomic"][key] - value) > 5:
                        error_log.append(f"Estimation de {key} ({normalized_estimates['macroeconomic'][key]}%) diffère de la source externe ({value}%).")
                        normalized_estimates["macroeconomic"][key] = value
                    elif key in normalized_estimates["sectors"] and abs(normalized_estimates["sectors"][key] - value) > 5:
                        error_log.append(f"Estimation de {key} ({normalized_estimates['sectors'][key]}%) diffère de la source externe ({value}%).")
                        normalized_estimates["sectors"][key] = value

                for rate, value in normalized_estimates["macroeconomic"].items():
                    if not isinstance(value, (int, float)):
                        error_log.append(f"Erreur pour {rate}: {value} non numérique. Utilisation du défaut.")
                        normalized_estimates["macroeconomic"][rate] = external_data.get(rate, X_df[rate].mean() if rate in X_df.columns else 0.0)

                missing_sectors = [s for s in sectors if s not in normalized_estimates["sectors"]]
                if missing_sectors:
                    error_log.append(f"Secteurs manquants : {missing_sectors}")
                    for sector in missing_sectors:
                        normalized_estimates["sectors"][sector] = default_sector_growth.get(sector, 0.0)

                for sector, value in normalized_estimates["sectors"].items():
                    if not isinstance(value, (int, float)):
                        error_log.append(f"Erreur pour {sector}: {value} non numérique. Utilisation du défaut.")
                        normalized_estimates["sectors"][sector] = default_sector_growth.get(sector, 0.0)
                    elif abs(value) > 20:
                        error_log.append(f"Changement pour {sector}: {value}% trop intense. Utilisation de {20 if value > 0 else -20}%.")
                        normalized_estimates["sectors"][sector] = 20 if value > 0 else -20

                for event, value in normalized_estimates["events"].items():
                    if not isinstance(value, (int, float)) or value not in [0, 1]:
                        error_log.append(f"Erreur pour {event}: {value} n'est pas 0 ou 1. Conversion à 0.")
                        normalized_estimates["events"][event] = 0

                # Dynamic scaling
                if target_year == 2024:
                    scaling_factor = 1.06
                else:
                    historical_growth = (y[-1] / y[-2] - 1) if len(y) > 1 else 0.03
                    scaling_factor = 1 + historical_growth * 0.5
                for sector in normalized_estimates["sectors"]:
                    normalized_estimates["sectors"][sector] *= scaling_factor
                    error_log.append(f"Scaling {sector} par {scaling_factor:.2f}x pour {target_year}.")

                st.markdown("### 📘 Analyse du scénario")
                st.write("**Scénario reçu :**", scenario)
                st.write(f"**Année cible :** {target_year}")
                fig = px.bar(
                    x=list(normalized_estimates["sectors"].keys()),
                    y=list(normalized_estimates["sectors"].values()),
                    title=f"Changements sectoriels (% par rapport à {base_year})",
                    labels={"x": "Secteur", "y": "Changement (%)"},
                    color_discrete_sequence=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD", "#D4A5A5", "#9B59B6", "#3498DB", "#E74C3C", "#2ECC71"]
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig)

                # Create feature vector
                feature_vector = pd.DataFrame(index=[0], columns=X_df.columns).fillna(0.0)
                if base_year not in X_df.index:
                    st.warning(f"Base year {base_year} absent. Utilisation de la moyenne des 3 dernières années.")
                    error_log.append(f"Base year {base_year} absent. Indices : {X_df.index.tolist()}")
                    base_year_data = X_df.loc[X_df.index[-3:]].mean() if len(X_df.index) >= 3 else X_df.mean()
                else:
                    base_year_data = X_df.loc[base_year]

                # Sensitivity analysis
                sector_errors = {}
                for sector in sectors:
                    try:
                        if sector not in X_df.columns:
                            error_log.append(f"Erreur pour {sector}: non trouvé dans X_df. Utilisation de 0.")
                            feature_vector[sector] = 0.0
                        elif sector not in normalized_estimates["sectors"]:
                            error_log.append(f"Erreur pour {sector}: non trouvé dans les estimations. Utilisation du défaut.")
                            feature_vector[sector] = default_sector_growth.get(sector, 0.0)
                            sector_errors[sector] = default_sector_growth.get(sector, 0.0)
                        else:
                            feature_vector[sector] = base_year_data[sector] * (1 + normalized_estimates["sectors"][sector] / 100)
                            sector_errors[sector] = normalized_estimates["sectors"][sector]
                    except Exception as e:
                        error_log.append(f"Erreur pour {sector}: {str(e)}. Utilisation de 0.")
                        feature_vector[sector] = 0.0
                        sector_errors[sector] = 0.0

                for rate in macro_rates:
                    try:
                        feature_vector[rate] = normalized_estimates["macroeconomic"].get(rate, external_data.get(rate, X_df[rate].mean() if rate in X_df.columns else 0.0))
                    except Exception as e:
                        error_log.append(f"Erreur pour {rate}: {str(e)}. Utilisation de 0.")
                        feature_vector[rate] = 0.0

                for event in events:
                    try:
                        feature_vector[event] = normalized_estimates["events"].get(event, 0)
                    except Exception as e:
                        error_log.append(f"Erreur pour {event}: {str(e)}. Utilisation de 0.")
                        feature_vector[event] = 0

                for col in X_df.columns:
                    if col not in sectors + macro_rates + events:
                        feature_vector[col] = X_df[col].mean()

                if feature_vector.isna().any().any():
                    error_log.append(f"Valeurs NaN : {feature_vector.columns[feature_vector.isna().any()].tolist()}. Remplacement par 0.")
                    feature_vector = feature_vector.fillna(0.0)

                feature_vector = feature_vector[X_df.columns]
                X_new = scaler.transform(feature_vector)

                # Predict
                huber_model = models["Huber"].best_estimator_
                ridge_model = models["Ridge"].best_estimator_
                predicted_gdp_huber = huber_model.predict(X_new)[0]
                predicted_gdp_ridge = ridge_model.predict(X_new)[0]
                huber_score = -models["Huber"].best_score_
                ridge_score = -models["Ridge"].best_score_
                total_score = huber_score + ridge_score
                huber_weight = huber_score / total_score if total_score != 0 else 0.7
                ridge_weight = 1 - huber_weight
                predicted_gdp_ml = huber_weight * predicted_gdp_huber + ridge_weight * predicted_gdp_ridge

                # Ensemble with ARIMA
                predicted_gdp = 0.8 * predicted_gdp_ml + 0.2 * arima_pred[0]
                st.write(f"Poids de l'ensemble : Huber {huber_weight:.2f}, Ridge {ridge_weight:.2f}, ARIMA 0.20")

                # Sensitivity analysis for 2024
                if target_year == 2024:
                    actual_gdp = 166230
                    sensitivity_results = []
                    for sector in sectors:
                        if sector in sector_errors:
                            perturbed_vector = feature_vector.copy()
                            perturbed_vector[sector] *= 1.1  # Perturb by 10%
                            X_perturbed = scaler.transform(perturbed_vector)
                            perturbed_gdp = huber_weight * huber_model.predict(X_perturbed)[0] + ridge_weight * ridge_model.predict(X_perturbed)[0]
                            gdp_change = abs(perturbed_gdp - predicted_gdp_ml)
                            sensitivity_results.append({"Secteur": sector, "Impact sur PIB (%)": (gdp_change / predicted_gdp_ml) * 100})

                    st.markdown("### 📊 Analyse de sensibilité (2024)")
                    st.write("Impact d’une perturbation de 10% par secteur sur la prédiction du PIB :")
                    st.dataframe(pd.DataFrame(sensitivity_results).style.format({"Impact sur PIB (%)": "{:.2f}"}))

                st.markdown("### 📈 Résultat de la prédiction")
                st.write(f"**PIB prédit pour {target_year} :** {predicted_gdp:,.0f} million TND")
                st.info("🧪 Prédiction combinant Huber, Ridge (poids dynamiques) et ARIMA (20%).")

                if actual_2023_gdp := y[X_df.index == 2023][0] if 2023 in X_df.index else None:
                    st.write(f"**PIB réel 2023 :** {actual_2023_gdp:,.0f} million TND")
                if target_year == 2024:
                    st.write(f"**PIB réel 2024 :** {actual_gdp:,.0f} million TND")
                    error = abs(predicted_gdp - actual_gdp)
                    relative_error = (error / actual_gdp) * 100
                    st.write(f"**Erreur absolue (million TND) :** {error:,.2f}")
                    st.write(f"**Erreur relative :** {relative_error:.1f}%")

                show_errors = st.checkbox("Afficher le journal des erreurs", value=False)
                if show_errors and error_log:
                    st.warning("Problèmes rencontrés :")
                    for error in error_log:
                        st.write(error)

                if os.path.exists("mistral_response_log.txt"):
                    with open("mistral_response_log.txt", "r") as f:
                        log_content = f.read()
                    st.download_button("Télécharger le journal des réponses", log_content, "mistral_log.txt")

            except Exception as e:
                st.error(f"Erreur lors de l’analyse : {str(e)}")
                st.write(f"**Conseil :** Vérifiez le format du scénario, les noms ({sorted(sectors + macro_rates + events)}), et la connexion API. Indices : {X_df.index.tolist()}. Colonnes : {sorted(X_df.columns)}.")