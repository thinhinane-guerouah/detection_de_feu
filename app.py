import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Chargement des outils (les cerveaux)
# On utilise st.cache_resource pour ne pas recharger les modèles à chaque clic
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('mon_scaler.pkl')
        classif_model = joblib.load('mon_modele_classification.pkl')
        poly_features = joblib.load('mon_poly_features.pkl')
        reg_model = joblib.load('mon_modele_regression.pkl')
    except FileNotFoundError:
        st.error("Fichiers .pkl manquants. Relancez d'abord `python mon_script_ml.py` pour régénérer les modèles (incluant mon_poly_features.pkl).")
        st.stop()
    return scaler, classif_model, poly_features, reg_model

scaler, classif_model, poly_features, reg_model = load_models()

# 2. Titre et Description de l'app
st.title("🔥 Prédiction des Feux de Forêt (Algérie)")
st.write("""
Cette application utilise l'Intelligence Artificielle pour :
1. **Détecter** s'il y a un risque de feu.
2. **Estimer** la gravité potentielle (Indice FWI).
""")

st.write("---")

# 3. Formulaire pour l'utilisateur (Barre latérale)
st.sidebar.header("Paramètres Météo")

# Rappel : Tes features étaient ['Temperature', 'RH', 'Ws', 'Rain']
temp = st.sidebar.slider("Température (°C)", 0, 34, 30)
rh = st.sidebar.slider("Humidité Relative (%)", 40, 100, 50)
ws = st.sidebar.slider("Vitesse du vent (km/h)", 0, 30, 15)
rain = st.sidebar.number_input("Pluie (cm)", 0.0, 20.0, 0.0, step=0.1)

# Création du bouton pour lancer le calcul
if st.sidebar.button("Lancer la prédiction"):

    # 4. Préparation des données
    # On met les données dans le même format que lors de l'entraînement
    # Conserver les noms de features pour éviter l'avertissement du scaler
    input_df = pd.DataFrame([[temp, rh, ws, rain]], columns=['Temperature', 'RH', 'Ws', 'Rain'])

    # On applique la mise à l'échelle (StandardScaler)
    input_data_scaled = scaler.transform(input_df)
    input_data_poly = poly_features.transform(input_data_scaled)

    # 5. Prédictions
    prediction_feu = classif_model.predict(input_data_scaled)[0] # 0 ou 1
    prediction_fwi = reg_model.predict(input_data_poly)[0]     # Valeur continue via modèle polynomial

    # 6. Affichage des résultats
    st.subheader("Résultats de l'analyse :")

    # Règle de cohérence : si FWI très élevé, on déclenche l'alerte même si le classif dit non feu
    danger_fwi = prediction_fwi >= 20
    feu_final = prediction_feu == 1 or danger_fwi

    # Colonnes pour un affichage joli
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Risque de Feu")
        if feu_final:
            st.error("ALERTE : FEU DÉTECTÉ 🔥")

        else:
            st.success("Pas de feu détecté ✅")
 
    with col2:
        st.write("### Indice de Danger (FWI)")
        st.metric(label="FWI Estimé", value=f"{prediction_fwi:.2f}")
        
        # Interprétation simple du FWI
        if prediction_fwi < 10:
            st.info("Danger Faible")
        elif prediction_fwi < 40:
            st.warning("Danger Modéré")
        else:
            st.error("Danger Élevé !")

else:
    st.info("Modifiez les paramètres à gauche et cliquez sur 'Lancer la prédiction'.")