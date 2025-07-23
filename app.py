import streamlit as st
import pandas as pd
import joblib

# ---------- CONFIGURACIÓN GLOBAL ----------
st.set_page_config(page_title="Predicción de Casas", page_icon="🏡", layout="centered")

# ---------- ESTILOS CSS PERSONALIZADOS ----------
st.markdown("""
    <style>
        /* Fondo y fuente */
        body {
            background-color: #0f1117;
            color: #FFFFFF;
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #0f1117;
            padding: 2rem;
        }
        h1 {
            text-align: center;
            color: #00f7ff;
            font-size: 3em;
        }
        label {
            font-weight: bold;
            color: #e0e0e0 !important;
        }
        .stButton button {
            background-color: #00f7ff;
            color: black;
            border-radius: 12px;
            padding: 0.75em 2em;
            font-size: 1em;
            transition: all 0.3s ease-in-out;
        }
        .stButton button:hover {
            background-color: #00c2d6;
            transform: scale(1.05);
        }
        .prediction-box {
            background-color: #1c1f26;
            border-left: 5px solid #00f7ff;
            padding: 1.5em;
            margin-top: 2em;
            border-radius: 10px;
            font-size: 1.5em;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1> Predicción de Precios de Casas</h1>", unsafe_allow_html=True)

model = joblib.load('housing_model.pkl')

with st.form("form_prediccion"):
    col1, col2 = st.columns(2)

    with col1:
        longitude = st.number_input('🌍 Longitud(Entre -114 hasta -124)', -124.0, -114.0, -119.0)
        latitude = st.number_input('📍 Latitud (Entre 32 hasta 42)', 32.0, 42.0, 37.0)
        housing_median_age = st.number_input('🏗️ Edad media de la vivienda (Entre 0 y 100)', 0, 100, 20)
        total_rooms = st.number_input('🛏️ Total de habitaciones (Entre 1 y 100)', 1, 100, 20)
        total_bedrooms = st.number_input('🛌 Total de dormitorios (entre 1 y 50)', 1, 50, 5)

    with col2:
        population = st.number_input('👥 Población', 0, 10000, 1000)
        households = st.number_input('🏠 Hogares', 0, 5000, 400)
        median_income = st.number_input('💰 Ingreso medio (Entre 0 a 15$/h)', 0.0, 15.0, 3.0)
        ocean_proximity = st.selectbox('🌊 Cercanía al océano', 
                                       ['NEAR BAY', 'INLAND', '<1H OCEAN', 'NEAR OCEAN', 'ISLAND'])

    submitted = st.form_submit_button("🚀 Predecir Precio")

if submitted:
    input_data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity': [ocean_proximity]
    })

    input_data = pd.get_dummies(input_data, columns=['ocean_proximity'], drop_first=True)

    model_columns = model.feature_names_in_
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_data)

    # ---------- RESULTADO ----------
    st.markdown(f"""
        <div class='prediction-box'>
            🧠 Precio predicho: <strong>${prediction[0]:,.2f}</strong>
        </div>
    """, unsafe_allow_html=True)