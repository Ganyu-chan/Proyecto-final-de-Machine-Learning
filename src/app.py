
import streamlit as st
import pandas as pd
import joblib

st.title('Predicción de Precios de Casas')

model = joblib.load('housing_model.pkl')

longitude = st.number_input('Longitud', -124.0, -114.0, -119.0)
latitude = st.number_input('Latitud', 32.0, 42.0, 37.0)
housing_median_age = st.number_input('Edad media de la vivienda', 0, 100, 20)
total_rooms = st.number_input('Total de habitaciones', 0, 10000, 2000)
total_bedrooms = st.number_input('Total de dormitorios', 0, 5000, 500)
population = st.number_input('Población', 0, 10000, 1000)
households = st.number_input('Hogares', 0, 5000, 400)
median_income = st.number_input('Ingreso medio', 0.0, 15.0, 3.0)
ocean_proximity = st.selectbox('Cercanía al océano', ['NEAR BAY', 'INLAND', '<1H OCEAN', 'NEAR OCEAN', 'ISLAND'])

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
    # Añadir más variables según sea necesario
})

input_data = pd.get_dummies(input_data, columns=['ocean_proximity'], drop_first=True)

model_columns = joblib.load('housing_model.pkl').feature_names_in_
input_data = input_data.reindex(columns=model_columns, fill_value=0)

if st.button('Predecir'):
    prediction = model.predict(input_data)
    st.write(f'Precio predicho: ${prediction[0]:,.2f}')
