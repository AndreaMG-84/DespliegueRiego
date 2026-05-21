
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(layout='wide')

# Set the page title
st.title('Irrigation Need Prediction App')

# Load the pre-trained models and encoders
@st.cache_resource
def load_models():
    try:
        best_bagging_classifier_model = joblib.load('best_bagging_classifier_model.joblib')
        irrigation_need_label_encoder = joblib.load('irrigation_need_label_encoder.joblib')
        crop_growth_stage_onehot_encoder = joblib.load('crop_growth_stage_onehot_encoder.joblib')
        min_max_scaler = joblib.load('min_max_scaler.joblib')
        mulching_used_label_encoder = joblib.load('mulching_used_label_encoder.joblib')
        return (
            best_bagging_classifier_model,
            irrigation_need_label_encoder,
            crop_growth_stage_onehot_encoder,
            min_max_scaler,
            mulching_used_label_encoder,
        )
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}. Make sure all .joblib files are in the correct directory.")
        st.stop()

(best_bagging_classifier_model,
 irrigation_need_label_encoder,
 crop_growth_stage_onehot_encoder,
 min_max_scaler,
 mulching_used_label_encoder) = load_models()

st.markdown('Enter the details below to predict the irrigation need:')

# Create input widgets
col1, col2, col3 = st.columns(3)

with col1:
    soil_moisture = st.slider('Soil Moisture', 0, 100, 40)
    temperature = st.slider('Temperature (°C)', 0, 50, 32)

with col2:
    wind_speed = st.slider('Wind Speed (km/h)', 0, 50, 10)
    mulching_used = st.radio('Mulching Used', ['Yes', 'No'], index=1)

with col3:
    crop_growth_stage = st.selectbox(
        'Crop Growth Stage',
        ['flowering', 'Germination', 'Growth', 'Harvest']
    )

if st.button('Predict Irrigation Need'):
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Soil_Moisture': [soil_moisture],
        'Temperature': [temperature],
        'Wind_Speed': [wind_speed],
        'Mulching_Used': [mulching_used],
        'Crop_Growth_Stage': [crop_growth_stage]
    })

    # Preprocess 'Mulching_Used'
    input_data['Mulching_Used_Encoded'] = mulching_used_label_encoder.transform(input_data['Mulching_Used'])

    # Preprocess 'Crop_Growth_Stage'
    crop_growth_stage_encoded = crop_growth_stage_onehot_encoder.transform(input_data[['Crop_Growth_Stage']])
    crop_growth_stage_df = pd.DataFrame(crop_growth_stage_encoded, columns=crop_growth_stage_onehot_encoder.get_feature_names_out(['Crop_Growth_Stage']))

    # Combine preprocessed features
    preprocessed_data = pd.concat([
        input_data[['Soil_Moisture', 'Temperature', 'Wind_Speed', 'Mulching_Used_Encoded']].reset_index(drop=True),
        crop_growth_stage_df
    ], axis=1)

    # Rename columns to match the scaler's expected feature names
    preprocessed_data = preprocessed_data.rename(columns={'Temperature': 'Temperature_C', 'Wind_Speed': 'Wind_Speed_kmh'})

    # Scale numerical features
    scaled_features = min_max_scaler.transform(preprocessed_data[['Soil_Moisture', 'Temperature_C', 'Wind_Speed_kmh']])
    preprocessed_data[['Soil_Moisture', 'Temperature_C', 'Wind_Speed_kmh']] = scaled_features

    # Make prediction
    prediction_encoded = best_bagging_classifier_model.predict(preprocessed_data)
    predicted_irrigation_need = irrigation_need_label_encoder.inverse_transform(prediction_encoded)

    st.success(f'The predicted irrigation need is: **{predicted_irrigation_need[0]}**')
