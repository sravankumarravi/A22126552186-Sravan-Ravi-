import streamlit as st
import pandas as pd
import pickle

# Load the model and scaler
with open('best_linear_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('data_scalar.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Set page configuration for better look
st.set_page_config(
    page_title="Social Media Usage Duration Predictor",
    layout="centered",
    initial_sidebar_state="expanded",
)

# App Title and Description
st.title('Predict The Usage Duration Of Social Media')

st.write("""
    This app predicts the usage duration of users based on their age and total likes.
    - **Age**: Age of the user in years.
    - **Total Likes**: Total number of likes received by the user.
    - **Usage Duration**: The time (in hours) a user is expected to spend on social media.
""")

# Sliders for numerical input
Age = st.slider('Age (Years)', min_value=0, max_value=70, value=17, step=1)
TotalLikes = st.slider('Total Likes', min_value=0, max_value=30, value=3, step=1)
st.write("Adjust the sliders to see how your inputs affect the predicted usage duration.")

# Predict button
if st.button('🔮 Predict Usage Duration'):
    # Prepare input data
    input_data = pd.DataFrame([[Age, TotalLikes]], columns=['Age', 'TotalLikes'])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display prediction
    predicted_duration = prediction[0] if prediction.size > 0 else 0  # Ensure it has a value
    st.success(f'🎯 **Predicted Usage Duration**: {predicted_duration} hours')
