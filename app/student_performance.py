import streamlit as st
import joblib
import pandas as pd

# Load your saved model pipeline
pipeline = joblib.load('best_pipeline.pkl')

# Title
st.title("Student Performance Prediction")

# Input fields for user to enter features
parental_education = st.selectbox('Parental Level of Education', [
    'some high school', 'high school', 'some college', 'associate\'s degree', 'bachelor\'s degree', 'master\'s degree'])

gender = st.selectbox('Gender', ['female', 'male'])

race_ethnicity = st.selectbox('Race/Ethnicity', ['group A', 'group B', 'group C', 'group D', 'group E'])

lunch = st.selectbox('Lunch', ['standard', 'free/reduced'])

test_preparation = st.selectbox('Test Preparation Course', ['none', 'completed'])

math_score = st.number_input('Math Score', min_value=0, max_value=100, value=70)
reading_score = st.number_input('Reading Score', min_value=0, max_value=100, value=70)
writing_score = st.number_input('Writing Score', min_value=0, max_value=100, value=70)

# When the user clicks the button
if st.button('Predict Performance'):
    # Prepare input data for prediction
    input_df = pd.DataFrame({
        'parental level of education': [parental_education],
        'gender': [gender],
        'race/ethnicity': [race_ethnicity],
        'lunch': [lunch],
        'test preparation course': [test_preparation],
        'math score': [math_score],
        'reading score': [reading_score],
        'writing score': [writing_score]
    })

    # Predict using the loaded pipeline
    prediction = pipeline.predict(input_df)
    st.write(f"Predicted Performance: **{prediction[0]}**")
