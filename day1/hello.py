import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load dataset and train model
df = pd.read_csv('./Salary_dataset.csv')
model = LinearRegression()
model.fit(df[['YearsExperience']], df.Salary)

# Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title("Salary Prediction App")
years_experience = st.number_input("Enter years of experience", min_value=0.0, step=0.1)
if st.button("Predict Salary"):
    predicted_salary = model.predict([[years_experience]])
    st.write(f"Predicted Salary: ${predicted_salary[0]:,.2f}")

