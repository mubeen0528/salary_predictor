import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.utils import save_model, load_model

st.title("💼 Salary Predictor App")
st.write("Upload a CSV file with 'YearsExperience' and 'Salary' columns to train the model.")

# Step 1: Upload CSV file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the CSV
        data = pd.read_csv(uploaded_file)

        # Check columns
        if "YearsExperience" in data.columns and "Salary" in data.columns:
            st.success("✅ CSV file loaded successfully!")

            # Step 2: Train model
            X = data[['YearsExperience']]
            y = data['Salary']
            model = LinearRegression()
            model.fit(X, y)

            # Save trained model
            save_model(model)
            st.success("✅ Model trained successfully!")

            # Step 3: Take user input for experience
            experience = st.number_input("Enter Years of Experience:", min_value=0.0, max_value=50.0, step=0.5)

            # Step 4: Predict salary when button clicked
            if st.button("Predict Salary"):
                try:
                    exp = np.array([[experience]])
                    salary = model.predict(exp)[0]
                    st.success(f"Predicted Salary: ${salary:,.2f}")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
        else:
            st.error("❌ CSV must have 'YearsExperience' and 'Salary' columns!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a CSV file to train the model.")
