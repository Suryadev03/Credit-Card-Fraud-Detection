import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Title and description
st.title("Credit Card Fraud Detection")
st.write("This application detects fraudulent credit card transactions using machine learning.")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset preview:")
    st.dataframe(data.head())

    # Display dataset information
    st.write("Dataset Information:")
    st.write(data.info())

    # Display basic statistics
    st.write("Dataset Statistics:")
    st.write(data.describe())

    # Data preprocessing
    st.subheader("Data Preprocessing")

    # Checking for the required columns
    if 'Class' in data.columns:
        st.write("Splitting data into features and target variable.")

        X = data.drop('Class', axis=1)
        y = data['Class']

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Splitting data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        st.write("Data successfully split.")

        # Model training
        st.subheader("Model Training")
        st.write("Training a Random Forest Classifier.")

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        st.write("Model training completed.")

        # Model evaluation
        st.subheader("Model Evaluation")
        y_pred = model.predict(X_test)

        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Prediction
        st.subheader("Make Predictions")
        st.write("Enter data for prediction.")

        user_input = {}
        for col in X.columns:
            user_input[col] = st.number_input(f"{col}", value=float(X[col].mean()))

        if st.button("Predict"):
            user_df = pd.DataFrame([user_input])
            user_df_scaled = scaler.transform(user_df)
            prediction = model.predict(user_df_scaled)[0]
            st.write(f"The transaction is predicted to be {'Fraudulent' if prediction == 1 else 'Non-Fraudulent'}.")
    else:
        st.error("The dataset does not contain the required 'Class' column.")
else:
    st.info("Please upload a dataset to proceed.")
