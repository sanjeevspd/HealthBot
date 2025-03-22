import streamlit as st
import pandas as pd
import numpy as np
import openai
import requests
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = "Training.csv"
df = pd.read_csv(file_path)
df = df.drop(columns=["Unnamed: 133"], errors="ignore")

# Separate features and target
X = df.drop(columns=["prognosis"])
y = df["prognosis"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y_encoded)

# Extract symptom columns
symptom_columns = X.columns.tolist()

# Streamlit App Title
st.title("🩺 Disease Prediction System with AI Chatbot")
st.write("Select the symptoms you are experiencing to predict the disease.")

# Multi-select dropdown for symptoms
selected_symptoms = st.multiselect("Select Symptoms", symptom_columns)

# Prepare input data (initialize with zeros)
input_data = np.zeros(len(symptom_columns))

# Encode selected symptoms
for symptom in selected_symptoms:
    if symptom in symptom_columns:
        input_data[symptom_columns.index(symptom)] = 1

# Predict when button is clicked
if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        # Predict disease index
        prediction = rf_model.predict([input_data])[0]

        # Convert index back to disease name
        predicted_disease = label_encoder.inverse_transform([prediction])[0]
        st.success(f"🦠 Predicted Disease: {predicted_disease}")

# AI Chatbot Integration
OPENROUTER_API_KEY = "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"
YOUR_SITE_URL = "your-site-url"  # Optional
YOUR_SITE_NAME = "your-site-name"  # Optional

st.sidebar.title("🤖 AI Medical Assistant - DeepSeek Chatbot")
st.sidebar.write("Ask me about any symptom or disease!")

user_query = st.sidebar.text_input("💬 Enter your question:")

if st.sidebar.button("Chat 💬"):
    if user_query:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": YOUR_SITE_URL,  # Optional
                    "X-Title": YOUR_SITE_NAME,  # Optional
                },
                data=json.dumps({
                    "model": "deepseek/deepseek-r1:free",
                    "messages": [{"role": "user", "content": user_query}],
                }),
            )

            # Parse response
            if response.status_code == 200:
                result = response.json()
                st.sidebar.info(result["choices"][0]["message"]["content"])
            else:
                st.sidebar.error(f"Error: {response.text}")

        except Exception as e:
            st.sidebar.error(f"Error fetching response: {e}")
    else:
        st.sidebar.warning("⚠️ Please enter a question.")
